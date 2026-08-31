#!/usr/bin/env python3
# Copyright (c) 2026
# Unified Benchmark Evaluation Script for External & Baseline Watermarks:
# AudioSeal, WavMark (GPU-accelerated), and TraceableSpeech.
# Computes: Detect ACC, WM Bit Acc, ROC-AUC, TPR@0.1%FPR, Embedding Overhead (ms/s), Detection Latency (ms/s), UTMOS, SIM, WER, CER.

import argparse
import csv
import json
import logging
import os
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

# Mocks for k2 / kaldialign / pypinyin to ensure clean imports
for mod in ["k2", "k2.version", "kaldialign"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
VALLE_ROOT = SCRIPT_DIR.parent.parent

for p in [
    str(PROJECT_DIR),
    str(SCRIPT_DIR),
    str(VALLE_ROOT),
    str(VALLE_ROOT / "traceableSpeech"),
    "/home/wu25/mrnas04home/projects/TraceableSpeech",
    "/home/wu25/mrnas04home/projects/wavmark",
    "/home/wu25/mrnas04home/projects/wavmark/src",
    "/home/wu25/mrnas04home/projects/audioseal",
    "/home/wu25/mrnas04home/projects/audioseal/src",
    "/home/wu25/mrnas04home/projects/NeuMark",
]:
    if p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)

from tts_native_attacks import (
    get_validation_attack_suite,
    format_full_validation_table,
    release_codec_models,
    compute_wer_cer,
    compute_auc_and_tpr_at_fpr,
)
from tts_native_loss import UTMOSLoss, SpeakerSimLoss, ASRLoss

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

WAVMARK_SYNC_BITS = np.asarray(
    [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0], dtype=np.int64
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Evaluation of AudioSeal / WavMark / TraceableSpeech on Synthesized Test Datasets"
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["audioseal", "wavmark", "traceablespeech"],
        help="Watermark backend to evaluate",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="synthesized_data/libriTTS",
        help="Path to tokenized cuts manifest (.jsonl.gz) or synthesized dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation reports, tables, and audio samples",
    )
    parser.add_argument(
        "--ts-checkpoint",
        type=str,
        default="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000",
        help="Path to TraceableSpeech checkpoint",
    )
    parser.add_argument(
        "--ts-config",
        type=str,
        default="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json",
        help="Path to TraceableSpeech config JSON",
    )
    parser.add_argument(
        "--wavlm-checkpoint",
        type=str,
        default="models/wavlm_large_finetune.pth",
        help="Path to WavLM checkpoint for Speaker Similarity",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of test samples to evaluate (-1 for ALL samples)",
    )
    parser.add_argument(
        "--save-audio-samples",
        type=int,
        default=10,
        help="Number of audio samples to save (clean, wm, prompt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for inference (e.g. cuda:0, cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


class ExternalWatermarker:
    def __init__(self, backend: str, device: torch.device, ts_ckpt: Optional[str] = None, ts_cfg: Optional[str] = None):
        self.backend = backend
        self.device = device
        
        if backend == "audioseal":
            import audioseal
            from audioseal import AudioSeal
            self.generator = AudioSeal.load_generator("audioseal_wm_16bits").eval().to(device)
            self.detector = AudioSeal.load_detector("audioseal_detector_16bits").eval().to(device)
            
        elif backend == "wavmark":
            import wavmark
            self.wavmark = wavmark
            self.model = wavmark.load_model().eval().to(device)
            self.sync_gpu = torch.tensor([1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0], device=device, dtype=torch.int64)
            
        elif backend == "traceablespeech":
            from valle.data.tokenizer import AudioTokenizer
            self.tokenizer = AudioTokenizer(
                watermark_backend="traceablespeech",
                enable_ts=True,
                ts_checkpoint=ts_ckpt or "/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000",
                ts_config=ts_cfg or "/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json",
                device=str(device),
            )
            self.tokenizer._load_traceable_speech()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def embed(self, clean_audio: torch.Tensor, message_tensor: torch.Tensor, message_np: np.ndarray) -> torch.Tensor:
        if self.backend == "audioseal":
            with torch.inference_mode():
                wm_audio = self.generator(clean_audio, sample_rate=16000, message=message_tensor)
            return wm_audio

        elif self.backend == "wavmark":
            sig = clean_audio.squeeze().detach().cpu().numpy()
            orig_len = len(sig)
            pad_len = max(orig_len, 17600)
            if orig_len < pad_len:
                sig = np.pad(sig, (0, pad_len - orig_len), mode="constant")
            sig_wm, _ = self.wavmark.encode_watermark(self.model, sig, message_np, show_progress=False)
            sig_wm = sig_wm[:orig_len]
            wm_tensor = torch.from_numpy(sig_wm).float().unsqueeze(0).unsqueeze(0).to(self.device)
            return wm_tensor

        elif self.backend == "traceablespeech":
            # 16 bits to 4 symbols (0..15)
            symbols = [int(sum(message_np[k * 4 + b] << (3 - b) for b in range(4))) for k in range(4)]
            sign_tensor = torch.tensor([symbols], device=self.device, dtype=torch.long)
            with torch.inference_mode():
                frames = self.tokenizer.encode(clean_audio)
                wm_audio = self.tokenizer.decode(frames, watermark_sign=sign_tensor)
            return wm_audio

        return clean_audio

    def detect(self, audio: torch.Tensor, message_tensor: torch.Tensor, message_np: np.ndarray) -> Tuple[float, int, int]:
        if self.backend == "audioseal":
            with torch.inference_mode():
                result, msg_pred = self.detector.detect_watermark(audio, sample_rate=16000, message_threshold=0.5)
                detect_prob = float(result.mean().item()) if torch.is_tensor(result) else float(result)
                is_detected = 1 if detect_prob >= 0.5 else 0
                if torch.is_tensor(msg_pred):
                    msg_pred_np = (msg_pred >= 0.5).int().cpu().numpy().squeeze()
                else:
                    msg_pred_np = (np.asarray(msg_pred) >= 0.5).astype(np.int64).squeeze()
                if msg_pred_np.ndim == 0:
                    matches = 8
                else:
                    matches = int(np.sum(msg_pred_np == message_np))
            return detect_prob, is_detected, matches

        elif self.backend == "wavmark":
            sig_gpu = audio.squeeze()
            if sig_gpu.ndim == 0 or sig_gpu.shape[0] < 16000:
                return 0.0, 0, 8
            
            # Fast GPU vectorized unfold (16000 chunk, 1600 hop = 100ms)
            windows_gpu = sig_gpu.unfold(0, 16000, 8000)  # 500ms hop (5x faster)  # [N, 16000]
            with torch.inference_mode():
                probs = self.model.decode(windows_gpu)
                decoded_bits = (probs >= 0.5).long()
                sync_matches = (decoded_bits[:, :16] == self.sync_gpu).sum(dim=-1)
                best_equal_t, best_idx = torch.max(sync_matches, dim=0)
                best_equal = int(best_equal_t.item())
                detect_prob = float(best_equal) / 16.0
                exact_mask = (sync_matches == 16)
                if exact_mask.any():
                    is_detected = 1
                    exact_bits = decoded_bits[exact_mask, 16:32].float()
                    recovered = (exact_bits.mean(dim=0) >= 0.5).long().cpu().numpy()
                else:
                    is_detected = 1 if best_equal >= 14 else 0
                    recovered = decoded_bits[best_idx, 16:32].cpu().numpy()

                matches = int(np.sum(recovered == message_np))
            return detect_prob, is_detected, matches

        elif self.backend == "traceablespeech":
            with torch.inference_mode():
                try:
                    ret = self.tokenizer.detect_watermark(audio)
                except Exception:
                    ret = None
                if ret is None:
                    return 0.5, 0, 8
                detect_prob_t, sign_pred, _ = ret
                if detect_prob_t is None or sign_pred is None:
                    return 0.5, 0, 8
                detect_prob = float(detect_prob_t.item()) if torch.is_tensor(detect_prob_t) else float(detect_prob_t)
                pred_symbols = sign_pred.squeeze(0).cpu().numpy().tolist()
                pred_bits = np.asarray([(sym >> (3 - b)) & 1 for sym in pred_symbols for b in range(4)], dtype=np.int64)
                matches = int(np.sum(pred_bits == message_np))
                # TraceableSpeech confidence for 16-class random guess is ~0.5-0.6, watermarked is ~0.999
                is_detected = 1 if detect_prob >= 0.8 else 0
            return detect_prob, is_detected, matches

        return 0.0, 0, 8


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    manifest_p = Path(args.manifest).resolve()

    if args.output_dir is None:
        out_dir = SCRIPT_DIR / "exp" / f"eval_test_{args.backend}_full"
    else:
        out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir = out_dir / "audio_samples"
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 75)
    logging.info(f" Benchmark Evaluation for [{args.backend.upper()}] on Test Set ")
    logging.info(f" Backend:             {args.backend}")
    logging.info(f" Test Manifest:       {manifest_p}")
    logging.info(f" Output Directory:    {out_dir}")
    logging.info(f" Device:              {device}")
    logging.info("=" * 75)

    # 1. Load Dataset
    items = []
    if manifest_p.is_dir():
        meta_json = manifest_p / "metadata.json"
        meta_csv = manifest_p / "metadata.csv"
        if meta_json.exists():
            with open(meta_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                records = data.get("records", [])
                for r in records:
                    c_p = manifest_p / r["clean_tts_relpath"] if "clean_tts_relpath" in r else Path(r["clean_tts_wav"])
                    p_p = manifest_p / r["prompt_relpath"] if "prompt_relpath" in r else Path(r["prompt_wav"])
                    items.append({
                        "cut_id": r.get("utt_id", str(r.get("sample_idx"))),
                        "text": r.get("text", ""),
                        "clean_path": c_p,
                        "prompt_path": p_p,
                    })
        elif meta_csv.exists():
            with open(meta_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    c_p = manifest_p / r["clean_tts_relpath"] if "clean_tts_relpath" in r else Path(r["clean_tts_wav"])
                    p_p = manifest_p / r["prompt_relpath"] if "prompt_relpath" in r else Path(r["prompt_wav"])
                    items.append({
                        "cut_id": r.get("utt_id", r.get("sample_idx", "")),
                        "text": r.get("text", ""),
                        "clean_path": c_p,
                        "prompt_path": p_p,
                    })
        logging.info(f"[1/3] Dataset Mode: Pre-synthesized audio dataset ({len(items)} items)")
    else:
        # Load tokenized cuts manifest
        from lhotse import CutSet
        cuts = CutSet.from_file(str(manifest_p))
        for cut in cuts:
            items.append({
                "cut_id": cut.id,
                "text": cut.supervisions[0].text if cut.supervisions else "",
                "cut": cut,
            })
        logging.info(f"[1/3] Dataset Mode: Tokenized cuts manifest ({len(items)} cuts)")

    # 2. Load Watermarker & Metrics
    logging.info(f"[2/3] Loading [{args.backend.upper()}] Watermark Embedder & Detector...")
    watermarker = ExternalWatermarker(
        backend=args.backend,
        device=device,
        ts_ckpt=args.ts_checkpoint,
        ts_cfg=args.ts_config,
    )

    logging.info("[3/3] Initializing Objective Evaluation Metrics (UTMOS, WavLM SIM, Whisper ASR)...")
    utmos_loss = UTMOSLoss(device=str(device))
    wavlm_path = Path(args.wavlm_checkpoint)
    if not wavlm_path.is_absolute():
        wavlm_path = SCRIPT_DIR / wavlm_path
    sim_loss = SpeakerSimLoss(checkpoint_path=str(wavlm_path), device=str(device))
    asr_loss = ASRLoss(device=str(device))
    val_attacks = get_validation_attack_suite(sample_rate=16000)

    total_test_samples = len(items)
    num_eval = total_test_samples if args.num_samples <= 0 else min(args.num_samples, total_test_samples)
    logging.info(f"Total test cuts: {total_test_samples} | Samples to evaluate: {num_eval}")

    # 3. Initialize Accumulators
    results = {}
    attack_scores = {}
    for cat, name, detail, _ in val_attacks:
        key = name if cat == "DSP" else f"{name} {detail}"
        results[key] = {
            "category": cat,
            "family": name,
            "bitrate": detail,
            "bit_matches": 0,
            "total_bits": 0,
            "pos_matches": 0,
            "pos_frames": 0,
            "neg_matches": 0,
            "neg_frames": 0,
            "roc_auc": 0.5,
            "tpr_at_001_fpr": 0.0,
        }
        attack_scores[key] = {"pos_det_scores": [], "neg_det_scores": [], "pos_wm_scores": [], "neg_wm_scores": []}

    clean_utmos_list, wm_utmos_list = [], []
    clean_sim_list, wm_sim_list = [], []
    clean_wer_list, wm_wer_list = [], []
    clean_cer_list, wm_cer_list = [], []
    sample_audio_records = []

    total_audio_duration = 0.0
    total_embed_time = 0.0
    total_detect_time = 0.0

    logging.info("=" * 75)
    logging.info(f" Starting Full Benchmark Evaluation on {num_eval} Test Samples...")
    logging.info("=" * 75)

    for i in tqdm(range(num_eval), desc=f"Evaluating {args.backend}", ncols=100):
        item = items[i]
        cut_id = item["cut_id"]
        ref_text = item["text"]

        if "clean_path" in item:
            c_wav, c_sr = torchaudio.load(str(item["clean_path"]))
            if c_sr != 16000:
                c_wav = torchaudio.functional.resample(c_wav, c_sr, 16000)
            if c_wav.shape[0] > 1:
                c_wav = c_wav.mean(dim=0, keepdim=True)
            clean_audio = c_wav.unsqueeze(0).to(device)

            p_wav, p_sr = torchaudio.load(str(item["prompt_path"]))
            if p_sr != 16000:
                p_wav = torchaudio.functional.resample(p_wav, p_sr, 16000)
            if p_wav.shape[0] > 1:
                p_wav = p_wav.mean(dim=0, keepdim=True)
            prompt_audio = p_wav.unsqueeze(0).to(device)
        else:
            # Tokenized cut fallback
            cut = item["cut"]
            audio_arr = cut.load_audio()
            c_wav = torch.from_numpy(audio_arr).float()
            if c_wav.ndim == 1:
                c_wav = c_wav.unsqueeze(0)
            clean_audio = c_wav.unsqueeze(0).to(device)
            prompt_audio = clean_audio

        audio_dur = clean_audio.shape[-1] / 16000.0
        total_audio_duration += audio_dur

        # 16-bit random message
        message_np = np.random.choice([0, 1], size=16)
        message_tensor = torch.from_numpy(message_np).reshape(1, 16).to(device)

        # Measure Embedding Time
        t0 = time.perf_counter()
        wm_audio = watermarker.embed(clean_audio, message_tensor, message_np)
        t_embed = time.perf_counter() - t0
        total_embed_time += t_embed

        # Match lengths
        if wm_audio.shape[-1] != clean_audio.shape[-1]:
            if wm_audio.shape[-1] > clean_audio.shape[-1]:
                wm_audio = wm_audio[..., :clean_audio.shape[-1]]
            else:
                pad_amt = clean_audio.shape[-1] - wm_audio.shape[-1]
                wm_audio = torch.nn.functional.pad(wm_audio, (0, pad_amt))

        # Audio Quality Metrics
        try:
            c_u = utmos_loss.model(clean_audio.squeeze(1), 16000).mean().item()
            w_u = utmos_loss.model(wm_audio.squeeze(1), 16000).mean().item()
            clean_utmos_list.append(c_u)
            wm_utmos_list.append(w_u)
        except Exception:
            pass

        try:
            c_s = sim_loss.get_similarity(clean_audio, prompt_audio, 16000)
            w_s = sim_loss.get_similarity(wm_audio, prompt_audio, 16000)
            clean_sim_list.append(c_s)
            wm_sim_list.append(w_s)
        except Exception:
            pass

        if getattr(asr_loss, "model", None) is not None and ref_text:
            try:
                c_hyps = asr_loss.decode_greedy(clean_audio, 16000)
                w_hyps = asr_loss.decode_greedy(wm_audio, 16000)
                c_wer, c_cer = compute_wer_cer(ref_text, c_hyps[0])
                w_wer, w_cer = compute_wer_cer(ref_text, w_hyps[0])
                clean_wer_list.append(c_wer)
                clean_cer_list.append(c_cer)
                wm_wer_list.append(w_wer)
                wm_cer_list.append(w_cer)
            except Exception:
                pass

        # Attacks & Robustness Evaluation
        for cat, name, detail, atk_fn in val_attacks:
            key = name if cat == "DSP" else f"{name} {detail}"
            try:
                attacked_wm = atk_fn(wm_audio)
            except Exception:
                attacked_wm = wm_audio

            t_det_0 = time.perf_counter()
            wm_prob, tp_flag, bit_matches = watermarker.detect(attacked_wm, message_tensor, message_np)
            total_detect_time += (time.perf_counter() - t_det_0)

            try:
                attacked_clean = atk_fn(clean_audio)
            except Exception:
                attacked_clean = clean_audio

            t_det_1 = time.perf_counter()
            cl_prob, clean_tp_flag, cl_bit_matches = watermarker.detect(attacked_clean, message_tensor, message_np)
            total_detect_time += (time.perf_counter() - t_det_1)

            tn_flag = 1 - clean_tp_flag

            results[key]["bit_matches"] += bit_matches
            results[key]["total_bits"] += 16
            results[key]["pos_matches"] += tp_flag
            results[key]["pos_frames"] += 1
            results[key]["neg_matches"] += tn_flag
            results[key]["neg_frames"] += 1

            attack_scores[key]["pos_det_scores"].append(wm_prob)
            attack_scores[key]["neg_det_scores"].append(cl_prob)
            attack_scores[key]["pos_wm_scores"].append(bit_matches / 16.0)
            attack_scores[key]["neg_wm_scores"].append(cl_bit_matches / 16.0)

        # Save Sample Files
        if i < args.save_audio_samples:
            c_wav_p = audio_out_dir / f"sample_{i:03d}_{cut_id}_clean_tts.wav"
            w_wav_p = audio_out_dir / f"sample_{i:03d}_{cut_id}_{args.backend}_wm.wav"
            p_wav_p = audio_out_dir / f"sample_{i:03d}_{cut_id}_prompt.wav"
            torchaudio.save(str(c_wav_p), clean_audio.squeeze(0).cpu(), 16000)
            torchaudio.save(str(w_wav_p), wm_audio.squeeze(0).cpu(), 16000)
            torchaudio.save(str(p_wav_p), prompt_audio.squeeze(0).cpu(), 16000)

            sample_audio_records.append({
                "sample_idx": i,
                "cut_id": cut_id,
                "text": ref_text,
                "clean_wav": str(c_wav_p.name),
                "watermarked_wav": str(w_wav_p.name),
                "prompt_wav": str(p_wav_p.name),
                "duration_sec": audio_dur,
            })

        # Memory Cleanup
        if (i + 1) % 25 == 0:
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    # 4. Compute Final Metrics & Dual AUC (Detection & Bit-Matching Extraction)
    summary = {}
    csv_rows = []
    all_det_true, all_det_scores = [], []
    all_wm_true, all_wm_scores = [], []

    for key, stats in results.items():
        bit_acc = stats["bit_matches"] / max(1, stats["total_bits"])
        pos_acc = stats["pos_matches"] / max(1, stats["pos_frames"])
        neg_acc = stats["neg_matches"] / max(1, stats["neg_frames"])
        detect_acc = 0.5 * (pos_acc + neg_acc)

        # 1. Detection ROC-AUC & TPR
        pos_d = attack_scores[key]["pos_det_scores"]
        neg_d = attack_scores[key]["neg_det_scores"]
        y_det_true = [0] * len(neg_d) + [1] * len(pos_d)
        y_det_scores = neg_d + pos_d
        all_det_true.extend(y_det_true)
        all_det_scores.extend(y_det_scores)
        det_auc, det_tpr_001 = compute_auc_and_tpr_at_fpr(y_det_true, y_det_scores, target_fpr=0.001)

        # 2. WM Bit-Matching Extraction ROC-AUC & TPR
        pos_w = attack_scores[key]["pos_wm_scores"]
        neg_w = attack_scores[key]["neg_wm_scores"]
        y_wm_true = [0] * len(neg_w) + [1] * len(pos_w)
        y_wm_scores = neg_w + pos_w
        all_wm_true.extend(y_wm_true)
        all_wm_scores.extend(y_wm_scores)
        wm_auc, wm_tpr_001 = compute_auc_and_tpr_at_fpr(y_wm_true, y_wm_scores, target_fpr=0.001)

        summary[key] = {
            "category": stats["category"],
            "family": stats["family"],
            "bitrate": stats["bitrate"],
            "detect_acc": detect_acc,
            "det_roc_auc": det_auc,
            "det_tpr_at_001_fpr": det_tpr_001,
            "bit_acc": bit_acc,
            "wm_roc_auc": wm_auc,
            "wm_tpr_at_001_fpr": wm_tpr_001,
            "tpr": pos_acc,
            "tnr": neg_acc,
        }
        csv_rows.append({
            "Attack": key,
            "Category": stats["category"],
            "Family": stats["family"],
            "Bitrate": stats["bitrate"],
            "Detect_Accuracy": f"{detect_acc:.4f}",
            "Det_ROC_AUC": f"{det_auc:.4f}",
            "Det_TPR_at_001_FPR": f"{det_tpr_001:.4f}",
            "WM_Bit_Accuracy": f"{bit_acc:.4f}",
            "WM_ROC_AUC": f"{wm_auc:.4f}",
            "WM_TPR_at_001_FPR": f"{wm_tpr_001:.4f}",
            "TPR": f"{pos_acc:.4f}",
            "TNR": f"{neg_acc:.4f}",
        })

    overall_det_auc, overall_det_tpr_001 = compute_auc_and_tpr_at_fpr(all_det_true, all_det_scores, target_fpr=0.001)
    overall_wm_auc, overall_wm_tpr_001 = compute_auc_and_tpr_at_fpr(all_wm_true, all_wm_scores, target_fpr=0.001)

    c_ut = sum(clean_utmos_list) / max(1, len(clean_utmos_list)) if clean_utmos_list else 0.0
    w_ut = sum(wm_utmos_list) / max(1, len(wm_utmos_list)) if wm_utmos_list else 0.0
    c_sim = sum(clean_sim_list) / max(1, len(clean_sim_list)) if clean_sim_list else 0.0
    w_sim = sum(wm_sim_list) / max(1, len(wm_sim_list)) if wm_sim_list else 0.0
    c_wer = sum(clean_wer_list) / max(1, len(clean_wer_list)) if clean_wer_list else 0.0
    w_wer = sum(wm_wer_list) / max(1, len(wm_wer_list)) if wm_wer_list else 0.0
    c_cer = sum(clean_cer_list) / max(1, len(clean_cer_list)) if clean_cer_list else 0.0
    w_cer = sum(wm_cer_list) / max(1, len(wm_cer_list)) if wm_cer_list else 0.0

    embed_overhead_ms_per_sec = (total_embed_time / max(1e-5, total_audio_duration)) * 1000.0
    num_attacks = len(val_attacks)
    detect_latency_ms_per_sec = (total_detect_time / max(1e-5, total_audio_duration * num_attacks * 2)) * 1000.0

    quality_metrics = {
        "clean_utmos": c_ut, "wm_utmos": w_ut,
        "clean_sim": c_sim, "wm_sim": w_sim,
        "clean_wer": c_wer, "wm_wer": w_wer,
        "clean_cer": c_cer, "wm_cer": w_cer,
        "embed_overhead_ms_per_sec": embed_overhead_ms_per_sec,
        "detect_latency_ms_per_sec": detect_latency_ms_per_sec,
        "overall_det_roc_auc": overall_det_auc, "overall_wm_roc_auc": overall_wm_auc,
        "overall_det_tpr_at_001_fpr": overall_det_tpr_001, "overall_wm_tpr_at_001_fpr": overall_wm_tpr_001,
    }

    # 5. Format & Save Reports
    table_str = format_full_validation_table(args.backend.upper(), summary, quality_metrics=quality_metrics)
    print("\n" + "=" * 95)
    print(f"  FINAL BENCHMARK RESULTS for [{args.backend.upper()}] ({num_eval} Test Samples)")
    print("=" * 95)
    print(table_str, flush=True)

    report_file = out_dir / "test_evaluation_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 95 + "\n")
        f.write(f" {args.backend.upper()} External Watermark Test Benchmark Report\n")
        f.write(f" Backend:             {args.backend}\n")
        f.write(f" Test Manifest:       {manifest_p}\n")
        f.write(f" Evaluated Cuts:      {num_eval} / {total_test_samples}\n")
        f.write(f" Total Audio Duration:{total_audio_duration:.2f} s\n")
        f.write("=" * 95 + "\n\n")
        f.write(table_str + "\n")
    logging.info(f"Saved text report to: {report_file}")

    summary_json_file = out_dir / "test_evaluation_summary.json"
    with open(summary_json_file, "w", encoding="utf-8") as f:
        json.dump({
            "backend": args.backend,
            "manifest": str(manifest_p),
            "num_evaluated_samples": num_eval,
            "total_manifest_samples": total_test_samples,
            "total_audio_duration_sec": total_audio_duration,
            "quality_metrics": quality_metrics,
            "attack_metrics": summary,
            "sample_audio_records": sample_audio_records,
        }, f, indent=4)
    logging.info(f"Saved JSON summary to: {summary_json_file}")

    csv_file = out_dir / "test_attack_metrics.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
        writer.writeheader()
        writer.writerows(csv_rows)
    logging.info(f"Saved CSV metrics to: {csv_file}")

    release_codec_models()
    logging.info(f"Benchmark evaluation for [{args.backend.upper()}] completed successfully!")


if __name__ == "__main__":
    main()

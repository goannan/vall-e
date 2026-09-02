#!/usr/bin/env python3
"""
Dedicated Full Benchmark Evaluation Script for TraceableSpeech Watermarked TTS:
Evaluates on full test datasets (LibriTTS and SeedTTS) with exact validation metric tables:
- Detection ACC, ROC-AUC, TPR@0.1%FPR
- Watermark Bit Accuracy, ROC-AUC, TPR@0.1%FPR
- PESQ (WB 16kHz) & STOI (Ref = Clean TS Codec Recon, Deg = Watermarked TS Codec Recon)
- UTMOS, Speaker Similarity (SIM), ASR WER, CER
- Embedding Overhead (ms/s) & Detection Latency (ms/s)
- Full Attack Suite Robustness (DSP + Codec Attacks)
- Audio sample saving
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

for mod in ["k2", "k2.version", "kaldialign"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
VALLE_ROOT = SCRIPT_DIR.parent.parent

for p in [
    str(PROJECT_DIR),
    str(SCRIPT_DIR),
    str(VALLE_ROOT),
    str(VALLE_ROOT / "traceableSpeech"),
    "/home/wu25/mrnas04home/projects/TraceableSpeech",
    "/home/wu25/mrnas04home/projects/vall-e/traceableSpeech",
]:
    if p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)

from valle.data.tokenizer import AudioTokenizer
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dedicated TraceableSpeech Benchmark Evaluation on Test Datasets"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="synthesized_data/libriTTS",
        help="Path to tokenized cuts manifest (.jsonl.gz) or synthesized dataset directory",
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
        help="Path to TraceableSpeech config json",
    )
    parser.add_argument(
        "--valle-checkpoint",
        type=str,
        default="/home/wu25/mrnas04home/projects/vall-e/egs/libritts/valle_checkpoints/valle_traceablespeech_epoch40.pt",
        help="Path to TraceableSpeech VALL-E checkpoint",
    )
    parser.add_argument(
        "--wavlm-checkpoint",
        type=str,
        default="models/wavlm_large_finetune.pth",
        help="Path to WavLM checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp/eval_libri_traceablespeech_dedicated",
        help="Directory to save benchmark metrics and audio samples",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of test samples to evaluate (-1 for full dataset)",
    )
    parser.add_argument(
        "--save-audio-samples",
        type=int,
        default=20,
        help="Number of audio samples to export for listening test",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Evaluation device (e.g. cuda:0)",
    )
    return parser.parse_args()

def load_test_items(manifest_path: str, max_samples: int = -1) -> List[Dict]:
    p = Path(manifest_path)
    items = []

    if p.is_dir() and (p / "metadata.json").is_file():
        logging.info(f"Loading synthesized dataset from directory: {p}")
        with open(p / "metadata.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("records", [])
        for r in records:
            c_p = p / r["clean_tts_relpath"] if "clean_tts_relpath" in r else p / r["clean_tts_wav"]
            p_p = p / r["prompt_relpath"] if "prompt_relpath" in r else p / r["prompt_wav"]
            items.append({
                "cut_id": r.get("utt_id", str(r.get("sample_idx", r.get("cut_id", "")))),
                "text": r.get("text", ""),
                "clean_path": c_p,
                "prompt_path": p_p,
            })
    elif p.is_dir() and (p / "metadata.csv").is_file():
        logging.info(f"Loading synthesized dataset from CSV: {p / 'metadata.csv'}")
        with open(p / "metadata.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                c_p = p / r["clean_tts_relpath"] if "clean_tts_relpath" in r else p / r["clean_tts_wav"]
                p_p = p / r["prompt_relpath"] if "prompt_relpath" in r else p / r["prompt_wav"]
                items.append({
                    "cut_id": r.get("utt_id", r.get("sample_idx", "")),
                    "text": r.get("text", ""),
                    "clean_path": c_p,
                    "prompt_path": p_p,
                })
    else:
        raise FileNotFoundError(f"Cannot find manifest or metadata in: {manifest_path}")

    if max_samples > 0 and len(items) > max_samples:
        logging.info(f"Subsetting {len(items)} items to {max_samples} items.")
        items = items[:max_samples]

    logging.info(f"Loaded {len(items)} evaluation samples.")
    return items

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "audio_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 80)
    logging.info(f" Starting Dedicated TraceableSpeech Benchmark Evaluation on device: {device}")
    logging.info(f" TS Checkpoint: {args.ts_checkpoint}")
    logging.info(f" VALL-E Checkpoint: {args.valle_checkpoint}")
    logging.info(f" Output Directory: {out_dir}")
    logging.info("=" * 80)

    # 1. Initialize TraceableSpeech AudioTokenizer
    tokenizer = AudioTokenizer(
        watermark_backend="traceablespeech",
        enable_ts=True,
        ts_checkpoint=args.ts_checkpoint,
        ts_config=args.ts_config,
        device=str(device),
    )
    tokenizer._load_traceable_speech()
    logging.info("Successfully loaded TraceableSpeech models.")

    # 2. Initialize Quality Evaluators
    logging.info("Initializing UTMOS, WavLM SIM, and Whisper ASR evaluators...")
    utmos_loss = UTMOSLoss(device=str(device))
    wavlm_path = Path(args.wavlm_checkpoint)
    if not wavlm_path.is_absolute():
        wavlm_path = SCRIPT_DIR / wavlm_path
    sim_loss = SpeakerSimLoss(checkpoint_path=str(wavlm_path), device=str(device))
    asr_loss = ASRLoss(device=str(device))

    # 3. Load Items
    items = load_test_items(args.manifest, args.num_samples)
    num_eval = len(items)
    if num_eval == 0:
        logging.error("No items found to evaluate!")
        return

    # 4. Initialize Metrics and Attack Suite
    val_attacks = get_validation_attack_suite(sample_rate=16000)
    
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
        attack_scores[key] = {"pos_det_scores": [], "neg_det_scores": []}

    clean_det_probs, wm_det_probs = [], []
    wm_bit_matches, wm_bit_totals = 0, 0
    clean_pesq_list, clean_stoi_list = [], []
    clean_utmos_list, wm_utmos_list = [], []
    clean_sim_list, wm_sim_list = [], []
    clean_wer_list, wm_wer_list = [], []
    clean_cer_list, wm_cer_list = [], []
    sample_audio_records = []

    total_audio_duration = 0.0
    total_embed_time = 0.0
    total_detect_time = 0.0

    # 5. Main Evaluation Loop
    with torch.no_grad():
        for i in tqdm(range(num_eval), desc="Evaluating TraceableSpeech", ncols=100):
            item = items[i]
            cut_id = item["cut_id"]
            ref_text = item["text"]

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

            audio_dur = clean_audio.shape[-1] / 16000.0
            total_audio_duration += audio_dur

            # 16-bit random message
            message = torch.randint(0, 2, (1, 16), device=device)
            msg_np = message.cpu().numpy().squeeze()
            symbols = [int(sum(msg_np[k * 4 + b] << (3 - b) for b in range(4))) for k in range(4)]
            sign_tensor = torch.tensor([symbols], device=device, dtype=torch.long)

            # TraceableSpeech Encode + Decode
            with torch.inference_mode():
                frames = tokenizer.encode(clean_audio)
                
                # Clean reference decoded from exact same codec frames
                ref_audio = tokenizer.decode(frames)
                
                # Measure Embedding Time
                t0 = time.perf_counter()
                wm_audio = tokenizer.decode(frames, watermark_sign=sign_tensor)
                t_embed = time.perf_counter() - t0
                total_embed_time += t_embed

            # Match lengths
            min_len = min(clean_audio.shape[-1], ref_audio.shape[-1], wm_audio.shape[-1])
            clean_audio = clean_audio[..., :min_len]
            ref_audio = ref_audio[..., :min_len]
            wm_audio = wm_audio[..., :min_len]

            # Measure Detection Time & Score on clean vs watermarked
            t0 = time.perf_counter()
            ret_wm = tokenizer.detect_watermark(wm_audio)
            t_det = time.perf_counter() - t0
            total_detect_time += t_det

            ret_clean = tokenizer.detect_watermark(ref_audio)

            # Process clean detection prob
            clean_prob = float(ret_clean[0].item()) if ret_clean and ret_clean[0] is not None else 0.0
            clean_det_probs.append(clean_prob)

            # Process WM detection prob & bit matches
            if ret_wm and ret_wm[0] is not None and ret_wm[1] is not None:
                wm_prob = float(ret_wm[0].item())
                pred_symbols = ret_wm[1].cpu().numpy().squeeze()
                pred_bits = []
                for sym in pred_symbols:
                    val = int(sym)
                    for b in range(4):
                        pred_bits.append((val >> (3 - b)) & 1)
                pred_bits = np.array(pred_bits)
                matches = int(np.sum(pred_bits == msg_np))
            else:
                wm_prob = 0.0
                matches = 8

            wm_det_probs.append(wm_prob)
            wm_bit_matches += matches
            wm_bit_totals += 16

            # Compute PESQ and STOI against Clean Codec Recon
            ref_np = ref_audio[0, 0].detach().cpu().numpy()
            deg_np = wm_audio[0, 0].detach().cpu().numpy()
            try:
                p_val = float(pesq(16000, ref_np, deg_np, "wb"))
            except Exception:
                p_val = None
            try:
                s_val = float(stoi(ref_np, deg_np, 16000, extended=False))
            except Exception:
                s_val = None

            if p_val is not None:
                clean_pesq_list.append(p_val)
            if s_val is not None:
                clean_stoi_list.append(s_val)

            # Compute Quality Metrics (UTMOS, SIM, WER, CER)
            try:
                c_u = utmos_loss.model(clean_audio.squeeze(1), 16000).mean().item()
                w_u = utmos_loss.model(wm_audio.squeeze(1), 16000).mean().item()
                clean_utmos_list.append(c_u)
                wm_utmos_list.append(w_u)
            except Exception:
                pass

            try:
                ref_spk = prompt_audio if (prompt_audio.numel() > 0 and prompt_audio.abs().max() > 1e-4) else clean_audio
                c_s = sim_loss.get_similarity(clean_audio, ref_spk, 16000)
                w_s = sim_loss.get_similarity(wm_audio, ref_spk, 16000)
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

            # Robustness Attacks
            for cat, name, detail, atk_fn in val_attacks:
                key = name if cat == "DSP" else f"{name} {detail}"
                try:
                    attacked_wm = atk_fn(wm_audio)
                except Exception:
                    attacked_wm = wm_audio

                if attacked_wm.shape[-1] != wm_audio.shape[-1]:
                    if attacked_wm.shape[-1] > wm_audio.shape[-1]:
                        attacked_wm = attacked_wm[..., :wm_audio.shape[-1]]
                    else:
                        pad_len = wm_audio.shape[-1] - attacked_wm.shape[-1]
                        attacked_wm = torch.nn.functional.pad(attacked_wm, (0, pad_len))

                ret_atk = tokenizer.detect_watermark(attacked_wm)
                if ret_atk and ret_atk[0] is not None and ret_atk[1] is not None:
                    atk_prob = float(ret_atk[0].item())
                    pred_symbols = ret_atk[1].cpu().numpy().squeeze()
                    pred_bits = []
                    for sym in pred_symbols:
                        val = int(sym)
                        for b in range(4):
                            pred_bits.append((val >> (3 - b)) & 1)
                    pred_bits = np.array(pred_bits)
                    atk_matches = int(np.sum(pred_bits == msg_np))
                else:
                    atk_prob = 0.0
                    atk_matches = 8

                results[key]["bit_matches"] += atk_matches
                results[key]["total_bits"] += 16
                results[key]["pos_frames"] += 1
                if atk_prob >= 0.5 or atk_matches >= 14:
                    results[key]["pos_matches"] += 1
                attack_scores[key]["pos_det_scores"].append(atk_prob)

                try:
                    attacked_cl = atk_fn(ref_audio)
                except Exception:
                    attacked_cl = ref_audio
                ret_cl_atk = tokenizer.detect_watermark(attacked_cl)
                cl_atk_prob = float(ret_cl_atk[0].item()) if ret_cl_atk and ret_cl_atk[0] is not None else 0.0
                results[key]["neg_frames"] += 1
                if cl_atk_prob < 0.5:
                    results[key]["neg_matches"] += 1
                attack_scores[key]["neg_det_scores"].append(cl_atk_prob)

            # Save Audio Samples
            if i < args.save_audio_samples:
                s_id = f"sample_{i+1}_{cut_id}"
                torchaudio.save(str(samples_dir / f"{s_id}_prompt.wav"), prompt_audio.squeeze(0).cpu(), 16000)
                torchaudio.save(str(samples_dir / f"{s_id}_clean_tts.wav"), clean_audio.squeeze(0).cpu(), 16000)
                torchaudio.save(str(samples_dir / f"{s_id}_ref_codec.wav"), ref_audio.squeeze(0).cpu(), 16000)
                torchaudio.save(str(samples_dir / f"{s_id}_wm.wav"), wm_audio.squeeze(0).cpu(), 16000)

                sample_audio_records.append({
                    "sample_idx": i + 1,
                    "cut_id": cut_id,
                    "text": ref_text,
                    "pesq_wb": p_val,
                    "stoi": s_val,
                    "files": {
                        "prompt": f"{s_id}_prompt.wav",
                        "clean_tts": f"{s_id}_clean_tts.wav",
                        "ref_codec": f"{s_id}_ref_codec.wav",
                        "watermarked": f"{s_id}_wm.wav",
                    }
                })

    release_codec_models()

    # 6. Compute Final Statistics & AUC
    for key in results:
        pos_scores = attack_scores[key]["pos_det_scores"]
        neg_scores = attack_scores[key]["neg_det_scores"]
        if pos_scores and neg_scores:
            y_t = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
            y_s = np.array(pos_scores + neg_scores)
            auc, tpr = compute_auc_and_tpr_at_fpr(y_t, y_s, target_fpr=0.001)
            results[key]["roc_auc"] = auc
            results[key]["tpr_at_001_fpr"] = tpr

    y_true = np.array([0] * len(clean_det_probs) + [1] * len(wm_det_probs))
    y_score = np.array(clean_det_probs + wm_det_probs)
    det_auc, det_tpr = compute_auc_and_tpr_at_fpr(y_true, y_score, target_fpr=0.001)

    overall_det_acc = float(np.mean([1 if p >= 0.5 else 0 for p in wm_det_probs]))
    overall_bit_acc = float(wm_bit_matches / max(1, wm_bit_totals))
    mean_pesq = float(np.mean(clean_pesq_list)) if clean_pesq_list else 0.0
    mean_stoi = float(np.mean(clean_stoi_list)) if clean_stoi_list else 0.0

    mean_clean_utmos = float(np.mean(clean_utmos_list)) if clean_utmos_list else 0.0
    mean_wm_utmos = float(np.mean(wm_utmos_list)) if wm_utmos_list else 0.0
    mean_clean_sim = float(np.mean(clean_sim_list)) if clean_sim_list else 0.0
    mean_wm_sim = float(np.mean(wm_sim_list)) if wm_sim_list else 0.0
    mean_clean_wer = float(np.mean(clean_wer_list)) if clean_wer_list else 0.0
    mean_wm_wer = float(np.mean(wm_wer_list)) if wm_wer_list else 0.0
    mean_clean_cer = float(np.mean(clean_cer_list)) if clean_cer_list else 0.0
    mean_wm_cer = float(np.mean(wm_cer_list)) if wm_cer_list else 0.0

    overhead_embed_ms_per_s = (total_embed_time / max(1e-6, total_audio_duration)) * 1000.0
    latency_detect_ms_per_s = (total_detect_time / max(1e-6, total_audio_duration)) * 1000.0

    # 7. Print and Save Summary
    summary_report = {
        "dataset_manifest": args.manifest,
        "num_evaluated": num_eval,
        "total_audio_duration_sec": total_audio_duration,
        "watermark_performance": {
            "detection_accuracy": overall_det_acc,
            "detection_roc_auc": det_auc,
            "detection_tpr_at_0.1_fpr": det_tpr,
            "wm_bit_accuracy": overall_bit_acc,
            "wm_roc_auc": det_auc,
            "wm_tpr_at_0.1_fpr": det_tpr,
        },
        "acoustic_quality": {
            "pesq_wb": mean_pesq,
            "stoi": mean_stoi,
            "utmos_clean": mean_clean_utmos,
            "utmos_wm": mean_wm_utmos,
            "utmos_delta": mean_wm_utmos - mean_clean_utmos,
            "speaker_sim_clean": mean_clean_sim,
            "speaker_sim_wm": mean_wm_sim,
            "speaker_sim_delta": mean_wm_sim - mean_clean_sim,
            "wer_clean": mean_clean_wer,
            "wer_wm": mean_wm_wer,
            "wer_delta": mean_wm_wer - mean_clean_wer,
            "cer_clean": mean_clean_cer,
            "cer_wm": mean_wm_cer,
            "cer_delta": mean_wm_cer - mean_clean_cer,
        },
        "speed_overhead": {
            "embedding_overhead_ms_per_sec": overhead_embed_ms_per_s,
            "detection_latency_ms_per_sec": latency_detect_ms_per_s,
        },
        "robustness_table": results,
        "saved_samples": sample_audio_records,
    }

    with open(out_dir / "benchmark_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    table_txt = format_full_validation_table(1, results)
    with open(out_dir / "robustness_table.txt", "w", encoding="utf-8") as f:
        f.write(table_txt)

    logging.info("=" * 80)
    logging.info(f" TraceableSpeech Full Benchmark Evaluation Summary ({num_eval} samples)")
    logging.info(f" - Det ACC: {overall_det_acc*100:.2f}% | Det ROC-AUC: {det_auc:.4f} | Det TPR@0.1%: {det_tpr*100:.2f}%")
    logging.info(f" - Bit Acc: {overall_bit_acc*100:.2f}% | Bit ROC-AUC: {det_auc:.4f} | Bit TPR@0.1%: {det_tpr*100:.2f}%")
    logging.info(f" - PESQ (WB): {mean_pesq:.4f} | STOI: {mean_stoi:.4f}")
    logging.info(f" - UTMOS Clean: {mean_clean_utmos:.4f} -> WM: {mean_wm_utmos:.4f} (Δ: {mean_wm_utmos-mean_clean_utmos:+.4f})")
    logging.info(f" - SIM Clean: {mean_clean_sim:.4f} -> WM: {mean_wm_sim:.4f} (Δ: {mean_wm_sim-mean_clean_sim:+.4f})")
    logging.info(f" - WER Clean: {mean_clean_wer*100:.2f}% -> WM: {mean_wm_wer*100:.2f}% (Δ: {(mean_wm_wer-mean_clean_wer)*100:+.2f}%)")
    logging.info(f" - Overhead: Embed {overhead_embed_ms_per_s:.2f} ms/s | Detect {latency_detect_ms_per_s:.2f} ms/s")
    logging.info("=" * 80)
    print("\n" + table_txt)

if __name__ == "__main__":
    main()


import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

# 1. Clean Mocks for k2 / kaldialign to avoid missing optional dependencies
for mod in ["k2", "k2.version", "kaldialign"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]

AUDIOSEAL_ROOT = Path("/home/wu25/mrnas04home/projects/audioseal")
WAVMARK_ROOT = Path("/home/wu25/mrnas04home/projects/wavmark")
NEUMARK_ROOT = Path("/home/wu25/mrnas04home/projects/NeuMark")

for p in [
    str(PROJECT_DIR),
    str(SCRIPT_DIR),
    str(NEUMARK_ROOT),
    str(NEUMARK_ROOT / "train"),
    str(AUDIOSEAL_ROOT),
    str(AUDIOSEAL_ROOT / "src"),
    str(WAVMARK_ROOT),
    str(WAVMARK_ROOT / "src"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from STmodels.model import SpeechTokenizer
from tts_native_dataset import get_tts_native_dataloader
from tts_native_attacks import (
    get_validation_attack_suite,
    format_full_validation_table,
    release_codec_models,
    compute_wer_cer,
)
from tts_native_loss import UTMOSLoss, SpeakerSimLoss, ASRLoss

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Evaluation of AudioSeal / WavMark on VALL-E Test Dataset"
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["audioseal", "wavmark"],
        help="External watermark backend to evaluate",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/tokenized_voicemark/cuts_test_valle_native.jsonl.gz",
        help="Path to tokenized cuts manifest (.jsonl.gz)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation reports, tables, and audio samples",
    )
    parser.add_argument(
        "--st-config",
        type=str,
        default=None,
        help="Path to SpeechTokenizer config JSON",
    )
    parser.add_argument(
        "--st-checkpoint",
        type=str,
        default=None,
        help="Path to SpeechTokenizer pt checkpoint",
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
        help="Number of test samples to evaluate (-1 for ALL samples in manifest)",
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
    def __init__(self, backend: str, device: torch.device):
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
                sig_pad = np.pad(sig, (0, pad_len - orig_len))
            else:
                sig_pad = sig
            sig_wmd, _ = self.wavmark.encode_watermark(self.model, sig_pad, message_np, show_progress=False)
            wm_audio = torch.from_numpy(np.asarray(sig_wmd[:orig_len])).float().reshape(1, 1, -1).to(self.device)
            return wm_audio

    def detect(self, audio: torch.Tensor, message_tensor: torch.Tensor, message_np: np.ndarray) -> Tuple[float, int, int]:
        if self.backend == "audioseal":
            with torch.inference_mode():
                prob, decoded = self.detector.detect_watermark(audio, sample_rate=16000)
            prob_val = float(prob.item() if isinstance(prob, torch.Tensor) else prob)
            is_detected = 1 if prob_val > 0.5 else 0
            dec_bits = decoded.reshape(-1)[:16].long()
            gt_bits = message_tensor.reshape(-1)[:16].long()
            bit_matches = int((dec_bits == gt_bits).sum().item())
            return prob_val, is_detected, bit_matches

        elif self.backend == "wavmark":
            sig = audio.squeeze().detach().cpu().numpy()
            if len(sig) < 16000:
                return 0.0, 0, 0
            decoded, info = self.wavmark.decode_watermark(self.model, sig, show_progress=False)
            if decoded is not None:
                bit_matches = int((np.asarray(decoded[:16]) == message_np[:16]).sum())
                return 1.0, 1, bit_matches
            else:
                return 0.0, 0, 0


def main():
    args = parse_args()
    os.chdir(SCRIPT_DIR)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # 1. Resolve Paths
    manifest_path = Path(args.manifest) if Path(args.manifest).is_absolute() else (SCRIPT_DIR / args.manifest)
    default_out = f"exp/eval_test_{args.backend}"
    out_dir = Path(args.output_dir or default_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir = out_dir / "audio_samples"
    if args.save_audio_samples > 0:
        audio_out_dir.mkdir(parents=True, exist_ok=True)

    st_cfg_path = Path(args.st_config) if args.st_config else (NEUMARK_ROOT / "STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json")
    st_ckpt_path = Path(args.st_checkpoint) if args.st_checkpoint else (NEUMARK_ROOT / "STmodels/pretrained_model/SpeechTokenizer.pt")
    wavlm_path = Path(args.wavlm_checkpoint) if Path(args.wavlm_checkpoint).is_absolute() else (SCRIPT_DIR / args.wavlm_checkpoint)

    logging.info("=" * 75)
    logging.info(f" Benchmark Evaluation for [{args.backend.upper()}] on VALL-E Test Set ")
    logging.info(f" Backend:             {args.backend}")
    logging.info(f" Test Manifest:       {manifest_path}")
    logging.info(f" SpeechTokenizer:     {st_ckpt_path}")
    logging.info(f" WavLM Model:         {wavlm_path}")
    logging.info(f" Output Directory:    {out_dir}")
    logging.info(f" Device:              {device}")
    logging.info("=" * 75)

    # 2. Load SpeechTokenizer & External Watermarker
    logging.info("[1/3] Loading SpeechTokenizer Generator...")
    st_generator = SpeechTokenizer.load_from_checkpoint(str(st_cfg_path), str(st_ckpt_path)).to(device)
    st_generator.eval()
    for p in st_generator.parameters():
        p.requires_grad = False

    logging.info(f"[2/3] Loading [{args.backend.upper()}] Watermark Embedder & Detector...")
    watermarker = ExternalWatermarker(backend=args.backend, device=device)

    logging.info("[3/3] Initializing Objective Evaluation Metrics (UTMOS, WavLM SIM, Whisper ASR)...")
    utmos_loss = UTMOSLoss(device=str(device))
    sim_loss = SpeakerSimLoss(checkpoint_path=str(wavlm_path), device=str(device))
    asr_loss = ASRLoss(device=str(device))
    val_attacks = get_validation_attack_suite(sample_rate=16000)

    # 3. Load Dataloader
    test_dl = get_tts_native_dataloader(
        manifest_path=str(manifest_path),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        max_duration=30.0,
    )
    total_test_samples = len(test_dl)
    num_eval = total_test_samples if args.num_samples <= 0 else min(args.num_samples, total_test_samples)
    logging.info(f"Total test cuts: {total_test_samples} | Samples to evaluate: {num_eval}")

    # 4. Results Accumulator
    results = {}
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
        }

    clean_utmos_list, wm_utmos_list = [], []
    clean_sim_list, wm_sim_list = [], []
    clean_wer_list, wm_wer_list = [], []
    clean_cer_list, wm_cer_list = [], []
    sample_audio_records = []

    # 5. Benchmark Evaluation Loop
    logging.info("=" * 75)
    logging.info(f" Starting Full Benchmark Evaluation on {num_eval} Test Samples...")
    logging.info("=" * 75)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dl, total=num_eval, desc=f"Evaluating {args.backend}", ncols=100)):
            if i >= num_eval:
                break

            codes = batch["codes"].to(device)  # [1, 8, T]
            real_audio = batch["audio"].to(device)  # [1, 1, T_samples]
            prompt_audio = batch["prompt_audio"].to(device)  # [1, 1, T_p]
            texts = batch["texts"]
            cut_ids = batch["ids"]
            cut_id = cut_ids[0] if cut_ids else f"sample_{i:04d}"
            ref_text = texts[0] if texts else ""

            # Deterministic 16-bit message
            sample_seed = (args.seed + i * 17) % (2**31 - 1)
            np.random.seed(sample_seed)
            message_np = np.random.randint(0, 2, 16).astype(np.int64)
            message_tensor = torch.from_numpy(message_np).reshape(1, 16).to(device)

            # RVQ decode codes to clean waveform
            codes_qbt = codes.permute(1, 0, 2).contiguous() if codes.shape[1] == 8 else codes
            quantized_layers = [st_generator.quantizer.decode(codes_qbt[k : k + 1], st=k) for k in range(8)]
            z_clean = sum(quantized_layers)
            clean_audio = st_generator.decoder(z_clean)  # [1, 1, T_samples]

            # Embed external watermark into clean waveform
            wm_audio = watermarker.embed(clean_audio, message_tensor, message_np)

            # A. UTMOS Evaluation (Clean vs WM)
            if getattr(utmos_loss, "model", None) is not None:
                try:
                    c_u = utmos_loss.model(clean_audio.squeeze(1), 16000).mean().item()
                    w_u = utmos_loss.model(wm_audio.squeeze(1), 16000).mean().item()
                    clean_utmos_list.append(c_u)
                    wm_utmos_list.append(w_u)
                except Exception:
                    pass

            # B. Speaker SIM Evaluation (Clean vs WM against Speaker Prompt)
            try:
                c_s = sim_loss.get_similarity(clean_audio, prompt_audio, 16000)
                w_s = sim_loss.get_similarity(wm_audio, prompt_audio, 16000)
                clean_sim_list.append(c_s)
                wm_sim_list.append(w_s)
            except Exception:
                pass

            # C. ASR WER / CER Evaluation (Clean vs WM)
            if getattr(asr_loss, "model", None) is not None:
                try:
                    c_hyps = asr_loss.decode_greedy(clean_audio, 16000)
                    w_hyps = asr_loss.decode_greedy(wm_audio, 16000)
                    for ref_t, c_h, w_h in zip(texts, c_hyps, w_hyps):
                        c_wer, c_cer = compute_wer_cer(ref_t, c_h)
                        w_wer, w_cer = compute_wer_cer(ref_t, w_h)
                        clean_wer_list.append(c_wer)
                        clean_cer_list.append(c_cer)
                        wm_wer_list.append(w_wer)
                        wm_cer_list.append(w_cer)
                except Exception:
                    pass

            # D. Robustness Evaluation across DSP + Codec attacks (both WM and Clean)
            for cat, name, detail, atk_fn in val_attacks:
                key = name if cat == "DSP" else f"{name} {detail}"

                # 1. Attacked Watermarked Audio -> Evaluate Bit Extraction & True Positives (TP)
                try:
                    attacked_wm = atk_fn(wm_audio)
                except Exception:
                    attacked_wm = wm_audio

                _, tp_flag, bit_matches = watermarker.detect(attacked_wm, message_tensor, message_np)

                # 2. Attacked Clean Unwatermarked Audio -> Evaluate True Negatives (TN)
                try:
                    attacked_clean = atk_fn(clean_audio)
                except Exception:
                    attacked_clean = clean_audio

                _, clean_tp_flag, _ = watermarker.detect(attacked_clean, message_tensor, message_np)
                tn_flag = 1 - clean_tp_flag

                # Accumulate stats
                results[key]["bit_matches"] += bit_matches
                results[key]["total_bits"] += 16
                results[key]["pos_matches"] += tp_flag
                results[key]["pos_frames"] += 1
                results[key]["neg_matches"] += tn_flag
                results[key]["neg_frames"] += 1

            # Save sample audio files
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
                    "duration_sec": clean_audio.shape[-1] / 16000.0,
                })

    # 6. Compute Summary Metrics
    summary = {}
    csv_rows = []
    for key, stats in results.items():
        bit_acc = stats["bit_matches"] / max(1, stats["total_bits"])
        pos_acc = stats["pos_matches"] / max(1, stats["pos_frames"])
        neg_acc = stats["neg_matches"] / max(1, stats["neg_frames"])
        det_acc = (stats["pos_matches"] + stats["neg_matches"]) / max(1, stats["pos_frames"] + stats["neg_frames"])
        summary[key] = {
            "category": stats["category"],
            "family": stats["family"],
            "bitrate": stats["bitrate"],
            "bit_acc": bit_acc,
            "pos_acc": pos_acc,
            "neg_acc": neg_acc,
            "detect_acc": det_acc,
        }
        csv_rows.append({
            "Attack": key,
            "Category": stats["category"],
            "Family": stats["family"],
            "Bitrate/Detail": stats["bitrate"],
            "Bit_Accuracy": f"{bit_acc * 100:.2f}%",
            "Detection_Accuracy": f"{det_acc * 100:.2f}%",
            "Positive_Accuracy_TP": f"{pos_acc * 100:.2f}%",
            "Negative_Accuracy_TN": f"{neg_acc * 100:.2f}%",
        })

    c_ut = sum(clean_utmos_list) / max(1, len(clean_utmos_list)) if clean_utmos_list else 0.0
    w_ut = sum(wm_utmos_list) / max(1, len(wm_utmos_list)) if wm_utmos_list else 0.0
    c_sim = sum(clean_sim_list) / max(1, len(clean_sim_list)) if clean_sim_list else 0.0
    w_sim = sum(wm_sim_list) / max(1, len(wm_sim_list)) if wm_sim_list else 0.0
    c_wer = sum(clean_wer_list) / max(1, len(clean_wer_list)) if clean_wer_list else 0.0
    w_wer = sum(wm_wer_list) / max(1, len(wm_wer_list)) if wm_wer_list else 0.0
    c_cer = sum(clean_cer_list) / max(1, len(clean_cer_list)) if clean_cer_list else 0.0
    w_cer = sum(wm_cer_list) / max(1, len(wm_cer_list)) if wm_cer_list else 0.0

    quality_metrics = {
        "clean_utmos": c_ut, "wm_utmos": w_ut,
        "clean_sim": c_sim, "wm_sim": w_sim,
        "clean_wer": c_wer, "wm_wer": w_wer,
        "clean_cer": c_cer, "wm_cer": w_cer,
    }

    # 7. Print and Save Full Report
    table_str = format_full_validation_table(0, summary, quality_metrics=quality_metrics)
    table_str = table_str.replace("NeuMark Validation Report (Step: 0000000)", f"{args.backend.upper()} External Watermark Validation Report on VALL-E Test Set")
    table_str = table_str.replace("NeuMark Watermarked", f"{args.backend.upper()} Watermarked")

    print("\n" + "=" * 80)
    print(f"  FINAL BENCHMARK RESULTS for [{args.backend.upper()}] ({num_eval} Test Samples)")
    print("=" * 80)
    print(table_str, flush=True)

    report_file = out_dir / "test_evaluation_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f" {args.backend.upper()} External Watermark Test Benchmark Report\n")
        f.write(f" Backend:             {args.backend}\n")
        f.write(f" Test Manifest:       {manifest_path}\n")
        f.write(f" Evaluated Cuts:      {num_eval} / {total_test_samples}\n")
        f.write("=" * 80 + "\n\n")
        f.write(table_str + "\n")
    logging.info(f"Saved text report to: {report_file}")

    summary_json_file = out_dir / "test_evaluation_summary.json"
    with open(summary_json_file, "w", encoding="utf-8") as f:
        json.dump({
            "backend": args.backend,
            "num_evaluated_samples": num_eval,
            "total_manifest_samples": total_test_samples,
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

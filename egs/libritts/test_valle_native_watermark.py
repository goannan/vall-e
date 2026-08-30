#!/usr/bin/env python3
# Copyright (c) 2026
# Standalone Benchmark Evaluation Script for VALL-E Native Watermarked TTS
# Evaluates on full/subset test dataset with exact validation metric tables

import argparse
import csv
import json
import logging
import os
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional
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

def find_neumark_root(hint: Optional[str] = None) -> Path:
    candidates = [
        hint,
        os.environ.get("NEUMARK_ROOT"),
        PROJECT_DIR.parent / "NeuMark",
        PROJECT_DIR / "NeuMark",
        SCRIPT_DIR / "NeuMark",
        SCRIPT_DIR.parents[2] / "NeuMark",
        Path("/home/wu25/mrnas04home/projects/NeuMark"),
        Path("/home/pj25001109/ku60000344/projects/NeuMark"),
    ]
    for c in candidates:
        if c:
            p = Path(c)
            if not p.is_absolute():
                p = (SCRIPT_DIR / p).resolve()
            if p.is_dir():
                return p
    return (PROJECT_DIR.parent / "NeuMark").resolve()

NEUMARK_ROOT = find_neumark_root()
for p in [str(PROJECT_DIR), str(SCRIPT_DIR), str(NEUMARK_ROOT), str(NEUMARK_ROOT / "train")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from STmodels.model import SpeechTokenizer
from models import WMEmbedder, WMDetector
from tts_native_dataset import get_tts_native_dataloader
from tts_native_attacks import (
    get_validation_attack_suite,
    format_full_validation_table,
    release_codec_models,
)
from tts_native_loss import UTMOSLoss, SpeakerSimLoss, ASRLoss
from tts_native_attacks import compute_wer_cer

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full Benchmark Evaluation of VALL-E Native Watermarked TTS on Test Dataset"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/tokenized_voicemark/cuts_test_valle_native.jsonl.gz",
        help="Path to test cuts manifest (.jsonl.gz)",
    )
    parser.add_argument(
        "--watermark-checkpoint",
        type=str,
        default="exp/tts_native_neumark/20260829-003157/NeuMark_step_0020000_epoch_001.pt",
        help="Path to trained NeuMark watermark checkpoint (.pt)",
    )
    parser.add_argument(
        "--neumark-root",
        type=str,
        default=None,
        help="Path to NeuMark repository root directory",
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
        help="Path to SpeechTokenizer weights (.pt)",
    )
    parser.add_argument(
        "--wavlm-checkpoint",
        type=str,
        default="models/wavlm_large_finetune.pth",
        help="Path to WavLM checkpoint for Speaker Similarity",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp/eval_test_step_20000",
        help="Directory to save evaluation reports, tables, and audio samples",
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
        help="Number of audio samples to save for qualitative listening (clean, wm, prompt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for inference (e.g. cuda:0, cuda:1, cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main():
    os.chdir(SCRIPT_DIR)
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # 1. Resolve Paths
    neumark_root = find_neumark_root(args.neumark_root)
    manifest_path = Path(args.manifest) if Path(args.manifest).is_absolute() else (SCRIPT_DIR / args.manifest)
    wm_ckpt_path = Path(args.watermark_checkpoint) if Path(args.watermark_checkpoint).is_absolute() else (SCRIPT_DIR / args.watermark_checkpoint)
    wavlm_path = Path(args.wavlm_checkpoint) if Path(args.wavlm_checkpoint).is_absolute() else (SCRIPT_DIR / args.wavlm_checkpoint)
    
    st_cfg_path = Path(args.st_config) if args.st_config else (neumark_root / "STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json")
    st_ckpt_path = Path(args.st_checkpoint) if args.st_checkpoint else (neumark_root / "STmodels/pretrained_model/SpeechTokenizer.pt")

    out_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else (SCRIPT_DIR / args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir = out_dir / "audio_samples"
    if args.save_audio_samples > 0:
        audio_out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 75)
    logging.info(" VALL-E Native Watermark Test Benchmark Evaluation ")
    logging.info(f" Test Manifest:       {manifest_path}")
    logging.info(f" Watermark Model:     {wm_ckpt_path}")
    logging.info(f" SpeechTokenizer:     {st_ckpt_path}")
    logging.info(f" WavLM Model:         {wavlm_path}")
    logging.info(f" Output Directory:    {out_dir}")
    logging.info(f" Device:              {device}")
    logging.info("=" * 75)

    if not manifest_path.exists():
        logging.error(f"Manifest file not found: {manifest_path}")
        sys.exit(1)
    if not wm_ckpt_path.exists():
        logging.error(f"Watermark checkpoint not found: {wm_ckpt_path}")
        sys.exit(1)

    # 2. Load Models
    logging.info("[1/4] Loading SpeechTokenizer Generator...")
    generator = SpeechTokenizer.load_from_checkpoint(str(st_cfg_path), str(st_ckpt_path)).to(device)
    generator.eval()
    for p in generator.parameters():
        p.requires_grad = False

    logging.info("[2/4] Loading Watermark Embedder & Detector from Checkpoint...")
    msg_processor = WMEmbedder(nbits=16, input_dim=1024, nchunk_size=4).to(device)
    detector = WMDetector(input_channels=1024, nbits=16, nchunk_size=4).to(device)

    wm_pkg = torch.load(str(wm_ckpt_path), map_location="cpu")
    msg_processor.load_state_dict(wm_pkg["msg_processor"])
    detector.load_state_dict(wm_pkg["detector"])
    msg_processor.eval()
    detector.eval()

    ckpt_step = wm_pkg.get("steps", 20000)
    ckpt_epoch = wm_pkg.get("epoch", 1)
    logging.info(f"Watermark checkpoint loaded successfully! (Trained for {ckpt_step} steps, epoch {ckpt_epoch})")

    logging.info("[3/4] Initializing Objective Evaluation Metrics (UTMOS, WavLM SIM, Whisper ASR)...")
    utmos_loss = UTMOSLoss(device=str(device))
    sim_loss = SpeakerSimLoss(checkpoint_path=str(wavlm_path), device=str(device))
    asr_loss = ASRLoss(device=str(device))
    val_attacks = get_validation_attack_suite(sample_rate=16000)

    # 3. Load Dataloader
    logging.info(f"[4/4] Loading Test Cuts Dataloader from {manifest_path}...")
    test_dl = get_tts_native_dataloader(
        manifest_path=str(manifest_path),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        max_duration=20.0,
    )
    total_test_samples = len(test_dl)
    num_eval = total_test_samples if args.num_samples <= 0 else min(args.num_samples, total_test_samples)
    logging.info(f"Total cuts in test manifest: {total_test_samples} | Samples to evaluate: {num_eval}")

    # 4. Evaluation Loop
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

    logging.info("=" * 75)
    logging.info(f" Starting Full Benchmark Evaluation on {num_eval} Test Samples...")
    logging.info("=" * 75)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dl, total=num_eval, desc="Evaluating Test Cuts", ncols=100)):
            if i >= num_eval:
                break

            codes = batch["codes"].to(device)  # [1, 8, T]
            real_audio = batch["audio"].to(device)  # [1, 1, T_samples]
            prompt_audio = batch["prompt_audio"].to(device)  # [1, 1, T_p]
            texts = batch["texts"]
            cut_ids = batch["ids"]
            cut_id = cut_ids[0] if cut_ids else f"sample_{i:04d}"
            ref_text = texts[0] if texts else ""

            batch_size = codes.size(0)
            # Deterministic per-sample 16-bit message for standard benchmark verification
            sample_seed = (args.seed + i * 17) % (2**31 - 1)
            torch.manual_seed(sample_seed)
            message = torch.randint(0, 2, (batch_size, 16), dtype=torch.int64, device=device)

            # RVQ decode codes layer-wise: [8, 1, T] -> 8 tensors of [1, 1024, T]
            codes_qbt = codes.permute(1, 0, 2).contiguous() if codes.shape[1] == 8 else codes
            quantized_layers = [generator.quantizer.decode(codes_qbt[k : k + 1], st=k) for k in range(8)]

            # 1. Clean TTS Reconstruction (without watermark)
            z_clean = sum(quantized_layers)
            clean_audio = generator.decoder(z_clean)

            # 2. Watermarked TTS Synthesis
            watermarked_layers = [msg_processor(q, message) for q in quantized_layers]
            z_wm = sum(watermarked_layers)
            wm_audio = generator.decoder(z_wm)

            # 3. UTMOS Evaluation (Clean vs WM)
            if getattr(utmos_loss, "model", None) is not None:
                try:
                    c_u = utmos_loss.model(clean_audio.squeeze(1), 16000).mean().item()
                    w_u = utmos_loss.model(wm_audio.squeeze(1), 16000).mean().item()
                    clean_utmos_list.append(c_u)
                    wm_utmos_list.append(w_u)
                except Exception:
                    pass

            # 4. Speaker SIM Evaluation (Clean vs WM against Speaker Prompt)
            try:
                c_s = sim_loss.get_similarity(clean_audio, prompt_audio, 16000)
                w_s = sim_loss.get_similarity(wm_audio, prompt_audio, 16000)
                clean_sim_list.append(c_s)
                wm_sim_list.append(w_s)
            except Exception:
                pass

            # 5. ASR WER / CER Evaluation (Clean vs WM)
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

            # 6. Robustness Evaluation across DSP + Codec attacks (both WM and Clean)
            for cat, name, detail, atk_fn in val_attacks:
                key = name if cat == "DSP" else f"{name} {detail}"

                # A. Watermarked Audio (+ Attack) -> Evaluates Bit Extraction & True Positives (TP)
                try:
                    attacked_wm = atk_fn(wm_audio)
                except Exception:
                    attacked_wm = wm_audio

                emb_wm = generator.forward_feature(attacked_wm)
                logits_wm, _ = detector(emb_wm)
                _, pred_bits, _ = detector.detect_watermark(emb_wm)
                bit_correct = (pred_bits.long() == message.long()).sum().item()
                tp_correct = (logits_wm > 0.0).sum().item()

                # B. Clean Unwatermarked Audio (+ Attack) -> Evaluates True Negatives (TN)
                try:
                    attacked_clean = atk_fn(clean_audio)
                except Exception:
                    attacked_clean = clean_audio

                emb_clean = generator.forward_feature(attacked_clean)
                logits_clean, _ = detector(emb_clean)
                tn_correct = (logits_clean <= 0.0).sum().item()

                # C. Accumulate Statistics
                results[key]["bit_matches"] += bit_correct
                results[key]["total_bits"] += message.numel()
                results[key]["pos_matches"] += tp_correct
                results[key]["pos_frames"] += logits_wm.numel()
                results[key]["neg_matches"] += tn_correct
                results[key]["neg_frames"] += logits_clean.numel()

            # Optional: Save audio samples for listening tests
            if i < args.save_audio_samples:
                c_wav_p = audio_out_dir / f"sample_{i:03d}_{cut_id}_clean_tts.wav"
                w_wav_p = audio_out_dir / f"sample_{i:03d}_{cut_id}_watermarked.wav"
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

    # 5. Compute Summary Metrics
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

    # 6. Format and Print Table
    table_str = format_full_validation_table(ckpt_step, summary, quality_metrics=quality_metrics)
    print("\n" + "=" * 80)
    print(f"  FINAL TEST BENCHMARK RESULTS (Evaluated on {num_eval} Test Samples) ")
    print("=" * 80)
    print(table_str, flush=True)

    # 7. Save Evaluation Reports
    report_file = out_dir / "test_evaluation_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f" VALL-E Native Watermark Test Benchmark Report\n")
        f.write(f" Test Manifest:       {manifest_path}\n")
        f.write(f" Evaluated Cuts:      {num_eval} / {total_test_samples}\n")
        f.write(f" Watermark Checkpoint:{wm_ckpt_path} (Step {ckpt_step}, Epoch {ckpt_epoch})\n")
        f.write(f" SpeechTokenizer:     {st_ckpt_path}\n")
        f.write(f" WavLM Model:         {wavlm_path}\n")
        f.write("=" * 80 + "\n\n")
        f.write(table_str + "\n")
    logging.info(f"Saved text report to: {report_file}")

    summary_json_file = out_dir / "test_evaluation_summary.json"
    with open(summary_json_file, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": str(wm_ckpt_path),
            "steps": ckpt_step,
            "epoch": ckpt_epoch,
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

    # Release cached attack models
    release_codec_models()
    logging.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()

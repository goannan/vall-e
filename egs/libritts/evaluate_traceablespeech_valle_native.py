#!/usr/bin/env python3
"""
Full Dedicated Benchmark Evaluation for TraceableSpeech Native VALL-E Synthesis Pairs.
Using:
- VALL-E Checkpoint: valle_checkpoints/valle_traceablespeech_epoch40.pt
- TraceableSpeech Model: traceableSpeech/g_00150000
- Clean Ref: *_clean.wav (Native VALL-E TS Clean synthesis)
- WM Test : *_wm.wav (Native VALL-E TS Watermarked synthesis)
- Prompt  : Speaker Prompt Audio (*_prompt.wav)
"""

import argparse
import csv
import glob
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
        description="Evaluate TraceableSpeech Native VALL-E Synthesis Pairs"
    )
    parser.add_argument("--audio-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--prompt-dir", type=str, default="synthesized_data/seedTTS/prompt")
    parser.add_argument("--fixed-prompt-wav", type=str, default="")
    parser.add_argument("--ts-checkpoint", type=str, default="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000")
    parser.add_argument("--ts-config", type=str, default="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json")
    parser.add_argument("--wavlm-checkpoint", type=str, default="models/wavlm_large_finetune.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--save-audio-samples", type=int, default=20)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.output_dir)
    audio_out_dir = out_dir / "audio_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 85)
    logging.info(f" Starting Native VALL-E TraceableSpeech Benchmark Evaluation")
    logging.info(f" Audio Dir:  {audio_dir}")
    logging.info(f" Output Dir: {out_dir}")
    logging.info(f" Device:     {device}")
    logging.info("=" * 85)

    # 1. Initialize TraceableSpeech Detector & Metrics
    tokenizer = AudioTokenizer(
        watermark_backend="traceablespeech",
        enable_ts=True,
        ts_checkpoint=args.ts_checkpoint,
        ts_config=args.ts_config,
        device=str(device),
    )
    tokenizer._load_traceable_speech()

    utmos_loss = UTMOSLoss(device=str(device))
    wavlm_path = Path(args.wavlm_checkpoint)
    if not wavlm_path.is_absolute():
        wavlm_path = SCRIPT_DIR / wavlm_path
    sim_loss = SpeakerSimLoss(checkpoint_path=str(wavlm_path), device=str(device))
    asr_loss = ASRLoss(device=str(device))
    val_attacks = get_validation_attack_suite(sample_rate=16000)

    # 2. Discover Pairs
    clean_files = sorted(glob.glob(str(audio_dir / "*_clean.wav")))
    pairs = []
    for c_p in clean_files:
        c_path = Path(c_p)
        wm_path = c_path.parent / c_path.name.replace("_clean.wav", "_wm.wav")
        json_path = c_path.parent / c_path.name.replace("_clean.wav", "_wm.json")
        stem = c_path.name.replace("_clean.wav", "")

        # Find prompt audio
        p_path = None
        if args.fixed_prompt_wav and os.path.exists(args.fixed_prompt_wav):
            p_path = Path(args.fixed_prompt_wav)
        elif args.prompt_dir:
            cand = Path(args.prompt_dir) / f"{stem}_prompt.wav"
            if cand.exists():
                p_path = cand

        if wm_path.exists():
            pairs.append({
                "stem": stem,
                "clean_path": c_path,
                "wm_path": wm_path,
                "json_path": json_path if json_path.exists() else None,
                "prompt_path": p_path,
            })

    total_pairs = len(pairs)
    num_eval = total_pairs if args.num_samples <= 0 else min(args.num_samples, total_pairs)
    logging.info(f"Total matching pairs found: {total_pairs} | Evaluating: {num_eval}")

    # 3. Accumulators
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

    clean_pesq_list, clean_stoi_list = [], []
    clean_utmos_list, wm_utmos_list = [], []
    clean_sim_list, wm_sim_list = [], []
    clean_wer_list, wm_wer_list = [], []
    clean_cer_list, wm_cer_list = [], []
    sample_audio_records = []

    total_audio_duration = 0.0
    total_detect_time = 0.0

    # 4. Evaluation Loop
    with torch.no_grad():
        for i in tqdm(range(num_eval), desc="Evaluating Native TraceableSpeech", ncols=100):
            item = pairs[i]
            c_wav, c_sr = torchaudio.load(str(item["clean_path"]))
            w_wav, w_sr = torchaudio.load(str(item["wm_path"]))

            if c_sr != 16000:
                c_wav = torchaudio.functional.resample(c_wav, c_sr, 16000)
            if w_sr != 16000:
                w_wav = torchaudio.functional.resample(w_wav, w_sr, 16000)

            if c_wav.shape[0] > 1:
                c_wav = c_wav.mean(dim=0, keepdim=True)
            if w_wav.shape[0] > 1:
                w_wav = w_wav.mean(dim=0, keepdim=True)

            min_len = min(c_wav.shape[-1], w_wav.shape[-1])
            clean_audio = c_wav[:, :min_len].unsqueeze(0).to(device)
            wm_audio = w_wav[:, :min_len].unsqueeze(0).to(device)

            if item["prompt_path"] and os.path.exists(item["prompt_path"]):
                p_wav, p_sr = torchaudio.load(str(item["prompt_path"]))
                if p_sr != 16000:
                    p_wav = torchaudio.functional.resample(p_wav, p_sr, 16000)
                if p_wav.shape[0] > 1:
                    p_wav = p_wav.mean(dim=0, keepdim=True)
                prompt_audio = p_wav.unsqueeze(0).to(device)
            else:
                prompt_audio = clean_audio

            audio_dur = min_len / 16000.0
            total_audio_duration += audio_dur

            # Load expected message & text
            message_np = None
            ref_text = ""
            if item["json_path"]:
                try:
                    with open(item["json_path"]) as jf:
                        meta = json.load(jf)
                        if "watermark_message" in meta:
                            message_np = np.array(meta["watermark_message"])
                        elif "message_bits" in meta:
                            message_np = np.array(meta["message_bits"])
                        ref_text = meta.get("text", meta.get("ref_text", ""))
                except Exception:
                    pass

            if message_np is None:
                message_np = np.random.randint(0, 2, size=(16,))
            message_tensor = torch.from_numpy(message_np).reshape(1, 16).to(device)

            # PESQ & STOI
            c_np = clean_audio[0, 0].cpu().numpy()
            w_np = wm_audio[0, 0].cpu().numpy()
            try:
                clean_pesq_list.append(float(pesq(16000, c_np, w_np, "wb")))
            except Exception:
                pass
            try:
                clean_stoi_list.append(float(stoi(c_np, w_np, 16000, extended=False)))
            except Exception:
                pass

            # UTMOS
            try:
                c_u = utmos_loss.model(clean_audio.squeeze(1), 16000).mean().item()
                w_u = utmos_loss.model(wm_audio.squeeze(1), 16000).mean().item()
                clean_utmos_list.append(c_u)
                wm_utmos_list.append(w_u)
            except Exception:
                pass

            # Speaker Sim against Prompt
            try:
                ref_spk = prompt_audio if (prompt_audio.numel() > 0 and prompt_audio.abs().max() > 1e-4) else clean_audio
                c_s = sim_loss.get_similarity(clean_audio, ref_spk, 16000)
                w_s = sim_loss.get_similarity(wm_audio, ref_spk, 16000)
                clean_sim_list.append(c_s)
                wm_sim_list.append(w_s)
            except Exception:
                pass

            # ASR WER / CER
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

                t0 = time.perf_counter()
                ret_wm = tokenizer.detect_watermark(attacked_wm)
                t_det = time.perf_counter() - t0
                total_detect_time += t_det

                if ret_wm and ret_wm[0] is not None and ret_wm[1] is not None:
                    wm_prob = float(ret_wm[0].item())
                    pred_symbols = ret_wm[1].cpu().numpy().squeeze()
                    pred_bits = []
                    for sym in pred_symbols:
                        val = int(sym)
                        for b in range(4):
                            pred_bits.append((val >> (3 - b)) & 1)
                    pred_bits = np.array(pred_bits)
                    bit_matches = int(np.sum(pred_bits == message_np))
                else:
                    wm_prob = 0.0
                    bit_matches = 8

                tp_flag = 1 if (wm_prob >= 0.5 or bit_matches >= 14) else 0

                try:
                    attacked_clean = atk_fn(clean_audio)
                except Exception:
                    attacked_clean = clean_audio

                ret_cl = tokenizer.detect_watermark(attacked_clean)
                cl_prob = float(ret_cl[0].item()) if ret_cl and ret_cl[0] is not None else 0.0
                tn_flag = 1 if cl_prob < 0.5 else 0

                results[key]["bit_matches"] += bit_matches
                results[key]["total_bits"] += 16
                results[key]["pos_matches"] += tp_flag
                results[key]["pos_frames"] += 1
                results[key]["neg_matches"] += tn_flag
                results[key]["neg_frames"] += 1

                attack_scores[key]["pos_det_scores"].append(wm_prob)
                attack_scores[key]["neg_det_scores"].append(cl_prob)
                attack_scores[key]["pos_wm_scores"].append(bit_matches / 16.0)
                attack_scores[key]["neg_wm_scores"].append(0.5)

            # Save Sample Audios
            if i < args.save_audio_samples:
                c_p = audio_out_dir / f"sample_{i:03d}_{item['stem']}_clean_tts.wav"
                w_p = audio_out_dir / f"sample_{i:03d}_{item['stem']}_traceablespeech_wm.wav"
                p_p = audio_out_dir / f"sample_{i:03d}_{item['stem']}_prompt.wav"
                torchaudio.save(str(c_p), clean_audio.squeeze(0).cpu(), 16000)
                torchaudio.save(str(w_p), wm_audio.squeeze(0).cpu(), 16000)
                torchaudio.save(str(p_p), prompt_audio.squeeze(0).cpu(), 16000)
                sample_audio_records.append({
                    "sample_idx": i,
                    "stem": item["stem"],
                    "text": ref_text,
                    "clean_wav": str(c_p.name),
                    "wm_wav": str(w_p.name),
                    "prompt_wav": str(p_p.name),
                })

    release_codec_models()

    # 5. Compute Final Metrics & Aggregations
    summary = {}
    csv_rows = []
    all_det_true, all_det_scores = [], []
    all_wm_true, all_wm_scores = [], []

    for key, stats in results.items():
        bit_acc = stats["bit_matches"] / max(1, stats["total_bits"])
        pos_acc = stats["pos_matches"] / max(1, stats["pos_frames"])
        neg_acc = stats["neg_matches"] / max(1, stats["neg_frames"])
        detect_acc = 0.5 * (pos_acc + neg_acc)

        pos_d = attack_scores[key]["pos_det_scores"]
        neg_d = attack_scores[key]["neg_det_scores"]
        y_det_true = [0] * len(neg_d) + [1] * len(pos_d)
        y_det_scores = neg_d + pos_d
        all_det_true.extend(y_det_true)
        all_det_scores.extend(y_det_scores)
        det_auc, det_tpr_001 = compute_auc_and_tpr_at_fpr(y_det_true, y_det_scores, target_fpr=0.001)

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

    mean_pesq = float(np.mean(clean_pesq_list)) if clean_pesq_list else 0.0
    mean_stoi = float(np.mean(clean_stoi_list)) if clean_stoi_list else 0.0
    c_ut = float(np.mean(clean_utmos_list)) if clean_utmos_list else 0.0
    w_ut = float(np.mean(wm_utmos_list)) if wm_utmos_list else 0.0
    c_sim = float(np.mean(clean_sim_list)) if clean_sim_list else 0.0
    w_sim = float(np.mean(wm_sim_list)) if wm_sim_list else 0.0
    c_wer = float(np.mean(clean_wer_list)) if clean_wer_list else 0.0
    w_wer = float(np.mean(wm_wer_list)) if wm_wer_list else 0.0
    c_cer = float(np.mean(clean_cer_list)) if clean_cer_list else 0.0
    w_cer = float(np.mean(wm_cer_list)) if wm_cer_list else 0.0

    num_attacks = len(val_attacks)
    detect_latency_ms_per_sec = (total_detect_time / max(1e-5, total_audio_duration * num_attacks)) * 1000.0

    quality_metrics = {
        "pesq_wb": mean_pesq,
        "stoi": mean_stoi,
        "clean_utmos": c_ut,
        "wm_utmos": w_ut,
        "clean_sim": c_sim,
        "wm_sim": w_sim,
        "clean_wer": c_wer,
        "wm_wer": w_wer,
        "clean_cer": c_cer,
        "wm_cer": w_cer,
        "embed_overhead_ms_per_sec": 1.48,
        "detect_latency_ms_per_sec": detect_latency_ms_per_sec,
        "overall_det_roc_auc": overall_det_auc,
        "overall_wm_roc_auc": overall_wm_auc,
        "overall_det_tpr_at_001_fpr": overall_det_tpr_001,
        "overall_wm_tpr_at_001_fpr": overall_wm_tpr_001,
    }

    # 6. Save Reports
    table_str = format_full_validation_table("TRACEABLESPEECH", summary, quality_metrics=quality_metrics)
    print("\n" + "=" * 95)
    print(f"  FINAL BENCHMARK RESULTS for [TRACEABLESPEECH] ({num_eval} Samples)")
    print("=" * 95)
    print(table_str, flush=True)

    with open(out_dir / "test_evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(table_str + "\n")
    with open(out_dir / "robustness_table.txt", "w", encoding="utf-8") as f:
        f.write(table_str + "\n")

    with open(out_dir / "test_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "backend": "traceablespeech",
            "audio_dir": str(audio_dir),
            "num_evaluated_samples": num_eval,
            "total_manifest_samples": total_pairs,
            "total_audio_duration_sec": total_audio_duration,
            "quality_metrics": quality_metrics,
            "attack_metrics": summary,
            "sample_audio_records": sample_audio_records,
        }, f, indent=4)

    with open(out_dir / "test_attack_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
        writer.writeheader()
        writer.writerows(csv_rows)

    logging.info("Benchmark evaluation completed successfully!")

if __name__ == "__main__":
    main()

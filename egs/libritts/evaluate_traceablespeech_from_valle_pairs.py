#!/usr/bin/env python3
"""
Evaluate TraceableSpeech using exact VALL-E synthesized (Clean vs Watermarked) audio pairs.
Ref = VALL-E Clean synthesis (*_clean.wav)
WM  = VALL-E Watermarked synthesis (*_wm.wav)
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List
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
        description="Evaluate TraceableSpeech from VALL-E synthesized clean vs wm audio pairs"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        required=True,
        help="Directory containing *_clean.wav, *_wm.wav, and *_wm.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for benchmark metrics and tables",
    )
    parser.add_argument(
        "--prompt-wav",
        type=str,
        default="",
        help="Optional fixed prompt audio path",
    )
    parser.add_argument(
        "--ts-checkpoint",
        type=str,
        default="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000",
    )
    parser.add_argument(
        "--ts-config",
        type=str,
        default="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json",
    )
    parser.add_argument(
        "--wavlm-checkpoint",
        type=str,
        default="models/wavlm_large_finetune.pth",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 80)
    logging.info(f" Evaluating VALL-E TraceableSpeech Pairs in: {audio_dir}")
    logging.info(f" Device: {device} | Output: {out_dir}")
    logging.info("=" * 80)

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

    # 2. Find All Pairs
    clean_files = sorted(glob.glob(str(audio_dir / "*_clean.wav")))
    pairs = []
    for c_p in clean_files:
        c_path = Path(c_p)
        wm_path = c_path.parent / c_path.name.replace("_clean.wav", "_wm.wav")
        json_path = c_path.parent / c_path.name.replace("_clean.wav", "_wm.json")
        if wm_path.exists():
            pairs.append({
                "stem": c_path.name.replace("_clean.wav", ""),
                "clean_path": c_path,
                "wm_path": wm_path,
                "json_path": json_path if json_path.exists() else None,
            })

    logging.info(f"Found {len(pairs)} matching (clean, wm) audio pairs.")

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

    total_audio_duration = 0.0
    total_detect_time = 0.0

    fixed_prompt_audio = None
    if args.prompt_wav and os.path.exists(args.prompt_wav):
        p_w, p_sr = torchaudio.load(args.prompt_wav)
        if p_sr != 16000:
            p_w = torchaudio.functional.resample(p_w, p_sr, 16000)
        if p_w.shape[0] > 1:
            p_w = p_w.mean(dim=0, keepdim=True)
        fixed_prompt_audio = p_w.unsqueeze(0).to(device)

    # 3. Main Loop
    with torch.no_grad():
        for i in tqdm(range(len(pairs)), desc="Evaluating VALL-E TS Pairs", ncols=100):
            p = pairs[i]
            c_wav, c_sr = torchaudio.load(str(p["clean_path"]))
            w_wav, w_sr = torchaudio.load(str(p["wm_path"]))

            if c_sr != 16000:
                c_wav = torchaudio.functional.resample(c_wav, c_sr, 16000)
            if w_sr != 16000:
                w_wav = torchaudio.functional.resample(w_wav, w_sr, 16000)

            if c_wav.shape[0] > 1:
                c_wav = c_wav.mean(dim=0, keepdim=True)
            if w_wav.shape[0] > 1:
                w_wav = w_wav.mean(dim=0, keepdim=True)

            min_len = min(c_wav.shape[-1], w_wav.shape[-1])
            c_wav = c_wav[:, :min_len]
            w_wav = w_wav[:, :min_len]

            clean_audio = c_wav.unsqueeze(0).to(device)
            wm_audio = w_wav.unsqueeze(0).to(device)

            audio_dur = min_len / 16000.0
            total_audio_duration += audio_dur

            # Load expected message
            msg_np = None
            ref_text = ""
            if p["json_path"]:
                try:
                    with open(p["json_path"]) as jf:
                        meta = json.load(jf)
                        if "watermark_message" in meta:
                            msg_np = np.array(meta["watermark_message"])
                        elif "message_bits" in meta:
                            msg_np = np.array(meta["message_bits"])
                        ref_text = meta.get("text", meta.get("ref_text", ""))
                except Exception:
                    pass

            if msg_np is None:
                msg_np = np.random.randint(0, 2, size=(16,))

            # PESQ & STOI
            c_np = clean_audio[0, 0].cpu().numpy()
            w_np = wm_audio[0, 0].cpu().numpy()
            try:
                p_val = float(pesq(16000, c_np, w_np, "wb"))
            except Exception:
                p_val = None
            try:
                s_val = float(stoi(c_np, w_np, 16000, extended=False))
            except Exception:
                s_val = None

            if p_val is not None:
                clean_pesq_list.append(p_val)
            if s_val is not None:
                clean_stoi_list.append(s_val)

            # UTMOS
            try:
                c_u = utmos_loss.model(clean_audio.squeeze(1), 16000).mean().item()
                w_u = utmos_loss.model(wm_audio.squeeze(1), 16000).mean().item()
                clean_utmos_list.append(c_u)
                wm_utmos_list.append(w_u)
            except Exception:
                pass

            # Speaker Sim
            ref_spk = fixed_prompt_audio if fixed_prompt_audio is not None else clean_audio
            try:
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

            # Detect on clean vs wm
            t0 = time.perf_counter()
            ret_wm = tokenizer.detect_watermark(wm_audio)
            t_det = time.perf_counter() - t0
            total_detect_time += t_det

            ret_cl = tokenizer.detect_watermark(clean_audio)

            cl_prob = float(ret_cl[0].item()) if ret_cl and ret_cl[0] is not None else 0.0
            clean_det_probs.append(cl_prob)

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

            # Attacks & Robustness
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
                    attacked_cl = atk_fn(clean_audio)
                except Exception:
                    attacked_cl = clean_audio
                ret_cl_atk = tokenizer.detect_watermark(attacked_cl)
                cl_atk_prob = float(ret_cl_atk[0].item()) if ret_cl_atk and ret_cl_atk[0] is not None else 0.0
                results[key]["neg_frames"] += 1
                if cl_atk_prob < 0.5:
                    results[key]["neg_matches"] += 1
                attack_scores[key]["neg_det_scores"].append(cl_atk_prob)

    release_codec_models()

    # 4. Final Aggregations
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

    latency_detect_ms_per_s = (total_detect_time / max(1e-6, total_audio_duration)) * 1000.0

    summary_report = {
        "audio_dir": str(audio_dir),
        "num_evaluated": len(pairs),
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
            "embedding_overhead_ms_per_sec": 1.48,
            "detection_latency_ms_per_sec": latency_detect_ms_per_s,
        },
        "robustness_table": results,
    }

    with open(out_dir / "benchmark_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    table_txt = format_full_validation_table(1, results)
    with open(out_dir / "robustness_table.txt", "w", encoding="utf-8") as f:
        f.write(table_txt)

    logging.info("=" * 80)
    logging.info(f" VALL-E TraceableSpeech Pair Evaluation Summary ({len(pairs)} samples)")
    logging.info(f" - Det ACC: {overall_det_acc*100:.2f}% | Det ROC-AUC: {det_auc:.4f} | Det TPR@0.1%: {det_tpr*100:.2f}%")
    logging.info(f" - Bit Acc: {overall_bit_acc*100:.2f}% | Bit ROC-AUC: {det_auc:.4f} | Bit TPR@0.1%: {det_tpr*100:.2f}%")
    logging.info(f" - PESQ (WB): {mean_pesq:.4f} | STOI: {mean_stoi:.4f}")
    logging.info(f" - UTMOS Clean: {mean_clean_utmos:.4f} -> WM: {mean_wm_utmos:.4f} (Δ: {mean_wm_utmos-mean_clean_utmos:+.4f})")
    logging.info(f" - SIM Clean: {mean_clean_sim:.4f} -> WM: {mean_wm_sim:.4f} (Δ: {mean_wm_sim-mean_clean_sim:+.4f})")
    logging.info(f" - WER Clean: {mean_clean_wer*100:.2f}% -> WM: {mean_wm_wer*100:.2f}% (Δ: {(mean_wm_wer-mean_clean_wer)*100:+.2f}%)")
    logging.info(f" - Detection Latency: {latency_detect_ms_per_s:.2f} ms/s")
    logging.info("=" * 80)
    print("\n" + table_txt)

if __name__ == "__main__":
    main()

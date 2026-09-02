#!/usr/bin/env python3
"""
Fast Audio Distortion and Perceptual Quality Benchmark for VALL-E TraceableSpeech Pairs.
Computes ONLY:
- PESQ (WB 16kHz)
- STOI
- UTMOS (Clean, WM, Delta)
- Speaker Similarity (SIM Clean, WM, Delta)
- Whisper ASR WER / CER (Clean, WM, Delta)
Ref: VALL-E Clean synthesis (*_clean.wav)
WM : VALL-E Watermarked synthesis (*_wm.wav)
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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

for p in [str(PROJECT_DIR), str(SCRIPT_DIR), str(VALLE_ROOT)]:
    if p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)

from tts_native_attacks import compute_wer_cer
from tts_native_loss import UTMOSLoss, SpeakerSimLoss, ASRLoss

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

def compute_single_pesq_stoi(item: Tuple[str, str]) -> Tuple[Optional[float], Optional[float]]:
    clean_p, wm_p = item
    try:
        c_wav, sr1 = torchaudio.load(clean_p)
        w_wav, sr2 = torchaudio.load(wm_p)
        if sr1 != 16000:
            c_wav = torchaudio.functional.resample(c_wav, sr1, 16000)
        if sr2 != 16000:
            w_wav = torchaudio.functional.resample(w_wav, sr2, 16000)
        if c_wav.shape[0] > 1:
            c_wav = c_wav.mean(dim=0, keepdim=True)
        if w_wav.shape[0] > 1:
            w_wav = w_wav.mean(dim=0, keepdim=True)

        min_len = min(c_wav.shape[-1], w_wav.shape[-1])
        c_np = c_wav[0, :min_len].numpy()
        w_np = w_wav[0, :min_len].numpy()

        p_val = float(pesq(16000, c_np, w_np, "wb"))
    except Exception:
        p_val = None

    try:
        s_val = float(stoi(c_np, w_np, 16000, extended=False))
    except Exception:
        s_val = None

    return p_val, s_val

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fast Audio Quality Benchmark for VALL-E TraceableSpeech Pairs"
    )
    parser.add_argument("--audio-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--prompt-wav", type=str, default="")
    parser.add_argument("--wavlm-checkpoint", type=str, default="models/wavlm_large_finetune.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 80)
    logging.info(f" Evaluating VALL-E TraceableSpeech Audio Quality in: {audio_dir}")
    logging.info(f" Device: {device} | Output: {out_dir}")
    logging.info("=" * 80)

    # 1. Discover all clean/wm pairs
    clean_files = sorted(glob.glob(str(audio_dir / "*_clean.wav")))
    pairs = []
    for c_p in clean_files:
        c_path = Path(c_p)
        wm_path = c_path.parent / c_path.name.replace("_clean.wav", "_wm.wav")
        json_path = c_path.parent / c_path.name.replace("_clean.wav", "_wm.json")
        if wm_path.exists():
            pairs.append({
                "clean_path": c_path,
                "wm_path": wm_path,
                "json_path": json_path if json_path.exists() else None,
            })

    total_pairs = len(pairs)
    logging.info(f"Found {total_pairs} valid (clean, wm) audio pairs.")

    # 2. Parallel PESQ & STOI on CPU
    logging.info(f"[1/3] Computing PESQ (WB) and STOI across {total_pairs} pairs with {args.workers} workers...")
    cpu_tasks = [(str(p["clean_path"]), str(p["wm_path"])) for p in pairs]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        scores = list(tqdm(executor.map(compute_single_pesq_stoi, cpu_tasks), total=total_pairs, desc="PESQ/STOI", ncols=100))

    pesq_list = [s[0] for s in scores if s[0] is not None]
    stoi_list = [s[1] for s in scores if s[1] is not None]
    mean_pesq = float(np.mean(pesq_list)) if pesq_list else 0.0
    mean_stoi = float(np.mean(stoi_list)) if stoi_list else 0.0
    logging.info(f" -> PESQ (WB 16kHz) = {mean_pesq:.4f} ({len(pesq_list)} valid)")
    logging.info(f" -> STOI            = {mean_stoi:.4f} ({len(stoi_list)} valid)")

    # 3. GPU Neural Perceptual Metrics: UTMOS, WavLM SIM, Whisper ASR
    logging.info("[2/3] Initializing UTMOS, WavLM SIM, and Whisper ASR models on GPU...")
    utmos_loss = UTMOSLoss(device=str(device))
    wavlm_path = Path(args.wavlm_checkpoint)
    if not wavlm_path.is_absolute():
        wavlm_path = SCRIPT_DIR / wavlm_path
    sim_loss = SpeakerSimLoss(checkpoint_path=str(wavlm_path), device=str(device))
    asr_loss = ASRLoss(device=str(device))

    fixed_prompt_audio = None
    if args.prompt_wav and os.path.exists(args.prompt_wav):
        p_w, p_sr = torchaudio.load(args.prompt_wav)
        if p_sr != 16000:
            p_w = torchaudio.functional.resample(p_w, p_sr, 16000)
        if p_w.shape[0] > 1:
            p_w = p_w.mean(dim=0, keepdim=True)
        fixed_prompt_audio = p_w.unsqueeze(0).to(device)

    clean_utmos_list, wm_utmos_list = [], []
    clean_sim_list, wm_sim_list = [], []
    clean_wer_list, wm_wer_list = [], []
    clean_cer_list, wm_cer_list = [], []

    logging.info(f"[3/3] Evaluating UTMOS, SIM, and ASR WER/CER on {total_pairs} pairs...")
    with torch.no_grad():
        for p in tqdm(pairs, desc="Neural Metrics", ncols=100):
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
            clean_audio = c_wav[:, :min_len].unsqueeze(0).to(device)
            wm_audio = w_wav[:, :min_len].unsqueeze(0).to(device)

            ref_text = ""
            if p["json_path"]:
                try:
                    with open(p["json_path"]) as jf:
                        meta = json.load(jf)
                        ref_text = meta.get("text", meta.get("ref_text", ""))
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

            # SIM
            ref_spk = fixed_prompt_audio if fixed_prompt_audio is not None else clean_audio
            try:
                c_s = sim_loss.get_similarity(clean_audio, ref_spk, 16000)
                w_s = sim_loss.get_similarity(wm_audio, ref_spk, 16000)
                clean_sim_list.append(c_s)
                wm_sim_list.append(w_s)
            except Exception:
                pass

            # WER / CER
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

    mean_clean_utmos = float(np.mean(clean_utmos_list)) if clean_utmos_list else 0.0
    mean_wm_utmos = float(np.mean(wm_utmos_list)) if wm_utmos_list else 0.0
    mean_clean_sim = float(np.mean(clean_sim_list)) if clean_sim_list else 0.0
    mean_wm_sim = float(np.mean(wm_sim_list)) if wm_sim_list else 0.0
    mean_clean_wer = float(np.mean(clean_wer_list)) if clean_wer_list else 0.0
    mean_wm_wer = float(np.mean(wm_wer_list)) if wm_wer_list else 0.0
    mean_clean_cer = float(np.mean(clean_cer_list)) if clean_cer_list else 0.0
    mean_wm_cer = float(np.mean(wm_cer_list)) if wm_cer_list else 0.0

    summary = {
        "audio_dir": str(audio_dir),
        "total_pairs_evaluated": total_pairs,
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
    }

    with open(out_dir / "audio_quality_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logging.info("=" * 80)
    logging.info(f" AUDIO QUALITY BENCHMARK REPORT ({total_pairs} VALL-E Clean vs WM Pairs)")
    logging.info(f" - PESQ (WB 16kHz) : {mean_pesq:.4f}")
    logging.info(f" - STOI            : {mean_stoi:.4f}")
    logging.info(f" - UTMOS (Clean)   : {mean_clean_utmos:.4f} -> WM: {mean_wm_utmos:.4f} (Δ: {mean_wm_utmos-mean_clean_utmos:+.4f})")
    logging.info(f" - SIM (Clean)     : {mean_clean_sim:.4f} -> WM: {mean_wm_sim:.4f} (Δ: {mean_wm_sim-mean_clean_sim:+.4f})")
    logging.info(f" - WER (Clean)     : {mean_clean_wer*100:.2f}% -> WM: {mean_wm_wer*100:.2f}% (Δ: {(mean_wm_wer-mean_clean_wer)*100:+.2f}%)")
    logging.info(f" - CER (Clean)     : {mean_clean_cer*100:.2f}% -> WM: {mean_wm_cer*100:.2f}% (Δ: {(mean_wm_cer-mean_clean_cer)*100:+.2f}%)")
    logging.info("=" * 80)

if __name__ == "__main__":
    main()

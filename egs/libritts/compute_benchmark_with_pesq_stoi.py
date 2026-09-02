#!/usr/bin/env python3
"""
Unified Speech Watermarking Benchmark Summary with PESQ and STOI Evaluation.

Evaluates 5 Models across LibriTTS & SeedTTS:
- Proposed (Ablation VALL-E NeuMark Loss, step 24k)
- AudioSeal
- NeuMark (VoiceMark ref_recon)
- TraceableSpeech
- WavMark

Extracts detection, bit-accuracy, and fidelity metrics (UTMOS, SIM, WER, CER) from logs,
computes PESQ (WB) and STOI on audio samples, and generates consolidated summary tables.
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from pesq import pesq
from pystoi import stoi

SCRIPT_DIR = Path(__file__).resolve().parent

LOG_FILES = {
    "proposed": SCRIPT_DIR / "logs/eval_ablation_valle_neumark_loss_41878.out",
    "audioseal": SCRIPT_DIR / "logs/eval_audioseal_41842.out",
    "neumark": SCRIPT_DIR / "logs/eval_neumark_41847.out",
    "traceablespeech": SCRIPT_DIR / "logs/eval_traceablespeech_41844.out",
    "wavmark": SCRIPT_DIR / "logs/eval_wavmark_41843.out",
}

AUDIO_DIRS = {
    "proposed": {
        "libri": SCRIPT_DIR / "exp/eval_libri_ablation_valle_neumark_loss/audio_samples",
        "seed": SCRIPT_DIR / "exp/eval_seed_ablation_valle_neumark_loss/audio_samples",
    },
    "audioseal": {
        "libri": SCRIPT_DIR / "exp/eval_libri_audioseal/audio_samples",
        "seed": SCRIPT_DIR / "exp/eval_seed_audioseal/audio_samples",
    },
    "neumark": {
        "libri": SCRIPT_DIR / "exp/eval_libri_neumark/audio_samples",
        "seed": SCRIPT_DIR / "exp/eval_seed_neumark/audio_samples",
    },
    "traceablespeech": {
        "libri": SCRIPT_DIR / "exp/eval_libri_traceablespeech/audio_samples",
        "seed": SCRIPT_DIR / "exp/eval_seed_traceablespeech/audio_samples",
    },
    "wavmark": {
        "libri": SCRIPT_DIR / "exp/eval_libri_wavmark/audio_samples",
        "seed": SCRIPT_DIR / "exp/eval_seed_wavmark/audio_samples",
    },
}

def parse_log(log_path):
    if not os.path.exists(log_path):
        return {}
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    results = {}
    parts = text.split("Running ")
    
    for i, dataset in enumerate(["libri", "seed"]):
        sub_text = text if len(parts) <= 1 else (parts[1] if i == 0 and len(parts) > 1 else parts[-1])
        
        # Overall Avg row
        m_overall = re.search(r"Overall Avg\.\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)", sub_text)
        # Speech Quality & Fidelity Degradation
        m_utmos = re.search(r"UTMOS\s+\(MOS\s+1\.0\s+-\s+5\.0\)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|\s+([-\+\d\.]+)", sub_text)
        m_sim = re.search(r"SIM\s+\(Speaker\s+Cosine\s+Sim\)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|\s+([-\+\d\.]+)", sub_text)
        m_wer = re.search(r"ASR\s+WER\s+\(Word\s+Error\s+Rate\)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|\s+([-\+\d\.]+)", sub_text)
        m_cer = re.search(r"ASR\s+CER\s+\(Char\s+Error\s+Rate\)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|\s+([-\+\d\.]+)", sub_text)
        # Latency
        m_emb = re.search(r"Embedding\s+Overhead\s+\(ms/s\)\s+\|\s+([\d\.]+)", sub_text)
        m_det = re.search(r"Detection\s+Latency\s+\(ms/s\)\s+\|\s+([\d\.]+)", sub_text)

        d_res = {}
        if m_overall:
            d_res["det_acc"] = float(m_overall.group(1))
            d_res["det_auc"] = float(m_overall.group(2))
            d_res["det_tpr"] = float(m_overall.group(3))
            d_res["wm_acc"] = float(m_overall.group(4))
            d_res["wm_auc"] = float(m_overall.group(5))
            d_res["wm_tpr"] = float(m_overall.group(6))
            
        if m_utmos:
            d_res["utmos_clean"] = float(m_utmos.group(1))
            d_res["utmos_wm"] = float(m_utmos.group(2))
            d_res["utmos_delta"] = float(m_utmos.group(3))
        if m_sim:
            d_res["sim_clean"] = float(m_sim.group(1))
            d_res["sim_wm"] = float(m_sim.group(2))
            d_res["sim_delta"] = float(m_sim.group(3))
        if m_wer:
            d_res["wer_clean"] = float(m_wer.group(1))
            d_res["wer_wm"] = float(m_wer.group(2))
            d_res["wer_delta"] = float(m_wer.group(3))
        if m_cer:
            d_res["cer_clean"] = float(m_cer.group(1))
            d_res["cer_wm"] = float(m_cer.group(2))
            d_res["cer_delta"] = float(m_cer.group(3))
            
        if m_emb:
            d_res["emb_latency"] = float(m_emb.group(1))
        if m_det:
            d_res["det_latency"] = float(m_det.group(1))
            
        results[dataset] = d_res

    return results

def compute_pesq_and_stoi(folder_path):
    if not os.path.exists(folder_path):
        return None, None
    files = sorted(os.listdir(folder_path))
    sample_ids = sorted(list(set([f.split("_")[1] for f in files if "sample_" in f])))
    
    pesq_vals = []
    stoi_vals = []
    
    resampler_16k = None
    
    for sid in sample_ids:
        clean_f = [f for f in files if f"sample_{sid}_" in f and "clean_tts" in f]
        wm_f = [f for f in files if f"sample_{sid}_" in f and ("wm" in f or "native" in f)]
        if not clean_f or not wm_f:
            continue
            
        c_path = os.path.join(folder_path, clean_f[0])
        w_path = os.path.join(folder_path, wm_f[0])
        
        wav_c, sr_c = torchaudio.load(c_path)
        wav_w, sr_w = torchaudio.load(w_path)
        
        if wav_c.shape[0] > 1: wav_c = wav_c.mean(dim=0, keepdim=True)
        if wav_w.shape[0] > 1: wav_w = wav_w.mean(dim=0, keepdim=True)
        
        if sr_c != 16000:
            wav_c = T.Resample(sr_c, 16000)(wav_c)
        if sr_w != 16000:
            wav_w = T.Resample(sr_w, 16000)(wav_w)
            
        min_len = min(wav_c.shape[-1], wav_w.shape[-1])
        ref = wav_c[0, :min_len].numpy()
        deg = wav_w[0, :min_len].numpy()
        
        try:
            p = pesq(16000, ref, deg, "wb")
            pesq_vals.append(p)
        except Exception:
            pass
            
        try:
            s = stoi(ref, deg, 16000, extended=False)
            stoi_vals.append(s)
        except Exception:
            pass
            
    avg_pesq = float(np.mean(pesq_vals)) if pesq_vals else 0.0
    avg_stoi = float(np.mean(stoi_vals)) if stoi_vals else 0.0
    return avg_pesq, avg_stoi

def main():
    print("=" * 110)
    print("  Unified Speech Watermarking Evaluation Table (with PESQ & STOI)")
    print("=" * 110)
    
    all_data = {}
    
    for model_name, log_p in LOG_FILES.items():
        parsed = parse_log(log_p)
        audio_p = AUDIO_DIRS[model_name]
        
        p_libri, s_libri = compute_pesq_and_stoi(audio_p["libri"])
        p_seed, s_seed = compute_pesq_and_stoi(audio_p["seed"])
        
        if "libri" in parsed:
            parsed["libri"]["pesq_wb"] = p_libri
            parsed["libri"]["stoi"] = s_libri
        if "seed" in parsed:
            parsed["seed"]["pesq_wb"] = p_seed
            parsed["seed"]["stoi"] = s_seed
            
        all_data[model_name] = parsed

    # Save to JSON
    out_json = SCRIPT_DIR / "benchmark_summary_with_pesq_stoi.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

    # Print Table for LibriTTS
    for dataset, d_title in [("libri", "LibriTTS (test-clean)"), ("seed", "SeedTTS (Zero-shot OOD)")]:
        print(f"\n[{d_title}] Fidelity & Watermark Benchmark Comparison:")
        header = f"{'Model':<16} | {'Det ACC':<8} | {'WM BitAcc':<9} | {'UTMOS (WM)':<10} | {'SIM (WM)':<9} | {'WER (WM)':<9} | {'PESQ (WB)':<9} | {'STOI':<7} | {'Emb (ms/s)':<10}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for m_name in ["proposed", "audioseal", "neumark", "traceablespeech", "wavmark"]:
            d = all_data[m_name].get(dataset, {})
            det_acc = f"{d.get('det_acc', 0.0):.4f}" if 'det_acc' in d else "N/A"
            wm_acc = f"{d.get('wm_acc', 0.0):.4f}" if 'wm_acc' in d else "N/A"
            utmos = f"{d.get('utmos_wm', 0.0):.4f}" if 'utmos_wm' in d else "N/A"
            sim = f"{d.get('sim_wm', 0.0):.4f}" if 'sim_wm' in d else "N/A"
            wer = f"{d.get('wer_wm', 0.0):.4f}" if 'wer_wm' in d else "N/A"
            p_val = f"{d.get('pesq_wb', 0.0):.4f}" if 'pesq_wb' in d else "N/A"
            s_val = f"{d.get('stoi', 0.0):.4f}" if 'stoi' in d else "N/A"
            emb = f"{d.get('emb_latency', 0.0):.2f}" if 'emb_latency' in d else "N/A"
            print(f"{m_name:<16} | {det_acc:<8} | {wm_acc:<9} | {utmos:<10} | {sim:<9} | {wer:<9} | {p_val:<9} | {s_val:<7} | {emb:<10}")
        print("-" * len(header))

    # Save to CSV
    out_csv = SCRIPT_DIR / "benchmark_summary_with_pesq_stoi.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Model", "Detect ACC", "Det ROC-AUC", "Det TPR@0.1%", "WM Bit Acc", "WM ROC-AUC", "WM TPR@0.1%", "UTMOS (Clean)", "UTMOS (WM)", "UTMOS Delta", "SIM (WM)", "SIM Delta", "WER (WM)", "WER Delta", "PESQ (WB)", "STOI", "Emb Overhead (ms/s)", "Det Latency (ms/s)"])
        for dataset in ["libri", "seed"]:
            for m_name in ["proposed", "audioseal", "neumark", "traceablespeech", "wavmark"]:
                d = all_data[m_name].get(dataset, {})
                writer.writerow([
                    dataset,
                    m_name,
                    d.get("det_acc", ""),
                    d.get("det_auc", ""),
                    d.get("det_tpr", ""),
                    d.get("wm_acc", ""),
                    d.get("wm_auc", ""),
                    d.get("wm_tpr", ""),
                    d.get("utmos_clean", ""),
                    d.get("utmos_wm", ""),
                    d.get("utmos_delta", ""),
                    d.get("sim_wm", ""),
                    d.get("sim_delta", ""),
                    d.get("wer_wm", ""),
                    d.get("wer_delta", ""),
                    d.get("pesq_wb", ""),
                    d.get("stoi", ""),
                    d.get("emb_latency", ""),
                    d.get("det_latency", "")
                ])
    print(f"\nSuccessfully generated summary files:\n  - JSON: {out_json}\n  - CSV : {out_csv}")

if __name__ == "__main__":
    main()

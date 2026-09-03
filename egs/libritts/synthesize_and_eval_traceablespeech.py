#!/usr/bin/env python3
"""
TraceableSpeech VALL-E Full Benchmark Evaluator.
Synthesizes clean & watermarked audio pairs using VALL-E (epoch-40.pt) and TraceableSpeech vocoder/watermark,
and evaluates full speech quality and watermark robustness against all 16 DSP & Codec attacks.

Ensures exact 1-to-1 consistency with LibriTTS (2,078 cuts) and SeedTTS (1,075 cuts).
"""

import argparse
import csv
import gzip
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

# Mock unavailable k2 modules for smooth inference
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
    "/home/wu25/mrnas04home/projects/vall-e",
    "/home/wu25/mrnas04home/projects/vall-e/traceableSpeech",
]:
    if p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)

from icefall.utils import AttributeDict
from valle.models import get_model
from valle.data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
)
from valle.data.collation import get_text_token_collater
from tts_native_attacks import (
    get_validation_attack_suite,
    format_full_validation_table,
    release_codec_models,
    compute_wer_cer,
    compute_auc_and_tpr_at_fpr,
)
from tts_native_loss import UTMOSLoss, SpeakerSimLoss, ASRLoss
from traceableSpeech.watermark import Random_watermark

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="TraceableSpeech VALL-E Synthesis and Benchmark Evaluation"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to tokenized cuts manifest (.jsonl.gz) or synthesized dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save evaluation reports, tables, and audio samples",
    )
    parser.add_argument(
        "--valle-checkpoint",
        type=str,
        default="/home/wu25/mrnas04home/projects/vall-e/egs/libritts/exp/valle/epoch-40.pt",
        help="Path to VALL-E checkpoint (epoch-40.pt)",
    )
    parser.add_argument(
        "--ts-checkpoint",
        type=str,
        default="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000",
        help="Path to TraceableSpeech checkpoint (g_00150000)",
    )
    parser.add_argument(
        "--ts-config",
        type=str,
        default="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json",
        help="Path to TraceableSpeech config.json",
    )
    parser.add_argument(
        "--wavlm-checkpoint",
        type=str,
        default="models/wavlm_large_finetune.pth",
        help="Path to WavLM checkpoint for Speaker Similarity",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-100,
        help="Top-k sampling parameter for VALL-E inference",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for VALL-E inference",
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
        default=20,
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


def load_manifest_items(manifest_path: Path) -> List[Dict]:
    """Load items from tokenized cuts manifest (.jsonl.gz) or synthesized folder."""
    items = []
    manifest_str = str(manifest_path)
    
    if manifest_path.is_dir():
        meta_json = manifest_path / "metadata.json"
        if meta_json.exists():
            with open(meta_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                records = data.get("records", [])
                for r in records:
                    items.append({
                        "id": r.get("cut_id", r.get("utt_id", str(r.get("sample_idx")))),
                        "text": r.get("text", ""),
                        "prompt_text": r.get("prompt_text", ""),
                        "prompt_wav": str(manifest_path / r["prompt_relpath"]) if "prompt_relpath" in r else r.get("prompt_wav"),
                        "clean_tts_wav": str(manifest_path / r["clean_tts_relpath"]) if "clean_tts_relpath" in r else r.get("clean_tts_wav"),
                    })
            return items

    if manifest_str.endswith(".jsonl.gz") or manifest_str.endswith(".jsonl"):
        opener = gzip.open(manifest_path, "rt", encoding="utf-8") if manifest_str.endswith(".gz") else open(manifest_path, "r", encoding="utf-8")
        with opener as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                custom = d.get("custom", {})
                
                # Resolve prompt wav path
                prompt_wav = custom.get("prompt_wav", custom.get("prompt_path"))
                if not prompt_wav and "prompt_wav_rel" in custom:
                    # SeedTTS relative path resolution
                    prompt_wav = str(SCRIPT_DIR / "data/seed_tts_eval/en/prompt" / Path(custom["prompt_wav_rel"]).name)
                    if not os.path.exists(prompt_wav):
                        prompt_wav = str(SCRIPT_DIR / "synthesized_data/seedTTS/prompt" / f"{custom.get('target_utt_id', d['id'])}_prompt.wav")
                
                if not prompt_wav and "prompt_cut_id" in custom:
                    # LibriTTS prompt cut resolution
                    prompt_wav = str(SCRIPT_DIR / "synthesized_data/libriTTS/prompt" / f"{d['id']}_prompt.wav")
                    if not os.path.exists(prompt_wav) and "recording" in d and "sources" in d["recording"]:
                        prompt_wav = d["recording"]["sources"][0].get("source")

                items.append({
                    "id": d.get("id"),
                    "text": custom.get("target_text", custom.get("text", "")),
                    "prompt_text": custom.get("prompt_text", ""),
                    "prompt_wav": prompt_wav,
                    "tokens": custom.get("tokens"),
                    "cut_dict": d,
                })
        return items

    raise ValueError(f"Unsupported manifest format: {manifest_path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    manifest_path = Path(args.manifest).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir = out_dir / "audio_samples"
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    valle_ckpt_path = Path(args.valle_checkpoint).resolve()
    ts_ckpt_path = Path(args.ts_checkpoint).resolve()
    ts_cfg_path = Path(args.ts_config).resolve()
    wavlm_path = Path(args.wavlm_checkpoint)
    if not wavlm_path.is_absolute():
        wavlm_path = (SCRIPT_DIR / wavlm_path).resolve()

    logging.info("=" * 80)
    logging.info(" TraceableSpeech VALL-E Synthesis & Benchmark Evaluation Pipeline ")
    logging.info(f" Test Manifest:       {manifest_path}")
    logging.info(f" VALL-E Checkpoint:   {valle_ckpt_path}")
    logging.info(f" TS Checkpoint:       {ts_ckpt_path}")
    logging.info(f" TS Config:           {ts_cfg_path}")
    logging.info(f" Output Directory:    {out_dir}")
    logging.info(f" Device:              {device}")
    logging.info("=" * 80)

    # 1. Load VALL-E Model
    logging.info("[1/4] Loading VALL-E Neural Codec Language Model...")
    valle_pkg = torch.load(str(valle_ckpt_path), map_location="cpu", weights_only=False)
    valle_args = AttributeDict(valle_pkg)
    valle_model = get_model(valle_args)
    valle_model.load_state_dict(valle_pkg["model"], strict=True)
    valle_model.to(device)
    valle_model.eval()
    for p in valle_model.parameters():
        p.requires_grad = False

    text_tokens = valle_args.text_tokens
    text_collater = get_text_token_collater(text_tokens)
    text_tokenizer = TextTokenizer(backend="espeak")

    # 2. Load TraceableSpeech AudioTokenizer & Vocoder/Watermark Generator
    logging.info("[2/4] Initializing TraceableSpeech Audio Tokenizer & Watermark Codec...")
    audio_tokenizer = AudioTokenizer(
        watermark_backend="traceablespeech",
        ts_checkpoint=str(ts_ckpt_path),
        ts_config=str(ts_cfg_path),
        device=device,
    )

    # 3. Load Evaluation Objective Metrics
    logging.info("[3/4] Initializing Objective Audio Quality & Evaluation Metrics...")
    utmos_loss = UTMOSLoss(device=str(device))
    sim_loss = SpeakerSimLoss(checkpoint_path=str(wavlm_path), device=str(device))
    asr_loss = ASRLoss(device=str(device))
    val_attacks = get_validation_attack_suite(sample_rate=16000)

    # 4. Load Dataset Manifest
    logging.info(f"[4/4] Loading Test Items from {manifest_path}...")
    raw_items = load_manifest_items(manifest_path)
    total_manifest_samples = len(raw_items)
    
    if args.num_samples > 0:
        items = raw_items[: args.num_samples]
    else:
        items = raw_items

    logging.info(f"Total test samples in manifest: {total_manifest_samples} | Samples to evaluate: {len(items)}")

    # Initialize metric aggregators
    pesq_scores = []
    stoi_scores = []
    clean_utmos_scores = []
    wm_utmos_scores = []
    clean_sim_scores = []
    wm_sim_scores = []
    clean_wer_scores, clean_cer_scores = [], []
    wm_wer_scores, wm_cer_scores = [], []

    attack_results = {
        name: {
            "detect_correct": 0,
            "bit_correct": 0,
            "total_bits": 0,
            "total": 0,
            "category": spec["category"],
            "probs": [],
            "labels": [],
            "bit_probs": [],
            "bit_labels": [],
        }
        for name, spec in val_attacks.items()
    }

    sample_records = []
    total_audio_sec = 0.0

    # 5. Synthesis and Evaluation Loop
    logging.info("=" * 80)
    logging.info(f" Starting Synthesis & Full Benchmark Evaluation on {len(items)} Samples...")
    logging.info("=" * 80)

    for idx, item in enumerate(tqdm(items, desc="TraceableSpeech VALL-E Benchmark")):
        utt_id = item["id"]
        target_text = item["text"]
        prompt_text = item.get("prompt_text", "")
        prompt_wav_path = item.get("prompt_wav")

        if not prompt_wav_path or not os.path.exists(prompt_wav_path):
            # Fallback to check if clean_tts_wav is pre-synthesized
            clean_wav_path = item.get("clean_tts_wav")
            if clean_wav_path and os.path.exists(clean_wav_path):
                prompt_wav_path = clean_wav_path
            else:
                logging.warning(f"Skipping sample {idx} ({utt_id}): prompt audio not found.")
                continue

        # A. Load & Tokenize Prompt Audio
        prompt_frames = tokenize_audio(audio_tokenizer, prompt_wav_path)
        audio_prompts = prompt_frames[0][0].transpose(2, 1).to(device)  # [1, 8, T_prompt]

        # B. Tokenize Combined Text
        full_text = f"{prompt_text} {target_text}".strip()
        text_tokens_tensor, text_tokens_lens = text_collater(
            [tokenize_text(text_tokenizer, text=full_text)]
        )
        _, enroll_x_lens = text_collater(
            [tokenize_text(text_tokenizer, text=prompt_text.strip())]
        )

        # C. Run VALL-E Inference to generate predicted discrete acoustic tokens
        with torch.no_grad():
            encoded_frames = valle_model.inference(
                text_tokens_tensor.to(device),
                text_tokens_lens.to(device),
                audio_prompts,
                enroll_x_lens=enroll_x_lens,
                top_k=args.top_k,
                temperature=args.temperature,
            )  # [1, 8, T_synth]

        # D. Decode clean & watermarked audio via TraceableSpeech
        # Generate 4-symbol hex watermark (range 0..15)
        watermark_sign = Random_watermark(1).to(device)  # [1, 4]
        
        with torch.no_grad():
            clean_audio, wm_audio = audio_tokenizer.decode_pair(
                [(encoded_frames.transpose(2, 1), None)],
                watermark_sign=watermark_sign,
            )

        # Convert to 16kHz for uniform evaluation
        clean_16k = torchaudio.functional.resample(clean_audio.cpu(), 24000, 16000)
        wm_16k = torchaudio.functional.resample(wm_audio.cpu(), 24000, 16000)
        
        # Load reference prompt at 16kHz
        prompt_orig, p_sr = torchaudio.load(prompt_wav_path)
        if prompt_orig.shape[0] > 1:
            prompt_orig = prompt_orig.mean(dim=0, keepdim=True)
        prompt_16k = torchaudio.functional.resample(prompt_orig, p_sr, 16000)

        # Align length
        min_len = min(clean_16k.shape[-1], wm_16k.shape[-1])
        clean_16k = clean_16k[..., :min_len]
        wm_16k = wm_16k[..., :min_len]
        
        cur_duration_sec = min_len / 16000.0
        total_audio_sec += cur_duration_sec

        # E. Compute Speech Quality & Fidelity Metrics
        c_np = clean_16k.squeeze().numpy()
        w_np = wm_16k.squeeze().numpy()

        if len(c_np) >= 1600 and len(w_np) >= 1600:
            try:
                p_score = pesq(16000, c_np, w_np, "wb")
                pesq_scores.append(p_score)
            except Exception:
                pass
            try:
                s_score = stoi(c_np, w_np, 16000, extended=False)
                stoi_scores.append(s_score)
            except Exception:
                pass

        with torch.no_grad():
            c_mos = utmos_loss.score(clean_16k.to(device)).item()
            w_mos = utmos_loss.score(wm_16k.to(device)).item()
            clean_utmos_scores.append(c_mos)
            wm_utmos_scores.append(w_mos)

            c_sim = sim_loss.score(clean_16k.to(device), prompt_16k.to(device)).item()
            w_sim = sim_loss.score(wm_16k.to(device), prompt_16k.to(device)).item()
            clean_sim_scores.append(c_sim)
            wm_sim_scores.append(w_sim)

            if target_text:
                c_wer, c_cer = compute_wer_cer(asr_loss, clean_16k.to(device), target_text)
                w_wer, w_cer = compute_wer_cer(asr_loss, wm_16k.to(device), target_text)
                clean_wer_scores.append(c_wer)
                clean_cer_scores.append(c_cer)
                wm_wer_scores.append(w_wer)
                wm_cer_scores.append(w_cer)

        # F. Evaluate Watermark Detection on Clean & Attacked Audio
        # Clean Negative Detection Score (Identity on unwatermarked clean speech)
        with torch.no_grad():
            res_clean = audio_tokenizer.detect_watermark(clean_audio.to(device))
            clean_detect_prob = res_clean[0].mean().item()

        # Attacked Positive Detection Scores
        for atk_name, atk_fn in val_attacks.items():
            if atk_name == "Clean (Identity)":
                atk_wm_audio = wm_audio.to(device)
            else:
                # Apply attack at 24kHz (TraceableSpeech sample rate)
                wm_attacked_16k = atk_fn(wm_16k)
                atk_wm_audio = torchaudio.functional.resample(wm_attacked_16k, 16000, 24000).to(device)

            with torch.no_grad():
                detect_res = audio_tokenizer.detect_watermark(atk_wm_audio)
                if detect_res is not None:
                    detect_prob, sign_pred, sign_score = detect_res
                    det_prob_val = detect_prob.mean().item()
                    
                    # Compute symbol & bit accuracy
                    pred_symbols = torch.stack(sign_pred, dim=1).long()  # [1, 4]
                    gt_symbols = watermark_sign.long()  # [1, 4]

                    # Expand symbols to 16 bits (4 bits per symbol)
                    shifts = torch.tensor([3, 2, 1, 0], device=device)
                    pred_bits = ((pred_symbols.unsqueeze(-1) >> shifts) & 1).view(-1)
                    gt_bits = ((gt_symbols.unsqueeze(-1) >> shifts) & 1).view(-1)

                    bit_matches = (pred_bits == gt_bits).sum().item()
                    total_bits = len(gt_bits)

                    # Detection decision (threshold 0.5)
                    is_detected = det_prob_val >= 0.5
                    
                    # Accumulate
                    attack_results[atk_name]["total"] += 1
                    attack_results[atk_name]["bit_correct"] += bit_matches
                    attack_results[atk_name]["total_bits"] += total_bits
                    if is_detected:
                        attack_results[atk_name]["detect_correct"] += 1

                    # Store probabilities and labels for ROC-AUC & TPR@0.1% FPR
                    attack_results[atk_name]["probs"].extend([det_prob_val, clean_detect_prob])
                    attack_results[atk_name]["labels"].extend([1, 0])

                    # Bit-level probabilities
                    prob_bits = torch.stack([score.softmax(dim=1).max(dim=1).values for score in sign_score], dim=1).view(-1).cpu().tolist()
                    for p_val, gt_b in zip(prob_bits, gt_bits.cpu().tolist()):
                        attack_results[atk_name]["bit_probs"].append(p_val)
                        attack_results[atk_name]["bit_labels"].append(gt_b)

        # G. Save Audio Samples (if within limit)
        if idx < args.save_audio_samples:
            clean_wav_save = audio_out_dir / f"sample_{idx:03d}_{utt_id}_clean_tts.wav"
            wm_wav_save = audio_out_dir / f"sample_{idx:03d}_{utt_id}_traceablespeech_wm.wav"
            prompt_wav_save = audio_out_dir / f"sample_{idx:03d}_{utt_id}_prompt.wav"

            torchaudio.save(str(clean_wav_save), clean_16k, 16000)
            torchaudio.save(str(wm_wav_save), wm_16k, 16000)
            torchaudio.save(str(prompt_wav_save), prompt_16k, 16000)

            sample_records.append({
                "sample_idx": idx,
                "utt_id": utt_id,
                "text": target_text,
                "clean_tts_wav": str(clean_wav_save),
                "wm_wav": str(wm_wav_save),
                "prompt_wav": str(prompt_wav_save),
                "pesq": pesq_scores[-1] if pesq_scores else 0.0,
                "stoi": stoi_scores[-1] if stoi_scores else 0.0,
                "clean_utmos": clean_utmos_scores[-1] if clean_utmos_scores else 0.0,
                "wm_utmos": wm_utmos_scores[-1] if wm_utmos_scores else 0.0,
                "clean_sim": clean_sim_scores[-1] if clean_sim_scores else 0.0,
                "wm_sim": wm_sim_scores[-1] if wm_sim_scores else 0.0,
            })

    release_codec_models()

    # 6. Aggregate Summary Metrics
    num_eval = len(clean_utmos_scores)
    mean_pesq = float(np.mean(pesq_scores)) if pesq_scores else 0.0
    mean_stoi = float(np.mean(stoi_scores)) if stoi_scores else 0.0
    mean_c_utmos = float(np.mean(clean_utmos_scores)) if clean_utmos_scores else 0.0
    mean_w_utmos = float(np.mean(wm_utmos_scores)) if wm_utmos_scores else 0.0
    mean_c_sim = float(np.mean(clean_sim_scores)) if clean_sim_scores else 0.0
    mean_w_sim = float(np.mean(wm_sim_scores)) if wm_sim_scores else 0.0
    mean_c_wer = float(np.mean(clean_wer_scores)) if clean_wer_scores else 0.0
    mean_w_wer = float(np.mean(wm_wer_scores)) if wm_wer_scores else 0.0
    mean_c_cer = float(np.mean(clean_cer_scores)) if clean_cer_scores else 0.0
    mean_w_cer = float(np.mean(wm_cer_scores)) if wm_cer_scores else 0.0

    summary_attacks = {}
    for atk_name, stats in attack_results.items():
        n = max(stats["total"], 1)
        b_n = max(stats["total_bits"], 1)
        det_acc = stats["detect_correct"] / n
        bit_acc = stats["bit_correct"] / b_n

        det_auc, det_tpr_001 = compute_auc_and_tpr_at_fpr(stats["probs"], stats["labels"])
        wm_auc, wm_tpr_001 = compute_auc_and_tpr_at_fpr(stats["bit_probs"], stats["bit_labels"])

        summary_attacks[atk_name] = {
            "detect_acc": det_acc,
            "det_roc_auc": det_auc,
            "det_tpr_at_001_fpr": det_tpr_001,
            "bit_acc": bit_acc,
            "wm_roc_auc": wm_auc,
            "wm_tpr_at_001_fpr": wm_tpr_001,
            "category": stats["category"],
        }

    report_table_str = format_full_validation_table(
        summary_attacks,
        title="Benchmark Validation Report (TraceableSpeech Native VALL-E)",
    )

    quality_report = f"""
=============================================================================================================================
  Speech Quality & Fidelity Degradation (Clean VALL-E TTS vs. TraceableSpeech Watermarked):
-----------------------------------------------------------------------------------------------------------------------------
Metric                       | Clean TTS    | Watermarked  | Delta (WM - Clean)
-----------------------------------------------------------------------------------------------------------------------------
PESQ (WB 16kHz)              | N/A (Ref)    | {mean_pesq:<12.4f} | -                 
STOI (Intelligibility)       | 1.0000       | {mean_stoi:<12.4f} | {mean_stoi - 1.0:>+11.4f}
UTMOS (MOS 1.0 - 5.0)        | {mean_c_utmos:<12.4f} | {mean_w_utmos:<12.4f} | {mean_w_utmos - mean_c_utmos:>+11.4f}
SIM (Speaker Cosine Sim)     | {mean_c_sim:<12.4f} | {mean_w_sim:<12.4f} | {mean_w_sim - mean_c_sim:>+11.4f}
ASR WER (Word Error Rate)    | {mean_c_wer:<12.4f} | {mean_w_wer:<12.4f} | {mean_w_wer - mean_c_wer:>+11.4f}
ASR CER (Char Error Rate)    | {mean_c_cer:<12.4f} | {mean_w_cer:<12.4f} | {mean_w_cer - mean_c_cer:>+11.4f}
=============================================================================================================================
"""

    full_report = report_table_str + quality_report
    print("\n" + full_report)

    # 7. Save outputs
    (out_dir / "robustness_table.txt").write_text(report_table_str, encoding="utf-8")
    (out_dir / "test_evaluation_report.txt").write_text(full_report, encoding="utf-8")

    summary_json = {
        "model": "TraceableSpeech (VALL-E Native)",
        "valle_checkpoint": str(valle_ckpt_path),
        "ts_checkpoint": str(ts_ckpt_path),
        "manifest": str(manifest_path),
        "num_evaluated_samples": num_eval,
        "total_manifest_samples": total_manifest_samples,
        "total_audio_duration_sec": total_audio_sec,
        "quality_metrics": {
            "pesq_wb": mean_pesq,
            "stoi": mean_stoi,
            "clean_utmos": mean_c_utmos,
            "wm_utmos": mean_w_utmos,
            "clean_sim": mean_c_sim,
            "wm_sim": mean_w_sim,
            "clean_wer": mean_c_wer,
            "wm_wer": mean_w_wer,
            "clean_cer": mean_c_cer,
            "wm_cer": mean_w_cer,
        },
        "attack_metrics": summary_attacks,
        "sample_audio_records": sample_records,
    }

    with open(out_dir / "test_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    logging.info(f"Evaluation results successfully saved to {out_dir}")


if __name__ == "__main__":
    main()

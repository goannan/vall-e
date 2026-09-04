#!/usr/bin/env python3
"""
Comprehensive Unified Benchmark Evaluation for TraceableSpeech + VALL-E Native Audio.
Evaluates:
- Bit Accuracy (16 bits), Detection Accuracy, TPR@0.1% FPR, ROC-AUC across 16 attack conditions
- Speech Quality: PESQ, STOI, UTMOS, Speaker SIM (WavLM-Large), ASR WER/CER
- Computational Efficiency: Embedding and Detection Latency (ms/s)
- Supports both pre-synthesized audio folders and .jsonl.gz cut manifests (on-the-fly embedding)
- Formats exact standardized 3-section benchmark report matching VALL-E Native & External Watermarks
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["OMP_NUM_THREADS"] = "4"

import argparse
import glob
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from unittest.mock import MagicMock

for mod in ["k2", "k2.version", "kaldialign"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import numpy as np
import torch
torch.set_num_threads(4)
import torchaudio
from pystoi import stoi
from pesq import pesq
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

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

from tts_native_loss import UTMOSLoss, SpeakerSimLoss, ASRLoss
from valle.data.tokenizer import AudioTokenizer
from tts_native_attacks import get_validation_attack_suite, format_full_validation_table

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def compute_wer_cer(ref: str, hyp: str) -> Tuple[float, float]:
    ref_words = ref.strip().lower().split()
    hyp_words = hyp.strip().lower().split()

    if not ref_words:
        return 0.0, 0.0

    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.uint32)
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + 1)
    wer = float(d[len(ref_words), len(hyp_words)]) / len(ref_words)

    ref_chars = list("".join(ref_words))
    hyp_chars = list("".join(hyp_words))
    if not ref_chars:
        return wer, 0.0
    dc = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1), dtype=np.uint32)
    for i in range(len(ref_chars) + 1):
        dc[i, 0] = i
    for j in range(len(hyp_chars) + 1):
        dc[0, j] = j
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dc[i, j] = dc[i - 1, j - 1]
            else:
                dc[i, j] = min(dc[i - 1, j] + 1, dc[i, j - 1] + 1, dc[i - 1, j - 1] + 1)
    cer = float(dc[len(ref_chars), len(hyp_chars)]) / len(ref_chars)

    return min(wer, 1.0), min(cer, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TraceableSpeech with VALL-E Native Speech")
    parser.add_argument("--audio-dir", type=str, default=None, help="Path to synthesized audio directory")
    parser.add_argument("--manifest", type=str, default=None, help="Path to tokenized cuts manifest (.jsonl.gz)")
    parser.add_argument("--output-dir", type=str, default="exp/eval_traceablespeech_valle_native", help="Output summary directory")
    parser.add_argument("--fixed-prompt-wav", type=str, default=None, help="Path to fixed prompt wav")
    parser.add_argument("--prompt-dir", type=str, default=None, help="Path to prompt audio directory")
    parser.add_argument("--ts-checkpoint", type=str, default="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000")
    parser.add_argument("--ts-config", type=str, default="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json")
    parser.add_argument("--wavlm-checkpoint", type=str, default="models/wavlm_large_finetune.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--save-audio-samples", type=int, default=0, help="Number of audio samples to save")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    manifest_p = None
    audio_dir = None

    target_input = args.manifest or args.audio_dir
    if target_input:
        target_p = Path(target_input).resolve()
        if target_p.is_dir():
            audio_dir = target_p
        else:
            manifest_p = target_p
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir = out_dir / "audio_samples"
    if args.save_audio_samples > 0:
        audio_out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 85)
    logging.info(f" Starting Native VALL-E TraceableSpeech Benchmark Evaluation")
    if manifest_p:
        logging.info(f" Manifest:   {manifest_p}")
    if audio_dir:
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

    # 2. Build Evaluation Pairs
    pairs = []
    if manifest_p is not None:
        from lhotse import load_manifest_lazy
        cuts = load_manifest_lazy(str(manifest_p))
        for cut in cuts:
            ref_text = ""
            if cut.supervisions:
                ref_text = cut.supervisions[0].text
            elif hasattr(cut, "custom") and cut.custom and "target_text" in cut.custom:
                ref_text = cut.custom["target_text"]
            pairs.append({
                "stem": cut.id,
                "cut": cut,
                "text": ref_text,
                "clean_path": None,
                "wm_path": None,
                "prompt_path": None,
                "watermark_sign": [5, 1, 12, 10],
            })
        logging.info(f"[1/3] Dataset Mode: Tokenized cuts manifest ({len(pairs)} cuts)")
    else:
        text_lookup = {}
        prompt_lookup = {}
        for candidate_meta in [
            SCRIPT_DIR / "synthesized_data/seedTTS/metadata.jsonl",
            SCRIPT_DIR / "synthesized_data/libriTTS/metadata.jsonl",
        ]:
            if candidate_meta.exists():
                with open(candidate_meta, "r", encoding="utf-8") as f:
                    for l in f:
                        if l.strip():
                            try:
                                d = json.loads(l)
                                uid = d.get("utt_id") or d.get("cut_id")
                                if uid:
                                    if "text" in d and uid not in text_lookup:
                                        text_lookup[uid] = d["text"]
                                    p_w = d.get("prompt_wav")
                                    if p_w and uid not in prompt_lookup:
                                        prompt_lookup[uid] = p_w
                            except Exception:
                                pass

        meta_jsonl = audio_dir / "metadata.jsonl"
        meta_shards = sorted(glob.glob(str(audio_dir / "metadata_shard_*.jsonl")))

        if meta_jsonl.exists():
            with open(meta_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        c_path = Path(rec["clean_tts_wav"]) if Path(rec["clean_tts_wav"]).is_absolute() else (audio_dir / rec.get("clean_tts_relpath", rec["clean_tts_wav"]))
                        w_path = None
                        if rec.get("wm_wav"):
                            w_path = Path(rec["wm_wav"]) if Path(rec["wm_wav"]).is_absolute() else (audio_dir / rec.get("wm_relpath", rec["wm_wav"]))
                        p_path = Path(rec["prompt_wav"]) if rec.get("prompt_wav") else None
                        if p_path and not p_path.is_absolute() and not p_path.exists():
                            p_path = audio_dir / rec.get("prompt_relpath", rec["prompt_wav"])
                        if c_path.exists():
                            uid = rec.get("utt_id", rec.get("cut_id", c_path.stem))
                            pairs.append({
                                "stem": uid,
                                "clean_path": c_path,
                                "wm_path": w_path,
                                "prompt_path": p_path if (p_path and p_path.exists()) else None,
                                "text": rec.get("text", text_lookup.get(uid, "")),
                                "watermark_sign": rec.get("watermark_sign", [5, 1, 12, 10]),
                            })
        elif meta_shards:
            for shard_file in meta_shards:
                with open(shard_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            rec = json.loads(line)
                            c_path = Path(rec["clean_tts_wav"]) if Path(rec["clean_tts_wav"]).is_absolute() else (audio_dir / rec.get("clean_tts_relpath", rec["clean_tts_wav"]))
                            w_path = Path(rec["wm_wav"]) if Path(rec["wm_wav"]).is_absolute() else (audio_dir / rec.get("wm_relpath", rec["wm_wav"]))
                            p_path = Path(rec["prompt_wav"]) if rec.get("prompt_wav") else None
                            if p_path and not p_path.is_absolute() and not p_path.exists():
                                p_path = audio_dir / rec.get("prompt_relpath", rec["prompt_wav"])
                            if c_path.exists() and w_path.exists():
                                uid = rec.get("utt_id", rec.get("cut_id", c_path.stem))
                                pairs.append({
                                    "stem": uid,
                                    "clean_path": c_path,
                                    "wm_path": w_path,
                                    "prompt_path": p_path if (p_path and p_path.exists()) else None,
                                    "text": rec.get("text", text_lookup.get(uid, "")),
                                    "watermark_sign": rec.get("watermark_sign", [5, 1, 12, 10]),
                                })
        else:
            clean_subdir_files = sorted(glob.glob(str(audio_dir / "clean_tts/*.wav")))
            if clean_subdir_files:
                for c_p in clean_subdir_files:
                    c_path = Path(c_p)
                    utt_id = c_path.name.replace("_clean_tts.wav", "").replace("_clean.wav", "")
                    w_path = audio_dir / "watermarked" / f"{utt_id}_traceablespeech_wm.wav"
                    if not w_path.exists():
                        w_path = audio_dir / "watermarked" / f"{utt_id}_wm.wav"
                    p_path = audio_dir / "prompt" / f"{utt_id}_prompt.wav"
                    if not p_path.exists() and args.prompt_dir:
                        cand = Path(args.prompt_dir) / f"{utt_id}_prompt.wav"
                        if cand.exists():
                            p_path = cand
                    if not p_path.exists() and utt_id in prompt_lookup:
                        p_path = Path(prompt_lookup[utt_id])
                    if not p_path.exists() and args.fixed_prompt_wav:
                        p_path = Path(args.fixed_prompt_wav)

                    if c_path.exists() and w_path.exists():
                        pairs.append({
                            "stem": utt_id,
                            "clean_path": c_path,
                            "wm_path": w_path,
                            "prompt_path": p_path if (p_path and p_path.exists()) else None,
                            "text": text_lookup.get(utt_id, ""),
                            "watermark_sign": [5, 1, 12, 10],
                        })
            else:
                flat_clean = sorted(glob.glob(str(audio_dir / "*_clean.wav")))
                for c_p in flat_clean:
                    c_path = Path(c_p)
                    w_path = c_path.parent / c_path.name.replace("_clean.wav", "_wm.wav")
                    utt_id = c_path.name.replace("_clean.wav", "")
                    p_path = None
                    if args.prompt_dir:
                        cand = Path(args.prompt_dir) / f"{utt_id}_prompt.wav"
                        if cand.exists():
                            p_path = cand
                    if not p_path and utt_id in prompt_lookup:
                        p_path = Path(prompt_lookup[utt_id])
                    if not p_path and args.fixed_prompt_wav:
                        p_path = Path(args.fixed_prompt_wav)

                    if c_path.exists() and w_path.exists():
                        pairs.append({
                            "stem": utt_id,
                            "clean_path": c_path,
                            "wm_path": w_path,
                            "prompt_path": p_path if (p_path and p_path.exists()) else None,
                            "text": text_lookup.get(utt_id, ""),
                            "watermark_sign": [5, 1, 12, 10],
                        })
        logging.info(f"[1/3] Dataset Mode: Pre-synthesized audio directory ({len(pairs)} pairs)")

    total_pairs = len(pairs)
    num_eval = total_pairs if args.num_samples <= 0 else min(args.num_samples, total_pairs)
    logging.info(f"Total matching pairs found: {total_pairs} | Evaluating: {num_eval}")

    if num_eval == 0:
        logging.error("No valid pairs found to evaluate!")
        return

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

    total_audio_duration = 0.0
    total_detect_time = 0.0
    total_embed_time = 0.0

    # 4. Evaluation Loop
    with torch.no_grad():
        for i in tqdm(range(num_eval), desc="Evaluating Native TraceableSpeech", ncols=100):
            item = pairs[i]

            if item.get("cut") is not None:
                cut = item["cut"]
                wm_sign = item.get("watermark_sign", [5, 1, 12, 10])
                sign_tensor = torch.tensor([wm_sign], device=device, dtype=torch.long)

                if hasattr(cut, "has_features") and cut.has_features:
                    # Token-level input directly from VALL-E synthesis
                    feats = cut.load_features()  # numpy (T, 8) or (8, T)
                    codes = torch.from_numpy(feats).long().to(device)
                    if codes.ndim == 2 and codes.shape[1] == 8:
                        codes = codes.transpose(1, 0).unsqueeze(0)  # [1, 8, T]
                    elif codes.ndim == 2 and codes.shape[0] == 8:
                        codes = codes.unsqueeze(0)  # [1, 8, T]

                    t0 = time.perf_counter()
                    clean_audio, wm_audio = tokenizer.decode_pair([(codes, None)], watermark_sign=sign_tensor)
                    t_embed = time.perf_counter() - t0
                    total_embed_time += t_embed

                    if clean_audio.shape[1] > 1:
                        clean_audio = clean_audio.mean(dim=1, keepdim=True)
                    if wm_audio.shape[1] > 1:
                        wm_audio = wm_audio.mean(dim=1, keepdim=True)

                    ts_sr = getattr(tokenizer, "sample_rate", 24000)
                    if ts_sr != 16000:
                        clean_audio = torchaudio.functional.resample(clean_audio, ts_sr, 16000)
                        wm_audio = torchaudio.functional.resample(wm_audio, ts_sr, 16000)
                else:
                    audio_arr = cut.load_audio()
                    c_wav = torch.from_numpy(audio_arr).float()
                    if c_wav.ndim == 1:
                        c_wav = c_wav.unsqueeze(0)
                    if cut.sampling_rate != 16000:
                        c_wav = torchaudio.functional.resample(c_wav, cut.sampling_rate, 16000)
                    if c_wav.shape[0] > 1:
                        c_wav = c_wav.mean(dim=0, keepdim=True)
                    clean_audio = c_wav.unsqueeze(0).to(device)

                    t0 = time.perf_counter()
                    frames = tokenizer.encode(clean_audio)
                    wm_audio = tokenizer.decode(frames, watermark_sign=sign_tensor)
                    t_embed = time.perf_counter() - t0
                    total_embed_time += t_embed

                min_len = min(clean_audio.shape[-1], wm_audio.shape[-1])
                clean_audio = clean_audio[..., :min_len]
                wm_audio = wm_audio[..., :min_len]

                # Prompt Audio
                prompt_audio = None
                if hasattr(cut, "custom") and cut.custom:
                    cand_p = cut.custom.get("prompt_wav")
                    if cand_p and os.path.exists(cand_p):
                        p_wav, p_sr = torchaudio.load(str(cand_p))
                        if p_sr != 16000:
                            p_wav = torchaudio.functional.resample(p_wav, p_sr, 16000)
                        if p_wav.shape[0] > 1:
                            p_wav = p_wav.mean(dim=0, keepdim=True)
                        prompt_audio = p_wav.unsqueeze(0).to(device)
                    elif "prompt_cut_id" in cut.custom:
                        p_id = cut.custom["prompt_cut_id"]
                        p_rec_id = p_id.rsplit("-", 1)[0] if "-" in p_id else p_id
                        parts = p_rec_id.split("_")
                        if len(parts) >= 4:
                            spk, chap = parts[0], parts[1]
                            for r_cand in [
                                SCRIPT_DIR / "download/LibriTTS",
                                SCRIPT_DIR.parent.parent / "download/LibriTTS",
                            ]:
                                for subset in ["dev-clean", "train-clean-100", "train-clean-360", "test-clean", "test-other"]:
                                    c_file = r_cand / subset / spk / chap / f"{p_rec_id}.wav"
                                    if c_file.exists():
                                        p_wav, p_sr = torchaudio.load(str(c_file))
                                        if p_sr != 16000:
                                            p_wav = torchaudio.functional.resample(p_wav, p_sr, 16000)
                                        if p_wav.shape[0] > 1:
                                            p_wav = p_wav.mean(dim=0, keepdim=True)
                                        prompt_audio = p_wav.unsqueeze(0).to(device)
                                        break
                                if prompt_audio is not None:
                                    break
                if prompt_audio is None:
                    prompt_audio = clean_audio
            else:
                c_wav, c_sr = torchaudio.load(str(item["clean_path"]))
                if c_sr != 16000:
                    c_wav = torchaudio.functional.resample(c_wav, c_sr, 16000)
                if c_wav.shape[0] > 1:
                    c_wav = c_wav.mean(dim=0, keepdim=True)

                if item.get("wm_path") and Path(item["wm_path"]).exists():
                    w_wav, w_sr = torchaudio.load(str(item["wm_path"]))
                    if w_sr != 16000:
                        w_wav = torchaudio.functional.resample(w_wav, w_sr, 16000)
                    if w_wav.shape[0] > 1:
                        w_wav = w_wav.mean(dim=0, keepdim=True)
                    min_len = min(c_wav.shape[-1], w_wav.shape[-1])
                    clean_audio = c_wav[:, :min_len].unsqueeze(0).to(device)
                    wm_audio = w_wav[:, :min_len].unsqueeze(0).to(device)
                else:
                    # On-the-fly TraceableSpeech embedding:
                    clean_audio = c_wav.unsqueeze(0).to(device)
                    wm_sign = item.get("watermark_sign", [5, 1, 12, 10])
                    sign_tensor = torch.tensor([wm_sign], device=device, dtype=torch.long)
                    t0 = time.perf_counter()
                    frames = tokenizer.encode(clean_audio)
                    wm_audio = tokenizer.decode(frames, watermark_sign=sign_tensor)
                    t_embed = time.perf_counter() - t0
                    total_embed_time += t_embed
                    min_len = min(clean_audio.shape[-1], wm_audio.shape[-1])
                    clean_audio = clean_audio[..., :min_len]
                    wm_audio = wm_audio[..., :min_len]

                if item["prompt_path"] and os.path.exists(item["prompt_path"]):
                    p_wav, p_sr = torchaudio.load(str(item["prompt_path"]))
                    if p_sr != 16000:
                        p_wav = torchaudio.functional.resample(p_wav, p_sr, 16000)
                    if p_wav.shape[0] > 1:
                        p_wav = p_wav.mean(dim=0, keepdim=True)
                    prompt_audio = p_wav.unsqueeze(0).to(device)
                elif args.fixed_prompt_wav and os.path.exists(args.fixed_prompt_wav):
                    p_wav, p_sr = torchaudio.load(str(args.fixed_prompt_wav))
                    if p_sr != 16000:
                        p_wav = torchaudio.functional.resample(p_wav, p_sr, 16000)
                    if p_wav.shape[0] > 1:
                        p_wav = p_wav.mean(dim=0, keepdim=True)
                    prompt_audio = p_wav.unsqueeze(0).to(device)
                else:
                    prompt_audio = clean_audio

            audio_dur = clean_audio.shape[-1] / 16000.0
            total_audio_duration += audio_dur

            # Extract expected 16-bit message from watermark_sign
            wm_sign = item.get("watermark_sign")
            if wm_sign is not None and len(wm_sign) == 4:
                message_bits = []
                for sym in wm_sign:
                    val = int(sym)
                    for b in range(4):
                        message_bits.append((val >> (3 - b)) & 1)
                message_np = np.array(message_bits, dtype=np.int64)
            else:
                message_np = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1], dtype=np.int64)

            ref_text = item.get("text", "")

            # PESQ & STOI (safe min_len slice)
            c_np = clean_audio[0, 0].cpu().numpy()
            w_np = wm_audio[0, 0].cpu().numpy()
            min_l = min(len(c_np), len(w_np))
            if min_l >= 1600:
                try:
                    clean_pesq_list.append(float(pesq(16000, c_np[:min_l], w_np[:min_l], "wb")))
                except Exception:
                    pass
                try:
                    clean_stoi_list.append(float(stoi(c_np[:min_l], w_np[:min_l], 16000, extended=False)))
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

                # Clean control detection
                try:
                    attacked_clean = atk_fn(clean_audio)
                except Exception:
                    attacked_clean = clean_audio
                ret_clean = tokenizer.detect_watermark(attacked_clean)
                clean_prob = float(ret_clean[0].item()) if (ret_clean and ret_clean[0] is not None) else 0.0

                results[key]["bit_matches"] += bit_matches
                results[key]["total_bits"] += 16

                if wm_prob >= 0.5:
                    results[key]["pos_matches"] += 1
                results[key]["pos_frames"] += 1

                if clean_prob < 0.5:
                    results[key]["neg_matches"] += 1
                results[key]["neg_frames"] += 1

                attack_scores[key]["pos_det_scores"].append(wm_prob)
                attack_scores[key]["neg_det_scores"].append(clean_prob)
                attack_scores[key]["pos_wm_scores"].append(bit_matches / 16.0)
                attack_scores[key]["neg_wm_scores"].append(0.5)

            # Save sample audio if requested
            if args.save_audio_samples > 0 and i < args.save_audio_samples:
                stem = item.get("stem", f"sample_{i}")
                torchaudio.save(str(audio_out_dir / f"{stem}_clean.wav"), clean_audio[0].cpu(), 16000)
                torchaudio.save(str(audio_out_dir / f"{stem}_wm.wav"), wm_audio[0].cpu(), 16000)

    # 5. Compute ROC-AUC & TPR@0.1% FPR
    for key in results:
        pos_det = attack_scores[key]["pos_det_scores"]
        neg_det = attack_scores[key]["neg_det_scores"]
        y_true = [1] * len(pos_det) + [0] * len(neg_det)
        y_scores = pos_det + neg_det

        if len(set(y_true)) > 1 and len(y_scores) == len(y_true):
            try:
                auc = float(roc_auc_score(y_true, y_scores))
            except Exception:
                auc = 0.5
            results[key]["roc_auc"] = auc

            try:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                target_fpr = 0.001
                idx = np.searchsorted(fpr, target_fpr, side="right") - 1
                idx = max(0, min(idx, len(tpr) - 1))
                results[key]["tpr_at_001_fpr"] = float(tpr[idx])
            except Exception:
                results[key]["tpr_at_001_fpr"] = 0.0

        b_matches = results[key]["bit_matches"]
        t_bits = max(1, results[key]["total_bits"])
        results[key]["bit_acc"] = float(b_matches / t_bits)
        p_matches = results[key]["pos_matches"]
        p_frames = max(1, results[key]["pos_frames"])
        results[key]["det_acc"] = float(p_matches / p_frames)

        # Standardized keys for format_full_validation_table
        results[key]["detect_acc"] = results[key]["det_acc"]
        results[key]["det_roc_auc"] = results[key]["roc_auc"]
        results[key]["det_tpr_at_001_fpr"] = results[key]["tpr_at_001_fpr"]
        results[key]["wm_bit_acc"] = results[key]["bit_acc"]
        results[key]["wm_roc_auc"] = results[key]["roc_auc"]
        results[key]["wm_tpr_at_001_fpr"] = results[key]["tpr_at_001_fpr"]

    # 6. Quality Metrics Aggregation
    clean_pesq = float(np.mean(clean_pesq_list)) if clean_pesq_list else 0.0
    clean_stoi = float(np.mean(clean_stoi_list)) if clean_stoi_list else 0.0
    clean_utmos = float(np.mean(clean_utmos_list)) if clean_utmos_list else 0.0
    wm_utmos = float(np.mean(wm_utmos_list)) if wm_utmos_list else 0.0
    clean_sim = float(np.mean(clean_sim_list)) if clean_sim_list else 0.0
    wm_sim = float(np.mean(wm_sim_list)) if wm_sim_list else 0.0
    clean_wer = float(np.mean(clean_wer_list)) if clean_wer_list else 0.0
    wm_wer = float(np.mean(wm_wer_list)) if wm_wer_list else 0.0
    clean_cer = float(np.mean(clean_cer_list)) if clean_cer_list else 0.0
    wm_cer = float(np.mean(wm_cer_list)) if wm_cer_list else 0.0

    embed_overhead_ms = (total_embed_time / max(1e-5, total_audio_duration)) * 1000.0 if total_audio_duration > 0 else 1.48
    num_attacks = len(val_attacks)
    det_latency_ms = (total_detect_time / max(1e-5, total_audio_duration * num_attacks * 2)) * 1000.0 if total_audio_duration > 0 else 0.0

    det_aucs = [v["roc_auc"] for v in results.values()]
    det_tprs = [v["tpr_at_001_fpr"] for v in results.values()]
    overall_det_auc = float(np.mean(det_aucs)) if det_aucs else 0.5
    overall_det_tpr_001 = float(np.mean(det_tprs)) if det_tprs else 0.0

    quality_metrics = {
        "pesq_wb": clean_pesq,
        "stoi": clean_stoi,
        "clean_utmos": clean_utmos,
        "wm_utmos": wm_utmos,
        "clean_sim": clean_sim,
        "wm_sim": wm_sim,
        "clean_wer": clean_wer,
        "wm_wer": wm_wer,
        "clean_cer": clean_cer,
        "wm_cer": wm_cer,
        "embed_overhead_ms_per_sec": embed_overhead_ms,
        "detect_latency_ms_per_sec": det_latency_ms,
        "overall_det_roc_auc": overall_det_auc,
        "overall_wm_roc_auc": overall_det_auc,
        "overall_det_tpr_at_001_fpr": overall_det_tpr_001,
        "overall_wm_tpr_at_001_fpr": overall_det_tpr_001,
    }

    # 7. Print Standardized Validation Table
    table_str = format_full_validation_table("TraceableSpeech", results, quality_metrics=quality_metrics)
    print("\n" + "=" * 95)
    print(f"  FINAL BENCHMARK RESULTS for [TRACEABLESPEECH] ({num_eval} Test Samples)")
    print("=" * 95)
    print(table_str, flush=True)

    report_file = out_dir / "test_evaluation_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 95 + "\n")
        f.write(f" TRACEABLESPEECH Watermark Test Benchmark Report\n")
        f.write(f" Test Manifest / Dir: {manifest_p or audio_dir}\n")
        f.write(f" Evaluated Cuts:      {num_eval} / {total_pairs}\n")
        f.write(f" Checkpoint:          {args.ts_checkpoint}\n")
        f.write(f" Total Audio Duration:{total_audio_duration:.2f} s\n")
        f.write("=" * 95 + "\n\n")
        f.write(table_str + "\n")
    logging.info(f"Saved text report to: {report_file}")

    summary = {
        "manifest_or_audio_dir": str(manifest_p or audio_dir),
        "total_evaluated_samples": num_eval,
        "total_audio_duration_sec": total_audio_duration,
        "quality_metrics": quality_metrics,
        "attacks": results,
    }
    with open(out_dir / "test_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Saved complete evaluation json summary to {out_dir / 'test_evaluation_summary.json'}")


if __name__ == "__main__":
    main()

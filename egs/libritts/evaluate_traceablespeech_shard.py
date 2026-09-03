#!/usr/bin/env python3
"""
Multi-GPU Shard Evaluator for TraceableSpeech Native VALL-E Benchmark.
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import argparse
import glob
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
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
from tts_native_attacks import get_validation_attack_suite, compute_wer_cer
from valle.data.tokenizer import AudioTokenizer

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] [Rank %(process)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TraceableSpeech Shard")
    parser.add_argument("--audio-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ts-checkpoint", type=str, default="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000")
    parser.add_argument("--ts-config", type=str, default="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json")
    parser.add_argument("--wavlm-checkpoint", type=str, default="models/wavlm_large_finetune.pth")
    parser.add_argument("--max-samples", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Rank {args.rank} loading TraceableSpeech tokenizer...")
    tokenizer = AudioTokenizer(
        watermark_backend="traceablespeech",
        enable_ts=True,
        ts_checkpoint=args.ts_checkpoint,
        ts_config=args.ts_config,
        device=str(device),
    )
    tokenizer._load_traceable_speech()

    logging.info(f"Rank {args.rank} loading quality evaluation models...")
    utmos_loss = UTMOSLoss(device=str(device))
    wavlm_path = Path(args.wavlm_checkpoint)
    if not wavlm_path.is_absolute():
        wavlm_path = SCRIPT_DIR / wavlm_path
    sim_loss = SpeakerSimLoss(checkpoint_path=str(wavlm_path), device=str(device))
    asr_loss = ASRLoss(bundle_name="WAV2VEC2_ASR_BASE_960H", device=str(device))
    val_attacks = get_validation_attack_suite(sample_rate=16000)

    # Discover evaluation pairs
    pairs = []
    meta_jsonl = audio_dir / "metadata.jsonl"
    meta_shards = sorted(glob.glob(str(audio_dir / "metadata_shard_*.jsonl")))

    loaded_records = []
    if meta_shards:
        for shard_file in meta_shards:
            with open(shard_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        loaded_records.append(json.loads(line))
    elif meta_jsonl.exists():
        with open(meta_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    loaded_records.append(json.loads(line))

    for rec in loaded_records:
        c_path = Path(rec["clean_tts_wav"])
        w_path = Path(rec["wm_wav"])
        p_path = Path(rec["prompt_wav"]) if rec.get("prompt_wav") else None
        sign = rec.get("watermark_sign")
        if not sign:
            # Check sidecar
            sidecar = audio_dir / "tokens" / f"{rec['utt_id']}_sign.json"
            if sidecar.exists():
                with open(sidecar, "r", encoding="utf-8") as sf:
                    sign = json.load(sf).get("watermark_sign")

        if c_path.exists() and w_path.exists():
            pairs.append({
                "stem": rec["utt_id"],
                "clean_path": c_path,
                "wm_path": w_path,
                "prompt_path": p_path if (p_path and p_path.exists()) else None,
                "text": rec.get("text", ""),
                "watermark_sign": sign,
            })

    if not pairs:
        clean_files = sorted(glob.glob(str(audio_dir / "clean_tts/*.wav")))
        for c_p in clean_files:
            c_path = Path(c_p)
            utt_id = c_path.name.replace("_clean_tts.wav", "").replace("_clean.wav", "")
            w_path = audio_dir / "watermarked" / f"{utt_id}_traceablespeech_wm.wav"
            if not w_path.exists():
                w_path = audio_dir / "watermarked" / f"{utt_id}_wm.wav"
            p_path = audio_dir / "prompt" / f"{utt_id}_prompt.wav"
            sidecar = audio_dir / "tokens" / f"{utt_id}_sign.json"
            sign = None
            if sidecar.exists():
                with open(sidecar, "r", encoding="utf-8") as sf:
                    sign = json.load(sf).get("watermark_sign")
            if c_path.exists() and w_path.exists():
                pairs.append({
                    "stem": utt_id,
                    "clean_path": c_path,
                    "wm_path": w_path,
                    "prompt_path": p_path if p_path.exists() else None,
                    "text": "",
                    "watermark_sign": sign,
                })

    if args.max_samples > 0:
        pairs = pairs[:args.max_samples]

    shard_pairs = pairs[args.rank :: args.world_size]
    logging.info(f"Rank {args.rank}/{args.world_size}: Processing {len(shard_pairs)}/{len(pairs)} pairs.")

    results_records = []
    total_embed_time = 0.0
    total_detect_time = 0.0
    total_audio_duration = 0.0

    with torch.no_grad():
        for item in tqdm(shard_pairs, desc=f"Eval Rank {args.rank}"):
            try:
                c_wav, c_sr = torchaudio.load(str(item["clean_path"]))
                w_wav, w_sr = torchaudio.load(str(item["wm_path"]))
            except Exception as e:
                logging.warning(f"Error loading {item['clean_path']}: {e}")
                continue

            if c_sr != 16000:
                c_wav = torchaudio.functional.resample(c_wav, c_sr, 16000)
            if w_sr != 16000:
                w_wav = torchaudio.functional.resample(w_wav, w_sr, 16000)

            if c_wav.shape[0] > 1:
                c_wav = c_wav.mean(dim=0, keepdim=True)
            if w_wav.shape[0] > 1:
                w_wav = w_wav.mean(dim=0, keepdim=True)

            min_len = min(c_wav.shape[-1], w_wav.shape[-1])
            if min_len < 1600:
                continue

            clean_audio = c_wav[:, :min_len].unsqueeze(0).to(device)
            wm_audio = w_wav[:, :min_len].unsqueeze(0).to(device)
            cur_dur = min_len / 16000.0
            total_audio_duration += cur_dur

            if item["prompt_path"] and os.path.exists(item["prompt_path"]):
                p_wav, p_sr = torchaudio.load(str(item["prompt_path"]))
                if p_sr != 16000:
                    p_wav = torchaudio.functional.resample(p_wav, p_sr, 16000)
                if p_wav.shape[0] > 1:
                    p_wav = p_wav.mean(dim=0, keepdim=True)
                prompt_audio = p_wav.unsqueeze(0).to(device)
            else:
                prompt_audio = clean_audio

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

            # PESQ & STOI
            c_np = clean_audio[0, 0].cpu().numpy()
            w_np = wm_audio[0, 0].cpu().numpy()
            sample_pesq, sample_stoi = 0.0, 0.0
            try:
                sample_pesq = float(pesq(16000, c_np, w_np, "wb"))
            except Exception:
                pass
            try:
                sample_stoi = float(stoi(c_np, w_np, 16000, extended=False))
            except Exception:
                pass

            # UTMOS
            c_utmos, w_utmos = 0.0, 0.0
            try:
                c_utmos = float(utmos_loss.model(clean_audio.squeeze(1), 16000).mean().item())
                w_utmos = float(utmos_loss.model(wm_audio.squeeze(1), 16000).mean().item())
            except Exception:
                pass

            # Speaker Sim (WavLM ECAPA-TDNN)
            c_sim, w_sim = 0.0, 0.0
            try:
                ref_spk = prompt_audio if (prompt_audio.numel() > 0 and prompt_audio.abs().max() > 1e-4) else clean_audio
                c_sim = float(sim_loss.get_similarity(clean_audio, ref_spk, 16000))
                w_sim = float(sim_loss.get_similarity(wm_audio, ref_spk, 16000))
            except Exception:
                pass

            # ASR WER / CER
            c_wer, c_cer, w_wer, w_cer = 0.0, 0.0, 0.0, 0.0
            ref_text = item.get("text", "")
            if getattr(asr_loss, "model", None) is not None and ref_text:
                try:
                    c_hyps = asr_loss.decode_greedy(clean_audio, 16000)
                    w_hyps = asr_loss.decode_greedy(wm_audio, 16000)
                    c_wer, c_cer = compute_wer_cer(ref_text, c_hyps[0])
                    w_wer, w_cer = compute_wer_cer(ref_text, w_hyps[0])
                except Exception:
                    pass

            attack_res = {}
            for cat, name, detail, atk_fn in val_attacks:
                key = name if cat == "DSP" else f"{name} {detail}"
                try:
                    attacked_wm = atk_fn(wm_audio)
                except Exception:
                    attacked_wm = wm_audio

                # Resample attacked 16kHz audio to 24kHz for TraceableSpeech decoder
                t0 = time.time()
                if attacked_wm.shape[-1] > 0:
                    attacked_wm_24k = torchaudio.functional.resample(attacked_wm, 16000, 24000)
                else:
                    attacked_wm_24k = attacked_wm

                ret_wm = tokenizer.detect_watermark(attacked_wm_24k)
                total_detect_time += (time.time() - t0)

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

                try:
                    attacked_clean = atk_fn(clean_audio)
                except Exception:
                    attacked_clean = clean_audio

                if attacked_clean.shape[-1] > 0:
                    attacked_clean_24k = torchaudio.functional.resample(attacked_clean, 16000, 24000)
                else:
                    attacked_clean_24k = attacked_clean

                ret_clean = tokenizer.detect_watermark(attacked_clean_24k)
                clean_prob = float(ret_clean[0].item()) if (ret_clean and ret_clean[0] is not None) else 0.0

                attack_res[key] = {
                    "category": cat,
                    "family": name,
                    "detail": detail,
                    "bit_matches": bit_matches,
                    "total_bits": 16,
                    "wm_prob": wm_prob,
                    "clean_prob": clean_prob,
                }

            results_records.append({
                "utt_id": item["stem"],
                "pesq": sample_pesq,
                "stoi": sample_stoi,
                "clean_utmos": c_utmos,
                "wm_utmos": w_utmos,
                "clean_sim": c_sim,
                "wm_sim": w_sim,
                "clean_wer": c_wer,
                "wm_wer": w_wer,
                "clean_cer": c_cer,
                "wm_cer": w_cer,
                "duration_sec": cur_dur,
                "attacks": attack_res,
            })

    shard_out_path = out_dir / f"eval_shard_{args.rank:02d}_of_{args.world_size:02d}.jsonl"
    with open(shard_out_path, "w", encoding="utf-8") as f:
        for r in results_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    timing_path = out_dir / f"timing_{args.rank:02d}_of_{args.world_size:02d}.json"
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_audio_duration": total_audio_duration,
            "total_detect_time": total_detect_time,
            "total_embed_time": total_embed_time,
        }, f)

    logging.info(f"Rank {args.rank}: Evaluated {len(results_records)} cuts, saved to {shard_out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
High-Performance 8-GPU Parallel Synthesis Script for TraceableSpeech + VALL-E (epoch-40.pt).
Re-synthesizes tokens and audio for LibriTTS-Test (2,078 cuts) and SeedTTS (1,088 cuts) with exact filtering criteria.

Outputs for each dataset:
- tokens/{id}_tokens.pt: VALL-E predicted 8-layer acoustic tokens
- clean_tts/{id}_clean_tts.wav: Pure clean VALL-E speech decoded by TraceableSpeech (sign=0)
- watermarked/{id}_traceablespeech_wm.wav: Watermarked speech decoded by TraceableSpeech (sign)
- prompt/{id}_prompt.wav: Prompt speech reference
- metadata.jsonl: Complete metadata mapping
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

# Mock unavailable k2 modules for smooth execution
for mod in ["k2", "k2.version", "kaldialign"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import numpy as np
import torch
import torch.multiprocessing as mp
import torchaudio
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

from icefall.utils import AttributeDict
from valle.models import get_model
from valle.data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
)
from valle.data.collation import get_text_token_collater
from traceableSpeech.watermark import Random_watermark

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] [Rank %(process)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def load_dataset_records(dataset_name: str) -> List[Dict]:
    """Load canonical test records for LibriTTS (2,078) or SeedTTS (1,088)."""
    if dataset_name.lower() == "libritts":
        meta_file = SCRIPT_DIR / "synthesized_data/libriTTS/metadata.jsonl"
        with open(meta_file, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        
        items = []
        for r in records:
            items.append({
                "id": r["cut_id"],
                "text": r["text"],
                "prompt_text": r.get("prompt_text", ""),
                "prompt_wav": r["prompt_wav"],
            })
        return items

    elif dataset_name.lower() == "seedtts":
        meta_file = SCRIPT_DIR / "synthesized_data/seedTTS/metadata.jsonl"
        with open(meta_file, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        
        items = []
        for r in records:
            items.append({
                "id": r["utt_id"],
                "text": r["text"],
                "prompt_text": r.get("prompt_text", ""),
                "prompt_wav": r["prompt_wav"],
            })
        return items
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def worker_fn(
    rank: int,
    world_size: int,
    gpu_ids: List[int],
    dataset_name: str,
    output_base_dir: str,
    valle_ckpt_path: str,
    ts_ckpt_path: str,
    ts_cfg_path: str,
    top_k: int,
    temperature: float,
    seed: int,
):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    gpu_id = gpu_ids[rank % len(gpu_ids)]
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    out_dir = Path(output_base_dir) / dataset_name
    tokens_dir = out_dir / "tokens"
    clean_dir = out_dir / "clean_tts"
    wm_dir = out_dir / "watermarked"
    prompt_dir = out_dir / "prompt"

    if rank == 0:
        for d in [tokens_dir, clean_dir, wm_dir, prompt_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # 1. Load VALL-E Model
    valle_pkg = torch.load(valle_ckpt_path, map_location="cpu", weights_only=False)
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

    # 2. Load TraceableSpeech Audio Tokenizer
    audio_tokenizer = AudioTokenizer(
        watermark_backend="traceablespeech",
        ts_checkpoint=ts_ckpt_path,
        ts_config=ts_cfg_path,
        device=device,
    )

    all_items = load_dataset_records(dataset_name)
    # Shard items for this rank
    shard_items = all_items[rank::world_size]
    logging.info(f"[GPU {gpu_id} | Rank {rank}/{world_size}] Assigned {len(shard_items)}/{len(all_items)} samples for {dataset_name}.")

    shard_results = []
    
    for idx, item in enumerate(tqdm(shard_items, desc=f"Rank {rank} [{dataset_name}]", position=rank)):
        utt_id = item["id"]
        target_text = item["text"]
        prompt_text = item.get("prompt_text", "")
        prompt_wav_path = item["prompt_wav"]

        if not os.path.exists(prompt_wav_path):
            logging.warning(f"Rank {rank}: Prompt file {prompt_wav_path} not found. Skipping.")
            continue

        token_save_path = tokens_dir / f"{utt_id}_tokens.pt"
        clean_save_path = clean_dir / f"{utt_id}_clean_tts.wav"
        wm_save_path = wm_dir / f"{utt_id}_traceablespeech_wm.wav"
        prompt_save_path = prompt_dir / f"{utt_id}_prompt.wav"

        # Check if already synthesized
        if token_save_path.exists() and clean_save_path.exists() and wm_save_path.exists() and prompt_save_path.exists():
            shard_results.append({
                "utt_id": utt_id,
                "dataset": dataset_name,
                "text": target_text,
                "prompt_text": prompt_text,
                "token_path": str(token_save_path),
                "clean_tts_wav": str(clean_save_path),
                "wm_wav": str(wm_save_path),
                "prompt_wav": str(prompt_save_path),
                "clean_tts_relpath": f"clean_tts/{clean_save_path.name}",
                "wm_relpath": f"watermarked/{wm_save_path.name}",
                "prompt_relpath": f"prompt/{prompt_save_path.name}",
                "token_relpath": f"tokens/{token_save_path.name}",
            })
            continue

        try:
            # A. Tokenize Prompt Audio
            prompt_frames = tokenize_audio(audio_tokenizer, prompt_wav_path)
            audio_prompts = prompt_frames[0][0].transpose(2, 1).to(device)  # [1, 8, T_prompt]

            # B. Tokenize Combined Text
            full_text = f"{prompt_text} {target_text}".strip()
            text_tokens_tensor, text_tokens_lens = text_collater(
                [tokenize_text(text_tokenizer, text=full_text)]
            )
            enroll_x_lens = None
            if prompt_text.strip():
                _, enroll_x_lens = text_collater(
                    [tokenize_text(text_tokenizer, text=prompt_text.strip())]
                )

            # C. Synthesize Acoustic Tokens via VALL-E
            with torch.no_grad():
                encoded_frames = valle_model.inference(
                    text_tokens_tensor.to(device),
                    text_tokens_lens.to(device),
                    audio_prompts,
                    enroll_x_lens=enroll_x_lens,
                    top_k=top_k,
                    temperature=temperature,
                )  # [1, T_synth, 8] or [1, 8, T_synth]

            # Normalize token shape to [1, T, 8] for transpose(2, 1) -> [1, 8, T]
            if encoded_frames.dim() == 3:
                if encoded_frames.shape[1] == 8 and encoded_frames.shape[2] != 8:
                    encoded_frames = encoded_frames.transpose(2, 1)  # now [1, T, 8]

            # Ensure minimum sequence length >= 12 frames to satisfy conv upsampler kernel constraints
            if encoded_frames.shape[1] < 12:
                pad_amt = 12 - encoded_frames.shape[1]
                encoded_frames = torch.nn.functional.pad(encoded_frames, (0, 0, 0, pad_amt), mode="replicate")

            # Save Tokens
            torch.save(encoded_frames.transpose(2, 1).cpu(), str(token_save_path))

            # D. Decode Clean & Watermarked Audio via TraceableSpeech
            watermark_sign = Random_watermark(1).to(device)  # [1, 4]
            with torch.no_grad():
                clean_audio, wm_audio = audio_tokenizer.decode_pair(
                    [(encoded_frames.transpose(2, 1), None)],
                    watermark_sign=watermark_sign,
                )

            # Convert to standard 16kHz
            clean_16k = torchaudio.functional.resample(clean_audio.cpu(), 24000, 16000)
            wm_16k = torchaudio.functional.resample(wm_audio.cpu(), 24000, 16000)

            # Load Reference Prompt at 16kHz
            p_orig, p_sr = torchaudio.load(prompt_wav_path)
            if p_orig.shape[0] > 1:
                p_orig = p_orig.mean(dim=0, keepdim=True)
            prompt_16k = torchaudio.functional.resample(p_orig, p_sr, 16000)

            # Ensure 2D tensor [1, T]
            clean_16k_2d = clean_16k.reshape(1, -1)
            wm_16k_2d = wm_16k.reshape(1, -1)
            prompt_16k_2d = prompt_16k.reshape(1, -1)

            # Align lengths
            min_len = min(clean_16k_2d.shape[-1], wm_16k_2d.shape[-1])
            clean_16k_2d = clean_16k_2d[:, :min_len]
            wm_16k_2d = wm_16k_2d[:, :min_len]

            # Save Audio Files as 2D [1, T]
            torchaudio.save(str(clean_save_path), clean_16k_2d, 16000)
            torchaudio.save(str(wm_save_path), wm_16k_2d, 16000)
            torchaudio.save(str(prompt_save_path), prompt_16k_2d, 16000)

            dur_sec = min_len / 16000.0
            p_dur_sec = prompt_16k_2d.shape[-1] / 16000.0

            shard_results.append({
                "utt_id": utt_id,
                "dataset": dataset_name,
                "text": target_text,
                "prompt_text": prompt_text,
                "watermark_sign": watermark_sign.cpu().tolist()[0],
                "token_path": str(token_save_path),
                "clean_tts_wav": str(clean_save_path),
                "wm_wav": str(wm_save_path),
                "prompt_wav": str(prompt_save_path),
                "clean_tts_relpath": f"clean_tts/{clean_save_path.name}",
                "wm_relpath": f"watermarked/{wm_save_path.name}",
                "prompt_relpath": f"prompt/{prompt_save_path.name}",
                "token_relpath": f"tokens/{token_save_path.name}",
                "clean_duration_sec": round(dur_sec, 2),
                "prompt_duration_sec": round(p_dur_sec, 2),
            })
        except Exception as e:
            logging.error(f"Rank {rank}: Error synthesizing {utt_id}: {e}")

    # Write shard jsonl
    shard_meta_path = out_dir / f"metadata_shard_{rank:02d}_of_{world_size:02d}.jsonl"
    with open(shard_meta_path, "w", encoding="utf-8") as f:
        for item in shard_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logging.info(f"[Rank {rank}] Finished synthesizing {len(shard_results)} items for {dataset_name}.")


def merge_shards(dataset_name: str, output_base_dir: str, world_size: int):
    out_dir = Path(output_base_dir) / dataset_name
    all_records = []
    
    for rank in range(world_size):
        shard_path = out_dir / f"metadata_shard_{rank:02d}_of_{world_size:02d}.jsonl"
        if shard_path.exists():
            with open(shard_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_records.append(json.loads(line))

    # Sort deterministically
    all_records.sort(key=lambda x: x["utt_id"])

    unified_jsonl = out_dir / "metadata.jsonl"
    with open(unified_jsonl, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    unified_json = out_dir / "metadata.json"
    with open(unified_json, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset_name,
            "total_records": len(all_records),
            "records": all_records,
        }, f, indent=2, ensure_ascii=False)

    logging.info(f"Successfully merged {len(all_records)} records into {unified_jsonl}")


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Parallel Synthesis for TraceableSpeech + VALL-E")
    parser.add_argument("--dataset", type=str, default="both", choices=["libritts", "seedtts", "both"])
    parser.add_argument("--output-base-dir", type=str, default="exp/valle_traceablespeech_synthesis")
    parser.add_argument("--valle-checkpoint", type=str, default="exp/valle/epoch-40.pt")
    parser.add_argument("--ts-checkpoint", type=str, default="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000")
    parser.add_argument("--ts-config", type=str, default="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--top-k", type=int, default=-100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    world_size = len(gpu_ids)

    datasets = ["libritts", "seedtts"] if args.dataset == "both" else [args.dataset]

    logging.info("=" * 80)
    logging.info(f" Starting 8-GPU Parallel Synthesis for TraceableSpeech + VALL-E ")
    logging.info(f" Datasets:          {datasets}")
    logging.info(f" Output Base Dir:   {args.output_base_dir}")
    logging.info(f" VALL-E Checkpoint: {args.valle_checkpoint}")
    logging.info(f" GPUs ({world_size}):       {gpu_ids}")
    logging.info("=" * 80)

    for ds in datasets:
        logging.info(f"\n>>> Launching Multi-GPU Synthesis for Dataset: {ds} <<<")
        mp.spawn(
            worker_fn,
            args=(
                world_size,
                gpu_ids,
                ds,
                args.output_base_dir,
                args.valle_checkpoint,
                args.ts_checkpoint,
                args.ts_config,
                args.top_k,
                args.temperature,
                args.seed,
            ),
            nprocs=world_size,
            join=True,
        )
        merge_shards(ds, args.output_base_dir, world_size)

    logging.info("\n>>> All Multi-GPU Synthesis Completed Successfully! <<<")


if __name__ == "__main__":
    main()

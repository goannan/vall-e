#!/usr/bin/env python3
"""
Multi-GPU Parallel Synthesis Worker for TraceableSpeech Native VALL-E.
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import List, Dict
from unittest.mock import MagicMock

for mod in ["k2", "k2.version", "kaldialign"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import numpy as np
import torch
torch.set_num_threads(4)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel TraceableSpeech Synthesis Worker")
    parser.add_argument("--dataset", type=str, required=True, choices=["libritts", "seedtts"])
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--valle-checkpoint", type=str, default="/home/wu25/mrnas04home/projects/vall-e/egs/libritts/exp/valle/epoch-40.pt")
    parser.add_argument("--ts-checkpoint", type=str, default="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000")
    parser.add_argument("--ts-config", type=str, default="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json")
    parser.add_argument("--top-k", type=int, default=-100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--max-samples", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    tokens_dir = out_dir / "tokens"
    clean_dir = out_dir / "clean_tts"
    wm_dir = out_dir / "watermarked"
    prompt_dir = out_dir / "prompt"

    for d in [tokens_dir, clean_dir, wm_dir, prompt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logging.info(f"Rank {args.rank} loading VALL-E checkpoint from {args.valle_checkpoint} on {device}...")
    valle_ckpt = torch.load(args.valle_checkpoint, map_location="cpu", weights_only=False)
    valle_args = AttributeDict(valle_ckpt)
    valle_model = get_model(valle_args)
    valle_model.load_state_dict(valle_ckpt["model"], strict=True)
    valle_model.to(device)
    valle_model.eval()

    text_collater = get_text_token_collater(valle_args.text_tokens)
    text_tokenizer = TextTokenizer(backend="espeak")

    logging.info(f"Rank {args.rank} loading TraceableSpeech tokenizer on {device}...")
    ts_tok = AudioTokenizer(
        watermark_backend="traceablespeech",
        enable_ts=True,
        ts_checkpoint=args.ts_checkpoint,
        ts_config=args.ts_config,
        device=device,
    )

    all_records = load_dataset_records(args.dataset)
    if args.max_samples > 0:
        all_records = all_records[:args.max_samples]

    shard_records = all_records[args.rank :: args.world_size]
    logging.info(f"Rank {args.rank}/{args.world_size} on {device}: Assigned {len(shard_records)}/{len(all_records)} samples for {args.dataset}.")

    shard_results = []

    for item in tqdm(shard_records, desc=f"Rank {args.rank} [{args.dataset}]"):
        utt_id = item["id"]
        target_text = item["text"]
        prompt_text = item.get("prompt_text", "")
        prompt_wav_path = item["prompt_wav"]

        if not os.path.exists(prompt_wav_path):
            logging.warning(f"Missing prompt: {prompt_wav_path}")
            continue

        token_save_path = tokens_dir / f"{utt_id}_tokens.pt"
        clean_save_path = clean_dir / f"{utt_id}_clean_tts.wav"
        wm_save_path = wm_dir / f"{utt_id}_traceablespeech_wm.wav"
        prompt_save_path = prompt_dir / f"{utt_id}_prompt.wav"
        sign_save_path = out_dir / "tokens" / f"{utt_id}_sign.json"

        # Check existing & resume unless overwrite
        if not args.overwrite and token_save_path.exists() and clean_save_path.exists() and wm_save_path.exists() and prompt_save_path.exists() and sign_save_path.exists():
            if clean_save_path.stat().st_size > 1000 and wm_save_path.stat().st_size > 1000:
                with open(sign_save_path, "r", encoding="utf-8") as f:
                    sign_data = json.load(f)
                shard_results.append({
                    "utt_id": utt_id,
                    "dataset": args.dataset,
                    "text": target_text,
                    "prompt_text": prompt_text,
                    "watermark_sign": sign_data["watermark_sign"],
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
            # A. Encode Prompt Audio
            prompt_frames = tokenize_audio(ts_tok, prompt_wav_path)
            audio_prompts = prompt_frames[0][0].transpose(2, 1).to(device)

            # B. Text Tokenization with Prefix Conditioning
            full_text = f"{prompt_text} {target_text}".strip()
            text_tokens_tensor, text_tokens_lens = text_collater([tokenize_text(text_tokenizer, text=full_text)])
            enroll_x_lens = None
            if prompt_text.strip():
                _, enroll_x_lens = text_collater([tokenize_text(text_tokenizer, text=prompt_text.strip())])

            # C. VALL-E Inference
            with torch.no_grad():
                encoded_frames = valle_model.inference(
                    text_tokens_tensor.to(device),
                    text_tokens_lens.to(device),
                    audio_prompts,
                    enroll_x_lens=enroll_x_lens.to(device) if enroll_x_lens is not None else None,
                    top_k=args.top_k,
                    temperature=args.temperature,
                )

            if encoded_frames is None or encoded_frames.numel() == 0:
                continue

            # Normalize to [1, 8, T]
            if encoded_frames.shape[1] != 8 and encoded_frames.shape[2] == 8:
                codes_qbt = encoded_frames.transpose(2, 1)
            else:
                codes_qbt = encoded_frames

            if codes_qbt.shape[2] == 0:
                continue
            if codes_qbt.shape[2] < 12:
                pad_amt = 12 - codes_qbt.shape[2]
                codes_qbt = torch.nn.functional.pad(codes_qbt, (0, pad_amt), mode="replicate")

            # Save Tokens
            torch.save(codes_qbt.cpu(), str(token_save_path))

            # D. TraceableSpeech Watermark Generation & Decoding
            watermark_sign = Random_watermark(1).to(device)  # [1, 4]
            with torch.no_grad():
                clean_audio, wm_audio = ts_tok.decode_pair(
                    [(codes_qbt, None)],
                    watermark_sign=watermark_sign,
                )

            clean_16k_2d = torchaudio.functional.resample(clean_audio.cpu(), 24000, 16000).reshape(1, -1)
            wm_16k_2d = torchaudio.functional.resample(wm_audio.cpu(), 24000, 16000).reshape(1, -1)

            p_orig, p_sr = torchaudio.load(prompt_wav_path)
            if p_orig.shape[0] > 1:
                p_orig = p_orig.mean(dim=0, keepdim=True)
            prompt_16k_2d = torchaudio.functional.resample(p_orig, p_sr, 16000).reshape(1, -1)

            min_len = min(clean_16k_2d.shape[-1], wm_16k_2d.shape[-1])
            clean_16k_2d = clean_16k_2d[:, :min_len]
            wm_16k_2d = wm_16k_2d[:, :min_len]

            torchaudio.save(str(clean_save_path), clean_16k_2d, 16000)
            torchaudio.save(str(wm_save_path), wm_16k_2d, 16000)
            torchaudio.save(str(prompt_save_path), prompt_16k_2d, 16000)

            dur_sec = round(min_len / 16000.0, 2)
            p_dur_sec = round(prompt_16k_2d.shape[-1] / 16000.0, 2)
            sign_list = watermark_sign.cpu().tolist()[0]

            with open(sign_save_path, "w", encoding="utf-8") as f:
                json.dump({"utt_id": utt_id, "watermark_sign": sign_list}, f)

            shard_results.append({
                "utt_id": utt_id,
                "dataset": args.dataset,
                "text": target_text,
                "prompt_text": prompt_text,
                "watermark_sign": sign_list,
                "token_path": str(token_save_path),
                "clean_tts_wav": str(clean_save_path),
                "wm_wav": str(wm_save_path),
                "prompt_wav": str(prompt_save_path),
                "clean_tts_relpath": f"clean_tts/{clean_save_path.name}",
                "wm_relpath": f"watermarked/{wm_save_path.name}",
                "prompt_relpath": f"prompt/{prompt_save_path.name}",
                "token_relpath": f"tokens/{token_save_path.name}",
                "clean_duration_sec": dur_sec,
                "prompt_duration_sec": p_dur_sec,
            })
        except Exception as e:
            logging.error(f"Rank {args.rank}: Error on {utt_id}: {e}")

    shard_meta_path = out_dir / f"metadata_shard_{args.rank:02d}_of_{args.world_size:02d}.jsonl"
    with open(shard_meta_path, "w", encoding="utf-8") as f:
        for item in shard_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logging.info(f"Rank {args.rank}: Finished {len(shard_results)}/{len(shard_records)} for {args.dataset}.")


if __name__ == "__main__":
    main()

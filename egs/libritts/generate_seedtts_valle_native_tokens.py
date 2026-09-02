#!/usr/bin/env python3
"""
Generate VALL-E native 8-layer acoustic tokens for SeedTTS dataset (1,088 samples).
Input:  data/seed_tts_eval/en/meta.lst
Output: data/tokenized_voicemark/cuts_seedtts_valle_native.jsonl.gz
        data/tokenized_voicemark/seedtts_valle_native.h5
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

for mod in ["k2", "k2.version", "kaldialign"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import h5py
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent.parent

def find_neumark_root() -> Path:
    candidates = [
        os.environ.get("NEUMARK_ROOT"),
        SCRIPT_DIR.parent.parent.parent / "NeuMark",
        PROJECT_DIR.parent / "NeuMark",
        Path.cwd() / "NeuMark",
        Path.cwd().parent / "NeuMark",
        Path.home() / "projects" / "NeuMark",
    ]
    for c in candidates:
        if c:
            p = Path(c).resolve()
            if p.is_dir():
                return p
    return (PROJECT_DIR.parent / "NeuMark").resolve()

NEUMARK_ROOT = find_neumark_root()

for p in [str(PROJECT_DIR), str(SCRIPT_DIR), str(NEUMARK_ROOT), str(NEUMARK_ROOT / "train")]:
    if p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)

from icefall.utils import AttributeDict
from lhotse import CutSet, MonoCut, SupervisionSegment
from lhotse.features import Features
from STmodels.model import SpeechTokenizer
from valle.data.tokenizer import TextTokenizer, tokenize_text
from valle.data.collation import get_text_token_collater
from valle.models import get_model

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate VALL-E native tokens for SeedTTS dataset")
    parser.add_argument(
        "--valle-checkpoint",
        type=str,
        default="valle_checkpoints/valle_voicemark_epoch40.pt",
        help="Path to trained VALL-E checkpoint",
    )
    parser.add_argument(
        "--meta-lst",
        type=str,
        default="data/seed_tts_eval/en/meta.lst",
        help="Path to SeedTTS meta.lst",
    )
    parser.add_argument(
        "--prompt-wav-dir",
        type=str,
        default="data/seed_tts_eval/en",
        help="Base directory for SeedTTS prompt wavs",
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        default="data/tokenized_voicemark/cuts_seedtts_valle_native.jsonl.gz",
    )
    parser.add_argument(
        "--output-h5",
        type=str,
        default="data/tokenized_voicemark/seedtts_valle_native.h5",
    )
    parser.add_argument(
        "--text-tokens",
        type=str,
        default="data/tokenized/unique_text_tokens.k2symbols",
    )
    parser.add_argument(
        "--speechtokenizer-config",
        type=str,
        default=str(NEUMARK_ROOT / "STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json"),
    )
    parser.add_argument(
        "--speechtokenizer-checkpoint",
        type=str,
        default=str(NEUMARK_ROOT / "STmodels/pretrained_model/SpeechTokenizer.pt"),
    )
    parser.add_argument("--top-k", type=int, default=-100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info("=" * 80)
    logging.info(f" Starting VALL-E Native Token Generation for SeedTTS")
    logging.info(f" Meta LST:         {args.meta_lst}")
    logging.info(f" VALL-E Checkpoint:{args.valle_checkpoint}")
    logging.info(f" Device:           {device}")
    logging.info("=" * 80)

    # 1. Load VALL-E Model
    ckpt_path = Path(args.valle_checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = SCRIPT_DIR / ckpt_path
    logging.info(f"Loading VALL-E model from {ckpt_path}...")
    ckpt_data = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model_args = AttributeDict(ckpt_data)
    valle_model = get_model(model_args)
    valle_model.load_state_dict(ckpt_data["model"], strict=True)
    valle_model.to(device)
    valle_model.eval()

    # 2. Text Collater & Text Tokenizer
    text_tokens_file = args.text_tokens if (args.text_tokens and os.path.exists(args.text_tokens)) else model_args.text_tokens
    if not os.path.isabs(text_tokens_file):
        text_tokens_file = str(SCRIPT_DIR / text_tokens_file)
    text_collater = get_text_token_collater(text_tokens_file)
    text_tokenizer = TextTokenizer(backend="espeak")

    # 3. SpeechTokenizer
    logging.info(f"Loading SpeechTokenizer from {args.speechtokenizer_checkpoint}...")
    speech_tokenizer = SpeechTokenizer.load_from_checkpoint(
        args.speechtokenizer_config,
        args.speechtokenizer_checkpoint,
    ).to(device)
    speech_tokenizer.eval()

    # 4. Load SeedTTS meta.lst records
    records = []
    with open(args.meta_lst, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 4:
                continue
            utt_id, prompt_text, prompt_wav_rel, target_text = parts
            records.append({
                "utt_id": utt_id,
                "prompt_text": prompt_text.strip(),
                "prompt_wav_rel": prompt_wav_rel.strip(),
                "target_text": target_text.strip(),
            })

    total_records = len(records)
    logging.info(f"Loaded {total_records} records from {args.meta_lst}")

    # Shard for distributed multi-GPU generation
    if args.world_size > 1:
        records = records[args.rank :: args.world_size]
        logging.info(f"Rank {args.rank} assigned {len(records)} records.")

    out_h5_path = Path(args.output_h5)
    if not out_h5_path.is_absolute():
        out_h5_path = SCRIPT_DIR / out_h5_path
    if args.world_size > 1:
        out_h5_path = out_h5_path.with_name(f"{out_h5_path.stem}_rank{args.rank}{out_h5_path.suffix}")
    out_h5_path.parent.mkdir(parents=True, exist_ok=True)

    out_manifest_path = Path(args.output_manifest)
    if not out_manifest_path.is_absolute():
        out_manifest_path = SCRIPT_DIR / out_manifest_path
    if args.world_size > 1:
        stem = out_manifest_path.name
        if stem.endswith(".jsonl.gz"):
            base = stem[:-9]
        elif stem.endswith(".jsonl"):
            base = stem[:-6]
        else:
            base = out_manifest_path.stem
        out_manifest_path = out_manifest_path.with_name(f"{base}_rank{args.rank}.jsonl.gz")

    # 5. Open H5 file with Resume Support
    try:
        h5_file = h5py.File(str(out_h5_path), "a")
        existing_keys = set(h5_file.keys())
        logging.info(f"Found {len(existing_keys)} existing entries in {out_h5_path} (resuming)...")
    except Exception as ex:
        logging.warning(f"Recreating H5 file {out_h5_path} due to: {ex}")
        if out_h5_path.exists():
            out_h5_path.unlink()
        h5_file = h5py.File(str(out_h5_path), "w")
        existing_keys = set()

    generated_cuts = []
    base_prompt_dir = Path(args.prompt_wav_dir)
    if not base_prompt_dir.is_absolute():
        base_prompt_dir = SCRIPT_DIR / base_prompt_dir

    # 6. Inference Loop
    with torch.no_grad():
        for rec in tqdm(records, desc=f"SeedTTS VALL-E Native Gen [Rank {args.rank}]"):
            cut_key = f"{rec['utt_id']}_valle_native"

            if cut_key in existing_keys:
                gen_codes_np = h5_file[cut_key][:]
            else:
                p_wav_path = base_prompt_dir / rec["prompt_wav_rel"]
                if not p_wav_path.exists():
                    logging.warning(f"Missing prompt audio: {p_wav_path}")
                    continue

                try:
                    p_wav, p_sr = torchaudio.load(str(p_wav_path))
                    if p_sr != 16000:
                        p_wav = torchaudio.functional.resample(p_wav, p_sr, 16000)
                    if p_wav.shape[0] > 1:
                        p_wav = p_wav.mean(dim=0, keepdim=True)
                    p_wav = p_wav.unsqueeze(0).to(device)  # [1, 1, T]

                    # Tokenize prompt wav using SpeechTokenizer -> [8, 1, T_frames] -> [1, T_frames, 8]
                    prompt_codes = speech_tokenizer.encode(p_wav)  # [8, 1, T]
                    audio_prompt_tokens = prompt_codes.squeeze(1).transpose(1, 0).unsqueeze(0).to(device)  # [1, T, 8]
                except Exception as ex:
                    logging.warning(f"Error tokenizing prompt audio {p_wav_path}: {ex}")
                    continue

                if audio_prompt_tokens.shape[1] < 10:
                    continue

                full_text = f"{rec['prompt_text']} {rec['target_text']}".strip()
                text_tokens_idx, text_tokens_lens = text_collater(
                    [tokenize_text(text_tokenizer, text=full_text)]
                )
                text_tokens_idx = text_tokens_idx.to(device)
                text_tokens_lens = text_tokens_lens.to(device)

                _, enroll_x_lens = text_collater(
                    [tokenize_text(text_tokenizer, text=rec['prompt_text'])]
                )
                enroll_x_lens = enroll_x_lens.to(device)

                try:
                    # Run exact FP32 inference
                    gen_tokens = valle_model.inference(
                        text_tokens_idx,
                        text_tokens_lens,
                        audio_prompt_tokens,
                        enroll_x_lens=enroll_x_lens,
                        top_k=args.top_k,
                        temperature=args.temperature,
                    )
                    target_tokens = gen_tokens[0].cpu().numpy().astype(np.int16)
                    if target_tokens.shape[0] < 5:
                        continue

                    h5_file.create_dataset(cut_key, data=target_tokens, compression="gzip")
                    h5_file.flush()
                    gen_codes_np = target_tokens
                except Exception as ex:
                    logging.warning(f"Error generating {rec['utt_id']}: {ex}")
                    continue

            frame_shift = 0.02
            gen_duration = float(gen_codes_np.shape[0] * frame_shift)
            storage_p = str(out_h5_path)

            feat = Features(
                type="valle_native",
                num_frames=gen_codes_np.shape[0],
                num_features=8,
                frame_shift=frame_shift,
                sampling_rate=16000,
                start=0.0,
                duration=gen_duration,
                storage_type="numpy_hdf5",
                storage_path=storage_p,
                storage_key=cut_key,
            )

            supervision = SupervisionSegment(
                id=cut_key,
                recording_id=cut_key,
                start=0.0,
                duration=gen_duration,
                channel=0,
                text=rec["target_text"],
                language="en",
                speaker=rec["utt_id"].split("-")[0],
            )

            new_cut = MonoCut(
                id=cut_key,
                start=0.0,
                duration=gen_duration,
                channel=0,
                features=feat,
                supervisions=[supervision],
                custom={
                    "target_utt_id": rec["utt_id"],
                    "target_text": rec["target_text"],
                    "prompt_wav_rel": rec["prompt_wav_rel"],
                    "prompt_text": rec["prompt_text"],
                }
            )
            generated_cuts.append(new_cut)

    h5_file.close()
    logging.info(f"Saving output manifest to {out_manifest_path} ({len(generated_cuts)} cuts)...")
    CutSet.from_cuts(generated_cuts).to_file(out_manifest_path)
    logging.info(f"Done! Successfully generated and saved {len(generated_cuts)} cuts to {out_manifest_path}")

if __name__ == "__main__":
    main()

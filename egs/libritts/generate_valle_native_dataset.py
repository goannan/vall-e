#!/usr/bin/env python3
# Copyright (c) 2026
# Pre-generate VALL-E TTS-Native 8-layer acoustic tokens using Speaker-Paired Full Sentences (3-10s)

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import h5py
import numpy as np
import torch
from tqdm import tqdm

# Mock unneeded C++ dependencies
for mod in ["k2", "k2.version", "kaldialign", "pypinyin", "pypinyin.contrib", "pypinyin.contrib.tone_convert",
            "phonemizer", "phonemizer.backend", "phonemizer.backend.espeak", "phonemizer.backend.espeak.language_switch",
            "phonemizer.backend.espeak.words_mismatch", "phonemizer.punctuation", "phonemizer.separator"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent.parent
NEUMARK_ROOT = Path("/home/wu25/mrnas04home/projects/NeuMark").resolve()

for p in [str(PROJECT_DIR), str(SCRIPT_DIR), str(NEUMARK_ROOT), str(NEUMARK_ROOT / "train")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from icefall.utils import AttributeDict
from lhotse import CutSet, MonoCut, SupervisionSegment, load_manifest_lazy
from lhotse.features import Features
from valle.data import AudioTokenizer
from valle.data.collation import get_text_token_collater
from valle.models import get_model

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-generate VALL-E TTS-Native Token Dataset (3-10s Full Sentences)")
    parser.add_argument("--valle-checkpoint", type=str, default="exp/valle_voicemark/epoch-40.pt")
    parser.add_argument("--input-manifest", type=str, default="data/tokenized_voicemark/cuts_train.jsonl.gz")
    parser.add_argument("--output-manifest", type=str, default="data/tokenized_voicemark/cuts_train_valle_native.jsonl.gz")
    parser.add_argument("--output-h5", type=str, default="data/tokenized_voicemark/libritts_valle_native_train.h5")
    parser.add_argument("--text-tokens", type=str, default="data/tokenized_voicemark/unique_text_tokens.k2symbols")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for parallel generation (default 4)")
    parser.add_argument("--max-samples", type=int, default=-1, help="Max pairs to generate (-1 for all)")
    parser.add_argument("--min-duration", type=float, default=3.0, help="Min cut duration in seconds")
    parser.add_argument("--max-duration", type=float, default=10.0, help="Max cut duration in seconds")
    parser.add_argument("--prompt-max-frames", type=int, default=150, help="Max prompt frames (150 = ~2-3s)")
    parser.add_argument("--top-k", type=int, default=-100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def build_speaker_pairs(cuts, min_dur: float, max_dur: float):
    """Group cuts by speaker and pair each target cut with a different prompt cut from the same speaker."""
    spk_map = defaultdict(list)
    for c in cuts:
        if min_dur <= c.duration <= max_dur and c.supervisions:
            spk_id = c.supervisions[0].speaker
            spk_map[spk_id].append(c)

    pairs = []
    for spk_id, spk_cuts in spk_map.items():
        n = len(spk_cuts)
        if n < 2:
            continue
        for i, target_cut in enumerate(spk_cuts):
            prompt_cut = spk_cuts[(i + 1) % n]  # pick next utterance of same speaker as prompt
            pairs.append((prompt_cut, target_cut))
    return pairs


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Enable TF32 and cuDNN benchmark for maximum Tensor Core utilization
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    logging.info(f"Using device: {device}, Rank: {args.rank}/{args.world_size}")

    # 1. Load VALL-E Model
    logging.info(f"Loading VALL-E model from {args.valle_checkpoint}...")
    ckpt_data = torch.load(args.valle_checkpoint, map_location="cpu", weights_only=False)
    model_args = AttributeDict(ckpt_data)
    valle_model = get_model(model_args)
    valle_model.load_state_dict(ckpt_data["model"], strict=True)
    valle_model.to(device)
    valle_model.eval()

    # 2. Text Collater
    text_tokens_file = args.text_tokens if os.path.exists(args.text_tokens) else model_args.text_tokens
    text_collater = get_text_token_collater(text_tokens_file)

    # 3. Load Cuts & Build Pairs
    logging.info(f"Loading input cuts from {args.input_manifest} (Filtering: {args.min_duration}s - {args.max_duration}s)...")
    raw_cuts = load_manifest_lazy(args.input_manifest)
    pairs = build_speaker_pairs(raw_cuts, args.min_duration, args.max_duration)
    logging.info(f"Total valid same-speaker pairs found: {len(pairs)}")

    # Shard for distributed generation
    if args.world_size > 1:
        pairs = pairs[args.rank :: args.world_size]
        logging.info(f"Rank {args.rank} assigned {len(pairs)} pairs.")

    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]

    out_h5_path = Path(args.output_h5)
    if args.world_size > 1:
        out_h5_path = out_h5_path.with_name(f"{out_h5_path.stem}_rank{args.rank}{out_h5_path.suffix}")
    out_h5_path.parent.mkdir(parents=True, exist_ok=True)

    out_manifest_path = Path(args.output_manifest)
    if args.world_size > 1:
        out_manifest_path = out_manifest_path.with_name(f"{out_manifest_path.stem}_rank{args.rank}.jsonl.gz")

    # 4. Open H5 file and Resume check (with corruption recovery)
    try:
        h5_file = h5py.File(str(out_h5_path), "a")
        existing_keys = set(h5_file.keys())
        logging.info(f"Found {len(existing_keys)} already generated entries in {out_h5_path} (resuming)...")
    except Exception as ex:
        logging.warning(f"Warning: H5 file {out_h5_path} was corrupted by job cancellation ({ex}). Recreating clean file...")
        if out_h5_path.exists():
            out_h5_path.unlink()
        h5_file = h5py.File(str(out_h5_path), "w")
        existing_keys = set()

    generated_cuts = []

    # 5. Generation Loop (supports Batch Size >= 1 with 100% single-sample fallback)
    with torch.inference_mode():
        # Process in chunks of batch_size
        batch_size = max(1, args.batch_size)
        pair_batches = [pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size)]

        for batch in tqdm(pair_batches, desc=f"VALL-E Gen [Rank {args.rank}]"):
            pending_items = []
            for prompt_cut, target_cut in batch:
                cut_key = f"{target_cut.id}_paired_{prompt_cut.id}"
                if cut_key in existing_keys:
                    gen_codes_np = h5_file[cut_key][:]
                    # Construct Lhotse Cut
                    gen_duration = float(gen_codes_np.shape[0] * 0.013333333333333334)
                    feat = Features(
                        type="valle_native", num_frames=gen_codes_np.shape[0], num_features=8,
                        frame_shift=0.013333333333333334, sampling_rate=16000, start=0.0, duration=gen_duration,
                        storage_type="numpy_hdf5", storage_path=str(Path(args.output_h5).name if args.world_size == 1 else Path(args.output_h5).with_name(f"{Path(args.output_h5).stem}_rank{args.rank}{Path(args.output_h5).suffix}")),
                        storage_key=cut_key,
                    )
                    new_supervision = target_cut.supervisions[0] if target_cut.supervisions else None
                    generated_cuts.append(MonoCut(
                        id=cut_key, start=0.0, duration=gen_duration, channel=0, features=feat,
                        recording=target_cut.recording if target_cut.has_recording else None,
                        supervisions=[new_supervision] if new_supervision else [],
                        custom={"prompt_cut_id": prompt_cut.id, "target_cut_id": target_cut.id, "speaker": prompt_cut.supervisions[0].speaker if prompt_cut.supervisions else "unknown"}
                    ))
                else:
                    pending_items.append((prompt_cut, target_cut, cut_key))

            if not pending_items:
                continue

            if batch_size == 1 or len(pending_items) == 1:
                # === [ORIGINAL SINGLE SAMPLE PATH] ===
                for prompt_cut, target_cut, cut_key in pending_items:
                    try:
                        p_codes_np = prompt_cut.load_features()
                        if p_codes_np is None or p_codes_np.shape[0] < 20: continue
                        prompt_frames = min(args.prompt_max_frames, p_codes_np.shape[0])
                        audio_prompt_tokens = torch.from_numpy(p_codes_np[:prompt_frames]).long().unsqueeze(0).to(device)
                        prompt_len = audio_prompt_tokens.shape[1]

                        p_phonemes = prompt_cut.supervisions[0].custom["tokens"]["text"]
                        p_text_len = max(5, int(len(p_phonemes) * (prompt_frames / p_codes_np.shape[0])))
                        p_phonemes_slice = p_phonemes[:p_text_len]
                        t_phonemes = target_cut.supervisions[0].custom["tokens"]["text"]

                        full_phonemes = p_phonemes_slice + ["_"] + t_phonemes
                        text_tokens_idx, text_tokens_lens = text_collater([full_phonemes])
                        text_tokens_idx = text_tokens_idx.to(device)
                        text_tokens_lens = text_tokens_lens.to(device)
                        enroll_x_lens = torch.tensor([len(p_phonemes_slice)], device=device)

                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                            gen_tokens = valle_model.inference(
                                text_tokens_idx, text_tokens_lens, audio_prompt_tokens,
                                enroll_x_lens=enroll_x_lens, top_k=args.top_k, temperature=args.temperature,
                            )
                        target_tokens = gen_tokens[0, prompt_len:, :].cpu().numpy().astype(np.int16)
                        if target_tokens.shape[0] < 10: continue

                        h5_file.create_dataset(cut_key, data=target_tokens, compression="gzip")
                        h5_file.flush()
                        gen_codes_np = target_tokens

                        gen_duration = float(gen_codes_np.shape[0] * 0.013333333333333334)
                        feat = Features(
                            type="valle_native", num_frames=gen_codes_np.shape[0], num_features=8,
                            frame_shift=0.013333333333333334, sampling_rate=16000, start=0.0, duration=gen_duration,
                            storage_type="numpy_hdf5", storage_path=str(Path(args.output_h5).name if args.world_size == 1 else Path(args.output_h5).with_name(f"{Path(args.output_h5).stem}_rank{args.rank}{Path(args.output_h5).suffix}")),
                            storage_key=cut_key,
                        )
                        new_supervision = target_cut.supervisions[0] if target_cut.supervisions else None
                        generated_cuts.append(MonoCut(
                            id=cut_key, start=0.0, duration=gen_duration, channel=0, features=feat,
                            recording=target_cut.recording if target_cut.has_recording else None,
                            supervisions=[new_supervision] if new_supervision else [],
                            custom={"prompt_cut_id": prompt_cut.id, "target_cut_id": target_cut.id, "speaker": prompt_cut.supervisions[0].speaker if prompt_cut.supervisions else "unknown"}
                        ))
                    except Exception as ex:
                        logging.warning(f"Error generating target {target_cut.id}: {ex}")
            else:
                # === [PARALLEL BATCH INFERENCE PATH] ===
                try:
                    p_codes_list = []
                    phonemes_list = []
                    enroll_lens = []
                    valid_items = []

                    for prompt_cut, target_cut, cut_key in pending_items:
                        p_codes_np = prompt_cut.load_features()
                        if p_codes_np is None or p_codes_np.shape[0] < 20: continue
                        prompt_frames = min(args.prompt_max_frames, p_codes_np.shape[0])
                        p_codes_slice = p_codes_np[:prompt_frames]

                        p_phonemes = prompt_cut.supervisions[0].custom["tokens"]["text"]
                        p_text_len = max(5, int(len(p_phonemes) * (prompt_frames / p_codes_np.shape[0])))
                        p_phonemes_slice = p_phonemes[:p_text_len]
                        t_phonemes = target_cut.supervisions[0].custom["tokens"]["text"]

                        full_phonemes = p_phonemes_slice + ["_"] + t_phonemes
                        p_codes_list.append(p_codes_slice)
                        phonemes_list.append(full_phonemes)
                        enroll_lens.append(len(p_phonemes_slice))
                        valid_items.append((prompt_cut, target_cut, cut_key))

                    if not valid_items: continue

                    # Pad prompt audio codes to max_p_len
                    max_p_len = max(c.shape[0] for c in p_codes_list)
                    padded_audio_prompts = np.zeros((len(valid_items), max_p_len, 8), dtype=np.int64)
                    for idx, c in enumerate(p_codes_list):
                        padded_audio_prompts[idx, :c.shape[0], :] = c

                    audio_prompt_tokens = torch.from_numpy(padded_audio_prompts).to(device)
                    text_tokens_idx, text_tokens_lens = text_collater(phonemes_list)
                    text_tokens_idx = text_tokens_idx.to(device)
                    text_tokens_lens = text_tokens_lens.to(device)
                    enroll_x_lens = torch.tensor(enroll_lens, device=device)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                        batch_results = valle_model.inference_batch(
                            text_tokens_idx, text_tokens_lens, audio_prompt_tokens,
                            enroll_x_lens=enroll_x_lens, top_k=args.top_k, temperature=args.temperature,
                        )

                    for idx, (prompt_cut, target_cut, cut_key) in enumerate(valid_items):
                        target_tokens = batch_results[idx].cpu().numpy().astype(np.int16)
                        if target_tokens.shape[0] < 10: continue

                        h5_file.create_dataset(cut_key, data=target_tokens, compression="gzip")
                        h5_file.flush()
                        gen_codes_np = target_tokens

                        gen_duration = float(gen_codes_np.shape[0] * 0.013333333333333334)
                        feat = Features(
                            type="valle_native", num_frames=gen_codes_np.shape[0], num_features=8,
                            frame_shift=0.013333333333333334, sampling_rate=16000, start=0.0, duration=gen_duration,
                            storage_type="numpy_hdf5", storage_path=str(Path(args.output_h5).name if args.world_size == 1 else Path(args.output_h5).with_name(f"{Path(args.output_h5).stem}_rank{args.rank}{Path(args.output_h5).suffix}")),
                            storage_key=cut_key,
                        )
                        new_supervision = target_cut.supervisions[0] if target_cut.supervisions else None
                        generated_cuts.append(MonoCut(
                            id=cut_key, start=0.0, duration=gen_duration, channel=0, features=feat,
                            recording=target_cut.recording if target_cut.has_recording else None,
                            supervisions=[new_supervision] if new_supervision else [],
                            custom={"prompt_cut_id": prompt_cut.id, "target_cut_id": target_cut.id, "speaker": prompt_cut.supervisions[0].speaker if prompt_cut.supervisions else "unknown"}
                        ))
                except Exception as ex:
                    logging.warning(f"Batch generation exception: {ex}")

    h5_file.close()
    logging.info(f"Saving output manifest to {out_manifest_path} ({len(generated_cuts)} cuts)...")
    CutSet.from_cuts(generated_cuts).to_file(out_manifest_path)
    logging.info(f"Done! Saved {len(generated_cuts)} cuts to {out_manifest_path}")


if __name__ == "__main__":
    main()

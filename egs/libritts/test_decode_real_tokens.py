#!/usr/bin/env python3
# Copyright (c) 2026
# Decode REAL SpeechTokenizer Tokens from cuts_dev.jsonl.gz to test Codec Quality

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from lhotse import CutSet

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
NEUMARK_ROOT = Path("/home/wu25/mrnas04home/projects/NeuMark").resolve()

for p in [str(PROJECT_DIR), str(SCRIPT_DIR), str(NEUMARK_ROOT), str(NEUMARK_ROOT / "train")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from STmodels.model import SpeechTokenizer

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Decode REAL Speech Tokens from Lhotse Manifest")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/tokenized_voicemark/cuts_dev.jsonl.gz",
        help="Input Lhotse manifest containing real speech tokens",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp/real_token_decoded_samples",
        help="Output directory to save decoded audio files",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6,
        help="Number of samples to decode",
    )
    parser.add_argument(
        "--st-config",
        type=str,
        default=str(NEUMARK_ROOT / "STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json"),
        help="SpeechTokenizer config JSON path",
    )
    parser.add_argument(
        "--st-checkpoint",
        type=str,
        default=str(NEUMARK_ROOT / "STmodels/pretrained_model/SpeechTokenizer.pt"),
        help="SpeechTokenizer checkpoint (.pt) path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 80)
    logging.info(" Decoding REAL Ground-Truth SpeechTokenizer Tokens to Waveforms ")
    logging.info(f" Manifest:       {args.manifest}")
    logging.info(f" Output Dir:     {out_dir}")
    logging.info(f" ST Checkpoint:  {args.st_checkpoint}")
    logging.info(f" Device:         {device}")
    logging.info("=" * 80)

    # 1. Load SpeechTokenizer model
    logging.info("[1/3] Loading SpeechTokenizer codec model...")
    with open(args.st_config, "r") as f:
        st_cfg = json.load(f)
    st_model = SpeechTokenizer(st_cfg)
    st_state = torch.load(args.st_checkpoint, map_location="cpu")
    st_model.load_state_dict(st_state)
    st_model.to(device).eval()

    # 2. Load Real Cuts
    logging.info(f"[2/3] Loading real cuts from {args.manifest}...")
    cuts = CutSet.from_file(args.manifest)
    logging.info(f"Total available cuts: {len(cuts)}")

    # 3. Decode Samples
    logging.info(f"[3/3] Decoding first {args.num_samples} real token samples...")
    records = []

    for idx, cut in enumerate(cuts):
        if idx >= args.num_samples:
            break

        cut_id = cut.id
        speaker = cut.supervisions[0].speaker if cut.supervisions else "unknown"
        text = cut.supervisions[0].text if cut.supervisions else ""

        # Load real 8-layer tokens [T, 8]
        real_tokens = cut.load_features()  # np.ndarray shape (T, 8)
        if real_tokens is None:
            logging.warning(f"Skipping {cut_id}: No features found")
            continue

        tokens_tensor = torch.from_numpy(real_tokens).long()  # (T, 8)
        # Permute to (8, 1, T) for SpeechTokenizer.decode
        codes_tensor = tokens_tensor.permute(1, 0).unsqueeze(1).to(device)

        with torch.no_grad():
            decoded_wav = st_model.decode(codes_tensor)  # (1, 1, T_samples)
            decoded_wav = decoded_wav.squeeze(0).cpu()   # (1, T_samples)

        # File names
        f_decoded = out_dir / f"sample_{idx:03d}_real_token_decoded_spk{speaker}_{cut_id}.wav"
        f_raw_gt = out_dir / f"sample_{idx:03d}_original_raw_audio_spk{speaker}_{cut_id}.wav"

        # Save decoded wav (16kHz)
        torchaudio.save(str(f_decoded), decoded_wav, 16000)

        # Save original raw ground truth wav if recording source exists
        gt_dur = 0.0
        if cut.recording is not None:
            raw_samples = cut.load_audio()  # (channels, time)
            raw_sr = cut.sampling_rate
            if raw_sr != 16000:
                raw_samples = torchaudio.functional.resample(torch.from_numpy(raw_samples).float(), raw_sr, 16000)
            else:
                raw_samples = torch.from_numpy(raw_samples).float()
            if raw_samples.ndim == 1:
                raw_samples = raw_samples.unsqueeze(0)
            torchaudio.save(str(f_raw_gt), raw_samples, 16000)
            gt_dur = raw_samples.shape[-1] / 16000.0

        dec_dur = decoded_wav.shape[-1] / 16000.0
        logging.info(
            f"[{idx+1}/{args.num_samples}] Cut {cut_id} (Spk: {speaker}): "
            f"Frames={tokens_tensor.shape[0]}, DecDur={dec_dur:.2f}s, RawDur={gt_dur:.2f}s | "
            f"Text: \"{text[:50]}...\""
        )

        records.append({
            "sample_index": idx,
            "cut_id": cut_id,
            "speaker": speaker,
            "text": text,
            "token_shape": list(tokens_tensor.shape),
            "decoded_file": str(f_decoded.name),
            "decoded_duration_sec": dec_dur,
            "raw_gt_file": str(f_raw_gt.name),
            "raw_gt_duration_sec": gt_dur,
        })

    # Summary
    summary_path = out_dir / "real_tokens_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    summary_txt = out_dir / "real_tokens_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(" REAL Speech Token Decoding Summary \n")
        f.write(f" Manifest: {args.manifest}\n")
        f.write(f" Output Directory: {out_dir}\n")
        f.write("=" * 80 + "\n\n")
        for r in records:
            f.write(f"--- Sample {r['sample_index']:03d} (Cut ID: {r['cut_id']}) ---\n")
            f.write(f"  Speaker:      {r['speaker']}\n")
            f.write(f"  Text:         {r['text']}\n")
            f.write(f"  Token Shape:  {r['token_shape']}\n")
            f.write(f"  Decoded Wav:  {r['decoded_file']} ({r['decoded_duration_sec']:.2f}s)\n")
            f.write(f"  Raw GT Wav:   {r['raw_gt_file']} ({r['raw_gt_duration_sec']:.2f}s)\n\n")

    logging.info("=" * 80)
    logging.info(f" Successfully decoded {len(records)} REAL token samples!")
    logging.info(f" Saved to: {out_dir}")
    logging.info(f" Summary:  {summary_txt}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()

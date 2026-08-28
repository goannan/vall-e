#!/usr/bin/env python3
# Copyright (c) 2026
# Test & Decode pre-generated VALL-E native acoustic tokens into audio waveforms

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torchaudio
from lhotse import CutSet

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]

def find_neumark_root() -> Path:
    candidates = [
        os.environ.get("NEUMARK_ROOT"),
        SCRIPT_DIR.parent.parent.parent / "NeuMark",
        PROJECT_DIR.parent / "NeuMark",
        Path("/home/wu25/mrnas04home/projects/NeuMark"),
        Path.cwd() / "NeuMark",
    ]
    for c in candidates:
        if c:
            p = Path(c).resolve()
            if p.is_dir():
                return p
    return (PROJECT_DIR.parent / "NeuMark").resolve()

NEUMARK_ROOT = find_neumark_root()
for p in [str(PROJECT_DIR), str(SCRIPT_DIR), str(NEUMARK_ROOT), str(NEUMARK_ROOT / "train")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from STmodels.model import SpeechTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decode and verify pre-generated VALL-E native acoustic tokens into audio wavs"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/tokenized_voicemark/cuts_dev_valle_native.jsonl.gz",
        help="Path to pre-generated cuts manifest (jsonl.gz)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp/valle_native_test_samples/dev",
        help="Directory to save reconstructed wav files and summary report",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to decode (default: 5, set -1 for all)",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of cut indices to test (e.g. --indices 0 1 2 5 10)",
    )
    parser.add_argument(
        "--st-config",
        type=str,
        default=str(NEUMARK_ROOT / "STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json"),
        help="Path to SpeechTokenizer config JSON",
    )
    parser.add_argument(
        "--st-checkpoint",
        type=str,
        default=str(NEUMARK_ROOT / "STmodels/pretrained_model/SpeechTokenizer.pt"),
        help="Path to SpeechTokenizer model weights (.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Computation device (e.g. cuda:0 or cpu)",
    )
    parser.add_argument(
        "--save-gt",
        action="store_true",
        default=True,
        help="Save ground-truth target audio alongside synthesized audio for side-by-side comparison",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        logging.error(f"Manifest file not found: {manifest_path}")
        sys.exit(1)

    logging.info("=" * 65)
    logging.info(" VALL-E Native Token Synthesis & Quality Verification ")
    logging.info(f" Manifest:       {manifest_path}")
    logging.info(f" Output Dir:     {out_dir}")
    logging.info(f" ST Config:      {args.st_config}")
    logging.info(f" ST Checkpoint:  {args.st_checkpoint}")
    logging.info(f" Device:         {device}")
    logging.info("=" * 65)

    # 1. Load SpeechTokenizer model
    logging.info("[1/3] Loading SpeechTokenizer codec model...")
    with open(args.st_config, "r") as f:
        st_cfg = json.load(f)
    st_model = SpeechTokenizer(st_cfg)
    st_state = torch.load(args.st_checkpoint, map_location="cpu")
    st_model.load_state_dict(st_state)
    st_model.to(device)
    st_model.eval()
    logging.info("SpeechTokenizer successfully loaded and set to eval mode.")

    # 2. Load Manifest cuts
    logging.info(f"[2/3] Loading cuts from {manifest_path}...")
    cuts = CutSet.from_file(str(manifest_path))
    total_cuts = len(cuts)
    logging.info(f"Total available cuts in manifest: {total_cuts}")

    # Determine indices to process
    if args.indices is not None and len(args.indices) > 0:
        target_indices = [i for i in args.indices if 0 <= i < total_cuts]
    elif args.num_samples > 0:
        target_indices = list(range(min(args.num_samples, total_cuts)))
    else:
        target_indices = list(range(total_cuts))

    logging.info(f"[3/3] Decoding {len(target_indices)} samples into 16kHz audio waveforms...")

    summary_records = []

    with torch.no_grad():
        for seq_idx, cut_idx in enumerate(target_indices):
            cut = cuts[cut_idx]
            cut_id = cut.id
            spk = cut.custom.get("speaker", cut.supervisions[0].speaker if cut.supervisions else "unknown")
            text = cut.supervisions[0].text if cut.supervisions else ""

            # Load 8-layer generated acoustic tokens: [T, 8]
            try:
                codes_np = cut.load_features()
            except Exception as e:
                logging.error(f"Failed to load features for cut {cut_id} (index {cut_idx}): {e}")
                continue

            if codes_np is None or codes_np.ndim != 2:
                logging.warning(f"Invalid features shape for cut {cut_id}: {getattr(codes_np, 'shape', None)}")
                continue

            num_frames, num_layers = codes_np.shape
            # Convert to tensor: [8, 1, T] for SpeechTokenizer.decode(codes)
            codes_tensor = torch.from_numpy(codes_np).long().permute(1, 0).unsqueeze(1).to(device)

            # Decode tokens to audio waveform
            try:
                gen_wav = st_model.decode(codes_tensor).squeeze(0).cpu()  # [1, T_samples]
            except Exception as e:
                logging.error(f"Error decoding tokens for cut {cut_id}: {e}")
                continue

            gen_duration = gen_wav.shape[-1] / 16000.0
            gen_wav_path = out_dir / f"sample_{seq_idx:03d}_gen_{cut_id}.wav"
            torchaudio.save(str(gen_wav_path), gen_wav, 16000)

            # Optional: Load and save ground truth target audio if available
            gt_wav_path = None
            gt_duration = None
            if args.save_gt and cut.has_recording:
                try:
                    gt_audio_np = cut.load_audio()
                    gt_audio = torch.from_numpy(gt_audio_np).float()
                    if cut.sampling_rate != 16000:
                        gt_audio = torchaudio.functional.resample(gt_audio, cut.sampling_rate, 16000)
                    gt_wav_path = out_dir / f"sample_{seq_idx:03d}_gt_{cut_id}.wav"
                    torchaudio.save(str(gt_wav_path), gt_audio, 16000)
                    gt_duration = gt_audio.shape[-1] / 16000.0
                except Exception as ex:
                    logging.debug(f"Could not load GT audio for {cut_id}: {ex}")

            record = {
                "sample_idx": seq_idx,
                "cut_idx": cut_idx,
                "cut_id": cut_id,
                "speaker": spk,
                "text": text,
                "num_frames": int(num_frames),
                "num_layers": int(num_layers),
                "gen_duration_sec": round(gen_duration, 3),
                "gt_duration_sec": round(gt_duration, 3) if gt_duration else None,
                "gen_wav_file": gen_wav_path.name,
                "gt_wav_file": gt_wav_path.name if gt_wav_path else None,
            }
            summary_records.append(record)

            logging.info(
                f"[{seq_idx + 1}/{len(target_indices)}] Sample {seq_idx:03d} (Index {cut_idx}): "
                f"Spk='{spk}', Frames={num_frames}, GenDur={gen_duration:.2f}s | "
                f"Text: \"{text[:50]}{'...' if len(text) > 50 else ''}\""
            )

    # Save summary report
    summary_json_path = out_dir / "synthesis_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2, ensure_ascii=False)

    summary_txt_path = out_dir / "synthesis_summary.txt"
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(" VALL-E Native Token Synthesis Test Summary \n")
        f.write(f" Manifest: {manifest_path}\n")
        f.write(f" Output Directory: {out_dir}\n")
        f.write(f" Total Samples Decoded: {len(summary_records)}\n")
        f.write("=" * 80 + "\n\n")
        for rec in summary_records:
            f.write(f"--- Sample {rec['sample_idx']:03d} (Cut ID: {rec['cut_id']}) ---\n")
            f.write(f"  Speaker:      {rec['speaker']}\n")
            f.write(f"  Text:         {rec['text']}\n")
            f.write(f"  Token Shape:  [{rec['num_frames']}, {rec['num_layers']}]\n")
            f.write(f"  Gen Duration: {rec['gen_duration_sec']}s  -> {rec['gen_wav_file']}\n")
            if rec["gt_wav_file"]:
                f.write(f"  GT Duration:  {rec['gt_duration_sec']}s  -> {rec['gt_wav_file']}\n")
            f.write("\n")

    logging.info("=" * 65)
    logging.info(f" Successfully synthesized {len(summary_records)} audio files!")
    logging.info(f" Audio files and reports saved to: {out_dir}")
    logging.info(f" Summary JSON: {summary_json_path}")
    logging.info(f" Summary TXT:  {summary_txt_path}")
    logging.info("=" * 65)


if __name__ == "__main__":
    main()

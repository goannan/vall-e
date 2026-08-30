#!/usr/bin/env python3
# Copyright (c) 2026
# Speech synthesis script to decode VALL-E test tokens into Clean TTS audio
# and export corresponding Speaker Prompt audio and metadata.

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock

# 1. Clean Mocks for k2 / kaldialign to avoid missing optional dependencies
for mod in ["k2", "k2.version", "kaldialign"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import torch
import torchaudio
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]

def find_neumark_root(hint: Optional[str] = None) -> Path:
    candidates = [
        hint,
        os.environ.get("NEUMARK_ROOT"),
        PROJECT_DIR.parent / "NeuMark",
        PROJECT_DIR / "NeuMark",
        SCRIPT_DIR / "NeuMark",
        SCRIPT_DIR.parents[2] / "NeuMark",
        Path("/home/wu25/mrnas04home/projects/NeuMark"),
        Path("/home/pj25001109/ku60000344/projects/NeuMark"),
    ]
    for c in candidates:
        if c:
            p = Path(c)
            if not p.is_absolute():
                p = (SCRIPT_DIR / p).resolve()
            if p.is_dir():
                return p
    return (PROJECT_DIR.parent / "NeuMark").resolve()

NEUMARK_ROOT = find_neumark_root()
for p in [str(PROJECT_DIR), str(SCRIPT_DIR), str(NEUMARK_ROOT), str(NEUMARK_ROOT / "train")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from STmodels.model import SpeechTokenizer
from tts_native_dataset import get_tts_native_dataloader

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decode VALL-E Synthesized Test Tokens to Clean Audio & Save Prompt Audios"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/tokenized_voicemark/cuts_test_valle_native.jsonl.gz",
        help="Path to tokenized cuts manifest (.jsonl.gz)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp/valle_test_decoded_audio",
        help="Directory to save decoded clean audio, prompt audio, and metadata",
    )
    parser.add_argument(
        "--st-config",
        type=str,
        default=None,
        help="Path to SpeechTokenizer config JSON (optional)",
    )
    parser.add_argument(
        "--st-checkpoint",
        type=str,
        default=None,
        help="Path to SpeechTokenizer weights .pt (optional)",
    )
    parser.add_argument(
        "--neumark-root",
        type=str,
        default=None,
        help="Path to NeuMark repository root directory",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of samples to decode (-1 for ALL samples in manifest)",
    )
    parser.add_argument(
        "--save-gt",
        action="store_true",
        default=False,
        help="Also save ground truth target audio from LibriTTS",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Target device for SpeechTokenizer decoding (e.g. cuda:0, cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(SCRIPT_DIR)
    device = torch.device(args.device)

    # 1. Resolve Paths
    neumark_root = find_neumark_root(args.neumark_root)
    manifest_path = Path(args.manifest) if Path(args.manifest).is_absolute() else (SCRIPT_DIR / args.manifest)
    out_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else (SCRIPT_DIR / args.output_dir)
    
    st_cfg_path = Path(args.st_config) if args.st_config else (neumark_root / "STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json")
    st_ckpt_path = Path(args.st_checkpoint) if args.st_checkpoint else (neumark_root / "STmodels/pretrained_model/SpeechTokenizer.pt")

    # Output subdirectories
    clean_audio_dir = out_dir / "clean_tts"
    prompt_audio_dir = out_dir / "prompt"
    gt_audio_dir = out_dir / "ground_truth"

    clean_audio_dir.mkdir(parents=True, exist_ok=True)
    prompt_audio_dir.mkdir(parents=True, exist_ok=True)
    if args.save_gt:
        gt_audio_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 75)
    logging.info(" VALL-E Native Test Token Speech Synthesis & Prompt Export ")
    logging.info(f" Test Manifest:     {manifest_path}")
    logging.info(f" SpeechTokenizer:   {st_ckpt_path}")
    logging.info(f" Output Directory:  {out_dir}")
    logging.info(f" Clean Audio Dir:   {clean_audio_dir}")
    logging.info(f" Prompt Audio Dir:  {prompt_audio_dir}")
    logging.info(f" Device:            {device}")
    logging.info("=" * 75)

    if not manifest_path.exists():
        logging.error(f"Manifest file not found: {manifest_path}")
        sys.exit(1)
    if not st_ckpt_path.exists():
        logging.error(f"SpeechTokenizer checkpoint not found: {st_ckpt_path}")
        sys.exit(1)

    # 2. Load SpeechTokenizer Generator
    logging.info("[1/2] Loading SpeechTokenizer Generator...")
    generator = SpeechTokenizer.load_from_checkpoint(str(st_cfg_path), str(st_ckpt_path)).to(device)
    generator.eval()
    for p in generator.parameters():
        p.requires_grad = False

    # 3. Load Dataloader
    logging.info(f"[2/2] Loading Dataloader from {manifest_path}...")
    test_dl = get_tts_native_dataloader(
        manifest_path=str(manifest_path),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        max_duration=30.0,
    )
    total_samples = len(test_dl)
    num_decode = total_samples if args.num_samples <= 0 else min(args.num_samples, total_samples)
    logging.info(f"Total test cuts: {total_samples} | Samples to synthesize: {num_decode}")

    metadata_records = []
    csv_rows = []

    # 4. Decoding and Saving Loop
    logging.info("=" * 75)
    logging.info(f" Synthesizing Clean Audio for {num_decode} Samples...")
    logging.info("=" * 75)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dl, total=num_decode, desc="Synthesizing Clean Audio", ncols=100)):
            if i >= num_decode:
                break

            codes = batch["codes"].to(device)  # [1, 8, T]
            real_audio = batch["audio"]  # [1, 1, T_samples]
            prompt_audio = batch["prompt_audio"]  # [1, 1, T_p]
            texts = batch["texts"]
            cut_ids = batch["ids"]
            cut_id = cut_ids[0] if cut_ids else f"sample_{i:04d}"
            ref_text = texts[0] if texts else ""

            # Permute codes to [8, 1, T] for SpeechTokenizer RVQ decode
            codes_qbt = codes.permute(1, 0, 2).contiguous() if codes.shape[1] == 8 else codes
            quantized_layers = [generator.quantizer.decode(codes_qbt[k : k + 1], st=k) for k in range(8)]
            z_clean = sum(quantized_layers)
            clean_audio = generator.decoder(z_clean)  # [1, 1, T_samples]

            clean_wav_path = clean_audio_dir / f"{cut_id}_clean_tts.wav"
            prompt_wav_path = prompt_audio_dir / f"{cut_id}_prompt.wav"

            # Save clean synthesized audio and prompt audio (16kHz standard)
            torchaudio.save(str(clean_wav_path), clean_audio.squeeze(0).cpu(), 16000)
            torchaudio.save(str(prompt_wav_path), prompt_audio.squeeze(0).cpu(), 16000)

            record = {
                "sample_idx": i,
                "cut_id": cut_id,
                "text": ref_text,
                "clean_tts_wav": str(clean_wav_path),
                "clean_tts_relpath": f"clean_tts/{clean_wav_path.name}",
                "prompt_wav": str(prompt_wav_path),
                "prompt_relpath": f"prompt/{prompt_wav_path.name}",
                "clean_duration_sec": round(clean_audio.shape[-1] / 16000.0, 3),
                "prompt_duration_sec": round(prompt_audio.shape[-1] / 16000.0, 3),
            }

            if args.save_gt:
                gt_wav_path = gt_audio_dir / f"{cut_id}_gt.wav"
                torchaudio.save(str(gt_wav_path), real_audio.squeeze(0).cpu(), 16000)
                record["gt_wav"] = str(gt_wav_path)
                record["gt_relpath"] = f"ground_truth/{gt_wav_path.name}"
                record["gt_duration_sec"] = round(real_audio.shape[-1] / 16000.0, 3)

            metadata_records.append(record)
            csv_rows.append({
                "Index": i,
                "Cut_ID": cut_id,
                "Clean_TTS_File": clean_wav_path.name,
                "Prompt_File": prompt_wav_path.name,
                "Duration_Sec": record["clean_duration_sec"],
                "Prompt_Duration_Sec": record["prompt_duration_sec"],
                "Text": ref_text,
            })

    # 5. Export Manifests & Metadata
    meta_json_file = out_dir / "metadata.json"
    with open(meta_json_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_synthesized": len(metadata_records),
            "manifest_source": str(manifest_path),
            "speechtokenizer_checkpoint": str(st_ckpt_path),
            "records": metadata_records,
        }, f, indent=4, ensure_ascii=False)
    logging.info(f"Saved JSON metadata to: {meta_json_file}")

    meta_jsonl_file = out_dir / "metadata.jsonl"
    with open(meta_jsonl_file, "w", encoding="utf-8") as f:
        for rec in metadata_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logging.info(f"Saved JSONL manifest to: {meta_jsonl_file}")

    meta_csv_file = out_dir / "metadata.csv"
    with open(meta_csv_file, "w", newline="", encoding="utf-8") as f:
        if csv_rows:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    logging.info(f"Saved CSV index to: {meta_csv_file}")

    logging.info("=" * 75)
    logging.info(f" Speech Synthesis Completed! Output saved to: {out_dir}")
    logging.info("=" * 75)


if __name__ == "__main__":
    main()

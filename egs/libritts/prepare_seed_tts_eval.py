#!/usr/bin/env python3
"""Download the prompt side of the standard Seed-TTS-Eval TTS set."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/seed_tts_eval"),
    )
    parser.add_argument(
        "--language",
        choices=["en", "zh"],
        default="en",
    )
    parser.add_argument(
        "--repo-id",
        default="zhaochenyang20/seed-tts-eval",
        help=(
            "Hugging Face mirror of BytedanceSpeech/seed-tts-eval. Only the "
            "manifest and prompt WAVs are downloaded."
        ),
    )
    return parser.parse_args()


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(args.output_dir),
        allow_patterns=[
            f"{args.language}/meta.lst",
            f"{args.language}/prompt-wavs/*.wav",
        ],
    )

    manifest = args.output_dir / args.language / "meta.lst"
    if not manifest.is_file():
        raise FileNotFoundError(f"Downloaded manifest is missing: {manifest}")
    row_count = sum(1 for line in manifest.open(encoding="utf-8") if line.strip())
    print(f"Seed-TTS-Eval {args.language}: {row_count} rows")
    print(f"Manifest: {manifest.resolve()}")


if __name__ == "__main__":
    main()

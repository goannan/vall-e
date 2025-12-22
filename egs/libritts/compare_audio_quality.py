import argparse
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
from pesq import pesq
from pystoi import stoi
import soundfile as sf


def load_mono_resampled(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio, convert to mono, and resample to the target rate."""
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    audio = audio.astype(np.float32)
    return audio, target_sr


def compute_metrics(ref_path: Path, deg_path: Path, target_sr: int = 16000):
    ref, sr_ref = load_mono_resampled(ref_path, target_sr)
    deg, sr_deg = load_mono_resampled(deg_path, target_sr)

    if sr_ref != sr_deg:
        raise ValueError("Sample rates do not match after resampling")

    # Align lengths so metrics operate on the same window.
    min_len = min(len(ref), len(deg))
    ref = ref[:min_len]
    deg = deg[:min_len]

    pesq_wb = pesq(sr_ref, ref, deg, "wb")
    stoi_score = stoi(ref, deg, sr_ref, extended=False)

    return {
        "sr": sr_ref,
        "pesq_wb": float(pesq_wb),
        "stoi": float(stoi_score),
        "duration_s": min_len / sr_ref,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare multiple test wavs against one reference using PESQ and STOI.")
    parser.add_argument("ref", type=Path, help="Reference (clean) wav file")
    parser.add_argument("deg", type=Path, nargs="+", help="One or more degraded/test wav files")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate for metrics (8k or 16k)")
    args = parser.parse_args()

    for idx, deg_path in enumerate(args.deg, start=1):
        metrics = compute_metrics(args.ref, deg_path, args.sr)
        print(f"=== Result {idx}: {deg_path} ===")
        print(f"Sample rate      : {metrics['sr']} Hz")
        print(f"Aligned duration : {metrics['duration_s']:.2f} s")
        print(f"PESQ (wideband)  : {metrics['pesq_wb']:.3f}")
        print(f"STOI             : {metrics['stoi']:.3f}")
        print()


if __name__ == "__main__":
    main()

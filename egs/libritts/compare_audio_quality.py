import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple

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

    # PESQ requires at least 0.25s; pad tail if too short to avoid BufferTooShortError.
    min_required = int(0.25 * sr_ref)
    if min_len < min_required:
        pad = min_required - min_len
        ref = np.pad(ref, (0, pad))
        deg = np.pad(deg, (0, pad))

    try:
        pesq_wb = pesq(sr_ref, ref, deg, "wb")
    except Exception:
        pesq_wb = None

    try:
        stoi_score = stoi(ref, deg, sr_ref, extended=False)
    except Exception:
        stoi_score = None

    visqol_moslqo = run_visqol_cli(ref_path, deg_path, sr_ref)

    return {
        "sr": sr_ref,
        "pesq_wb": float(pesq_wb) if pesq_wb is not None else None,
        "stoi": float(stoi_score) if stoi_score is not None else None,
        "duration_s": min_len / sr_ref,
        "visqol_moslqo": float(visqol_moslqo) if visqol_moslqo is not None else None,
    }


def run_visqol_cli(ref_path: Path, deg_path: Path, sr: int) -> Optional[float]:
    """Run ViSQOL via external binary if configured.

    Requires env vars:
      VISQOL_BIN: path to visqol executable
      VISQOL_MODEL: path to speech similarity_to_quality_model (libsvm file)
    Returns MOS-LQO or None if unavailable/fails.
    """
    visqol_bin = os.environ.get("VISQOL_BIN")
    visqol_model = os.environ.get("VISQOL_MODEL")
    if not visqol_bin or not visqol_model:
        return None

    cmd = [
        visqol_bin,
        f"--reference_file={ref_path}",
        f"--degraded_file={deg_path}",
        "--use_speech_mode=true",
        f"--similarity_to_quality_model={visqol_model}",
        f"--audio_sample_rate={sr}",
        "--verbose=false",
    ]

    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except Exception:
        return None

    m = re.search(r"MOS-LQO\s*:?\s*([0-9.]+)", out)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


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

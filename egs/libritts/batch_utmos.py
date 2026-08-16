#!/usr/bin/env python3
"""Evaluate UTMOS on paired clean and watermarked WAV files only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
from tqdm import tqdm


SAMPLE_RATE = 16_000
# SpeechMOS v1.2.0 uses the standard seven-layer Wav2Vec2 feature extractor:
# [(kernel, stride) = (10, 5), (3, 2) x4, (2, 2) x2]. Its exact minimum
# receptive field is 400 samples. Shorter VALL-E early-EOS outputs are skipped
# because padding them would assign a MOS to an invalid, non-speech fragment.
MIN_UTMOS_SAMPLES = 400
DEFAULT_REPOSITORY = "tarepan/SpeechMOS:v1.2.0"
DEFAULT_MODEL = "utmos22_strong"


def load_predictor(
    repository: str, model_name: str, device: torch.device
) -> torch.nn.Module:
    local_repository = Path(repository).expanduser()
    if local_repository.is_dir():
        predictor = torch.hub.load(
            str(local_repository.resolve()), model_name, source="local"
        )
    else:
        predictor = torch.hub.load(
            repository,
            model_name,
            source="github",
            trust_repo=True,
        )
    return predictor.eval().to(device)


def load_mono_16k(path: Path, device: torch.device) -> torch.Tensor:
    wave, sample_rate = torchaudio.load(str(path))
    wave = wave.float()
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        wave = torchaudio.functional.resample(wave, sample_rate, SAMPLE_RATE)
    return wave.to(device)


def discover_pairs(audio_dir: Path) -> Tuple[List[Tuple[Path, Path]], List[Path]]:
    pairs = []
    missing = []
    for clean_path in sorted(audio_dir.glob("*_clean.wav")):
        wm_path = clean_path.with_name(
            clean_path.name[: -len("_clean.wav")] + "_wm.wav"
        )
        if wm_path.is_file():
            pairs.append((clean_path, wm_path))
        else:
            missing.append(wm_path)
    if not pairs:
        raise RuntimeError(f"No complete *_clean.wav/*_wm.wav pairs in {audio_dir}")
    return pairs, missing


def file_signature(path: Path) -> Dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def cache_is_complete(
    output_json: Path,
    pairs: List[Tuple[Path, Path]],
    repository: str,
    model_name: str,
) -> bool:
    if not output_json.is_file():
        return False
    try:
        cached = json.loads(output_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    model = cached.get("utmos_model") or {}
    if model.get("repository") != repository or model.get("entrypoint") != model_name:
        return False
    if cached.get("short_input_policy") != "skip_below_400_samples_at_16khz":
        return False
    details = cached.get("details") or []
    skipped_short = cached.get("skipped_short") or []
    if cached.get("input_pair_count") != len(pairs):
        return False
    if cached.get("count") != len(details):
        return False
    if cached.get("skipped_short_count") != len(skipped_short):
        return False
    if len(details) + len(skipped_short) != len(pairs):
        return False
    cached_rows = {
        (row.get("ref"), row.get("deg")): row
        for row in details + skipped_short
    }
    for clean_path, wm_path in pairs:
        row = cached_rows.get((clean_path.name, wm_path.name))
        if row is None:
            return False
        if row.get("ref_signature") != file_signature(clean_path):
            return False
        if row.get("deg_signature") != file_signature(wm_path):
            return False
    return True


def score_prepared_wave(
    predictor: torch.nn.Module, wave: torch.Tensor, path: Path
) -> float:
    try:
        with torch.inference_mode():
            score = predictor(wave, SAMPLE_RATE)
    except Exception as exc:
        raise RuntimeError(
            f"UTMOS inference failed for {path} "
            f"(prepared_samples_16k={wave.shape[-1]})"
        ) from exc
    value = float(score.detach().cpu().reshape(-1)[0])
    if not torch.isfinite(torch.tensor(value)):
        raise RuntimeError(f"UTMOS returned a non-finite score for {path}")
    return value


def score_wave(
    predictor: torch.nn.Module, path: Path, device: torch.device
) -> float:
    wave = load_mono_16k(path, device)
    if wave.shape[-1] < MIN_UTMOS_SAMPLES:
        raise ValueError(
            f"UTMOS input is too short: {path} "
            f"({wave.shape[-1]} < {MIN_UTMOS_SAMPLES} samples at 16 kHz)"
        )
    return score_prepared_wave(predictor, wave, path)


def evaluate(
    audio_dir: Path,
    output_json: Path,
    repository: str,
    model_name: str,
    device: torch.device,
    force: bool = False,
) -> Dict[str, Any]:
    pairs, missing = discover_pairs(audio_dir)
    if not force and cache_is_complete(output_json, pairs, repository, model_name):
        print(f"UTMOS cache complete; reuse: {output_json}", flush=True)
        return json.loads(output_json.read_text(encoding="utf-8"))

    print(f"Loading UTMOS: {repository} / {model_name} on {device}", flush=True)
    predictor = load_predictor(repository, model_name, device)
    details: List[Dict[str, Any]] = []
    skipped_short: List[Dict[str, Any]] = []
    for clean_path, wm_path in tqdm(pairs, desc="UTMOS clean/wm", dynamic_ncols=True):
        clean_wave = load_mono_16k(clean_path, device)
        wm_wave = load_mono_16k(wm_path, device)
        clean_samples = clean_wave.shape[-1]
        wm_samples = wm_wave.shape[-1]
        if min(clean_samples, wm_samples) < MIN_UTMOS_SAMPLES:
            skipped_short.append(
                {
                    "ref": clean_path.name,
                    "deg": wm_path.name,
                    "reason": "shorter_than_utmos_minimum",
                    "clean_samples_16k": clean_samples,
                    "wm_samples_16k": wm_samples,
                    "minimum_samples_16k": MIN_UTMOS_SAMPLES,
                    "ref_signature": file_signature(clean_path),
                    "deg_signature": file_signature(wm_path),
                }
            )
            tqdm.write(
                f"Skip short UTMOS pair: {clean_path.name} / {wm_path.name}: "
                f"clean={clean_samples}, wm={wm_samples}, "
                f"minimum={MIN_UTMOS_SAMPLES} samples at 16 kHz"
            )
            continue
        clean_score = score_prepared_wave(predictor, clean_wave, clean_path)
        wm_score = score_prepared_wave(predictor, wm_wave, wm_path)
        details.append(
            {
                "ref": clean_path.name,
                "deg": wm_path.name,
                "utmos_clean": clean_score,
                "utmos_wm": wm_score,
                "utmos_delta_wm_minus_clean": wm_score - clean_score,
                "ref_signature": file_signature(clean_path),
                "deg_signature": file_signature(wm_path),
            }
        )

    count = len(details)
    if count == 0:
        raise RuntimeError(
            f"All {len(pairs)} UTMOS pairs were shorter than "
            f"{MIN_UTMOS_SAMPLES} samples at 16 kHz"
        )
    summary: Dict[str, Any] = {
        "input_pair_count": len(pairs),
        "count": count,
        "skipped_short_count": len(skipped_short),
        "avg_utmos_clean": sum(row["utmos_clean"] for row in details) / count,
        "avg_utmos_wm": sum(row["utmos_wm"] for row in details) / count,
        "avg_utmos_delta_wm_minus_clean": sum(
            row["utmos_delta_wm_minus_clean"] for row in details
        )
        / count,
        "utmos_model": {
            "repository": repository,
            "entrypoint": model_name,
            "sample_rate": SAMPLE_RATE,
            "score": "predicted naturalness MOS; higher is better",
        },
        "audio_dir": str(audio_dir.resolve()),
        "short_input_policy": "skip_below_400_samples_at_16khz",
        "missing_pairs": [str(path) for path in missing],
        "skipped_short": skipped_short,
        "details": details,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "UTMOS-only evaluation for *_clean.wav/*_wm.wav pairs. "
            "No WER, SIM, PESQ, STOI, SI-SNR, or watermark detector is run."
        )
    )
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--repo", default=DEFAULT_REPOSITORY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    audio_dir = args.dir.expanduser().resolve()
    if not audio_dir.is_dir():
        parser.error(f"Missing audio directory: {audio_dir}")
    output_json = (
        args.json.expanduser().resolve()
        if args.json is not None
        else audio_dir / "utmos_clean_wm.json"
    )
    summary = evaluate(
        audio_dir=audio_dir,
        output_json=output_json,
        repository=args.repo,
        model_name=args.model,
        device=torch.device(args.device),
        force=args.force,
    )
    print(f"Input pairs : {summary['input_pair_count']}")
    print(f"UTMOS pairs : {summary['count']}")
    print(f"Skipped     : {summary['skipped_short_count']} short pairs")
    print(f"UTMOS clean : {summary['avg_utmos_clean']:.4f}")
    print(f"UTMOS wm    : {summary['avg_utmos_wm']:.4f}")
    print(
        "UTMOS delta : "
        f"{summary['avg_utmos_delta_wm_minus_clean']:+.4f} (wm - clean)"
    )
    print(f"UTMOS JSON  : {output_json}")


if __name__ == "__main__":
    main()

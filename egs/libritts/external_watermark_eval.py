#!/usr/bin/env python3
"""Embed and evaluate AudioSeal or WavMark on existing Seed-TTS clean WAVs.

Both systems are evaluated with a deterministic 16-bit binary payload.  The
script writes ``<id>_clean.wav``/``<id>_wm.wav`` pairs so the existing official
Seed-TTS WER/SIM pipeline can consume the result directory directly.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


SAMPLE_RATE = 16_000
BITS_PER_MESSAGE = 16
DEFAULT_UTMOS_REPO = "tarepan/SpeechMOS:v1.2.0"
DEFAULT_UTMOS_MODEL = "utmos22_strong"
WAVMARK_SYNC_BITS = np.asarray(
    [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0], dtype=np.int64
)


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def utterance_id(path: Path) -> str:
    suffix = "_clean.wav"
    if not path.name.endswith(suffix):
        raise ValueError(f"Expected *{suffix}: {path}")
    return path.name[: -len(suffix)]


def payload_for(utt_id: str) -> np.ndarray:
    """Stable payload shared by AudioSeal and WavMark for a given utterance."""
    digest = hashlib.sha256(f"external-watermark-v1:{utt_id}".encode()).digest()[:2]
    return np.unpackbits(np.frombuffer(digest, dtype=np.uint8)).astype(np.int64)


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode()).digest()[:8]
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def load_mono_16k(path: Path, device: torch.device | None = None) -> torch.Tensor:
    wav, sample_rate = torchaudio.load(str(path))
    wav = wav.float()
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sample_rate, SAMPLE_RATE)
    wav = wav.unsqueeze(0)  # [batch, channel, time]
    return wav.to(device) if device is not None else wav


def save_wave(path: Path, wav: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = wav.detach().float().cpu().reshape(1, -1).clamp(-1.0, 1.0)
    torchaudio.save(
        str(path), audio, SAMPLE_RATE, encoding="PCM_F", bits_per_sample=32
    )


def link_or_copy(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


class ExternalWatermarker:
    def __init__(
        self,
        backend: str,
        device: torch.device,
        wavmark_root: Path,
        need_generator: bool,
        need_detector: bool,
        wavmark_checkpoint: str,
    ) -> None:
        self.backend = backend
        self.device = device
        self.generator = None
        self.detector = None
        self.model = None
        self.wavmark = None

        if backend == "audioseal":
            from audioseal import AudioSeal

            if need_generator:
                print("Loading AudioSeal generator: audioseal_wm_16bits", flush=True)
                self.generator = AudioSeal.load_generator(
                    "audioseal_wm_16bits"
                ).eval().to(device)
            if need_detector:
                print("Loading AudioSeal detector: audioseal_detector_16bits", flush=True)
                self.detector = AudioSeal.load_detector(
                    "audioseal_detector_16bits"
                ).eval().to(device)
        elif backend == "wavmark":
            wavmark_src = wavmark_root / "src"
            sys.path.insert(0, str(wavmark_src))
            # The repository root can otherwise be resolved as an empty
            # namespace package named wavmark.
            sys.modules.pop("wavmark", None)
            import wavmark

            self.wavmark = wavmark
            print(f"Loading WavMark checkpoint: {wavmark_checkpoint}", flush=True)
            self.model = wavmark.load_model(wavmark_checkpoint).eval().to(device)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def embed(self, clean: torch.Tensor, payload: np.ndarray) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.backend == "audioseal":
            assert self.generator is not None
            message = torch.as_tensor(payload, device=self.device).reshape(1, -1)
            with torch.inference_mode():
                watermarked = self.generator(
                    clean, sample_rate=SAMPLE_RATE, message=message
                )
            return watermarked, {}

        assert self.wavmark is not None and self.model is not None
        original_length = clean.shape[-1]
        # WavMark uses 1.0 s watermark regions followed by 0.1 s shift regions.
        # Pad only for encoding, then restore the exact original length.
        padded_length = max(original_length, 17_600)
        padded = torch.nn.functional.pad(clean, (0, padded_length - original_length))
        signal_wmd, info = self.wavmark.encode_watermark(
            self.model,
            padded.detach().cpu().reshape(-1).numpy(),
            payload,
            show_progress=False,
        )
        watermarked = torch.from_numpy(np.asarray(signal_wmd)).float()
        return watermarked[:original_length].reshape(1, 1, -1).to(self.device), info

    def detect(self, audio: torch.Tensor, payload: np.ndarray) -> Tuple[float, float, List[int]]:
        if self.backend == "audioseal":
            assert self.detector is not None
            with torch.inference_mode():
                probability, decoded = self.detector.detect_watermark(
                    audio, sample_rate=SAMPLE_RATE
                )
            decoded_np = decoded.detach().cpu().reshape(-1).numpy().astype(np.int64)
            decoded_np = decoded_np[:BITS_PER_MESSAGE]
            bit_acc = float(np.mean(decoded_np == payload))
            return float(probability), bit_acc, decoded_np.tolist()

        assert self.model is not None
        signal = audio.detach().cpu().reshape(-1).numpy()
        if len(signal) < SAMPLE_RATE:
            decoded = np.zeros(BITS_PER_MESSAGE, dtype=np.int64)
            return 0.0, float(np.mean(decoded == payload)), decoded.tolist()

        positions = list(range(0, len(signal) - SAMPLE_RATE + 1, 800))
        best_equal = -1
        best_payload = np.zeros(BITS_PER_MESSAGE, dtype=np.int64)
        exact_payloads: List[np.ndarray] = []
        batch_size = 32
        for start in range(0, len(positions), batch_size):
            batch_positions = positions[start : start + batch_size]
            windows = np.stack(
                [signal[p : p + SAMPLE_RATE] for p in batch_positions]
            )
            with torch.inference_mode():
                decoded_batch = (
                    self.model.decode(torch.from_numpy(windows).float().to(self.device))
                    >= 0.5
                ).int().cpu().numpy()
            for decoded in decoded_batch:
                equal = int(np.sum(decoded[:16] == WAVMARK_SYNC_BITS))
                candidate = decoded[16:32].astype(np.int64)
                if equal > best_equal:
                    best_equal = equal
                    best_payload = candidate
                if equal == 16:
                    exact_payloads.append(candidate)

        if exact_payloads:
            best_payload = (np.mean(np.stack(exact_payloads), axis=0) >= 0.5).astype(
                np.int64
            )
            probability = 1.0
        else:
            probability = 0.0
        bit_acc = float(np.mean(best_payload == payload))
        return probability, bit_acc, best_payload.tolist()


def selected_sources(source_dir: Path, limit: int | None) -> List[Path]:
    sources = sorted(source_dir.glob("*_clean.wav"))
    if limit is not None:
        sources = sources[:limit]
    if not sources:
        raise RuntimeError(f"No *_clean.wav files found in {source_dir}")
    return sources


def embed_all(args, device: torch.device) -> Dict[str, Any]:
    sources = selected_sources(args.source_dir, args.limit)
    watermarker = ExternalWatermarker(
        args.backend,
        device,
        args.wavmark_root,
        need_generator=True,
        need_detector=False,
        wavmark_checkpoint=args.wavmark_checkpoint,
    )
    embedded = 0
    reused = 0
    skipped: List[Dict[str, Any]] = []
    for source in tqdm(sources, desc=f"Embed {args.backend}", dynamic_ncols=True):
        utt_id = utterance_id(source)
        clean_path = args.output_dir / f"{utt_id}_clean.wav"
        wm_path = args.output_dir / f"{utt_id}_wm.wav"
        metadata_path = args.output_dir / f"{utt_id}_wm.json"
        alias_path = args.output_dir / f"{utt_id}.wav"
        if (
            not args.force
            and clean_path.is_file()
            and wm_path.is_file()
            and metadata_path.is_file()
        ):
            if not alias_path.is_file():
                link_or_copy(wm_path, alias_path)
            reused += 1
            continue

        clean = load_mono_16k(source, device)
        duration = clean.shape[-1] / SAMPLE_RATE
        if duration < args.min_duration:
            skipped.append(
                {"utterance_id": utt_id, "duration_sec": duration, "reason": "too_short"}
            )
            continue

        payload = payload_for(utt_id)
        watermarked, embed_info = watermarker.embed(clean, payload)
        if not torch.isfinite(watermarked).all():
            skipped.append(
                {"utterance_id": utt_id, "duration_sec": duration, "reason": "non_finite_output"}
            )
            continue
        save_wave(clean_path, clean)
        save_wave(wm_path, watermarked)
        link_or_copy(wm_path, alias_path)
        metadata = {
            "utterance_id": utt_id,
            "backend": args.backend,
            "payload_bits": payload.tolist(),
            "bits_per_message": BITS_PER_MESSAGE,
            "sample_rate": SAMPLE_RATE,
            "source_clean_wav": str(source.resolve()),
            "duration_sec": duration,
            "embed_info": embed_info,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        embedded += 1

    summary = {
        "backend": args.backend,
        "source_dir": str(args.source_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "source_count": len(sources),
        "embedded_count": embedded,
        "reused_count": reused,
        "output_pair_count": len(list(args.output_dir.glob("*_wm.wav"))),
        "skipped": skipped,
        "sample_rate": SAMPLE_RATE,
        "bits_per_message": BITS_PER_MESSAGE,
    }
    (args.output_dir / "embedding_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def load_pairs(output_dir: Path) -> List[Tuple[str, Path, Path, np.ndarray]]:
    pairs = []
    for wm_path in sorted(output_dir.glob("*_wm.wav")):
        utt_id = wm_path.name[: -len("_wm.wav")]
        clean_path = output_dir / f"{utt_id}_clean.wav"
        metadata_path = output_dir / f"{utt_id}_wm.json"
        if not clean_path.is_file() or not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload = np.asarray(metadata["payload_bits"], dtype=np.int64)
        if payload.shape != (BITS_PER_MESSAGE,):
            raise ValueError(f"Invalid 16-bit payload in {metadata_path}")
        pairs.append((utt_id, clean_path, wm_path, payload))
    if not pairs:
        raise RuntimeError(f"No complete clean/wm/metadata pairs in {output_dir}")
    return pairs


def load_utmos_predictor(
    repository: str, model_name: str, device: torch.device
) -> torch.nn.Module:
    local_repository = Path(repository).expanduser()
    if local_repository.is_dir():
        predictor = torch.hub.load(
            str(local_repository.resolve()), model_name, source="local"
        )
    else:
        predictor = torch.hub.load(
            repository, model_name, source="github", trust_repo=True
        )
    return predictor.eval().to(device)


def evaluate_quality(args, device: torch.device) -> Dict[str, Any]:
    module = load_module_from_path(
        "external_batch_watermark_quality", args.recipe_dir / "batch_watermark_quality.py"
    )
    predictor = None
    utmos_scorer = None
    if not args.skip_utmos:
        print(
            f"Loading UTMOS predictor: {args.utmos_repo} / {args.utmos_model}",
            flush=True,
        )
        predictor = load_utmos_predictor(args.utmos_repo, args.utmos_model, device)

        def utmos_scorer(path: Path) -> float:
            wave = load_mono_16k(path, device).squeeze(1)
            with torch.inference_mode():
                score = predictor(wave, SAMPLE_RATE)
            return float(score.detach().cpu().reshape(-1)[0])

    metric_names = "PESQ-WB, STOI, and SI-SNR"
    if utmos_scorer is not None:
        metric_names += ", plus UTMOS for clean and watermarked audio"
    print(f"Audio quality: {metric_names}", flush=True)
    summary = module.evaluate_pairs(
        args.output_dir, SAMPLE_RATE, utmos_scorer=utmos_scorer
    )
    if predictor is not None:
        summary["utmos_model"] = {
            "repository": args.utmos_repo,
            "entrypoint": args.utmos_model,
            "sample_rate": SAMPLE_RATE,
            "score": "predicted naturalness MOS; higher is better",
        }
        utmos_scorer = None
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    path = args.output_dir / "audio_quality_clean_vs_wm.json"
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return float(sum(items) / len(items))


def evaluate_robustness(args, device: torch.device) -> Dict[str, Any]:
    pairs = load_pairs(args.output_dir)
    attacks_module = load_module_from_path(
        "external_valid_attacks", args.vall_e_root / "valle" / "bin" / "attacks.py"
    )
    attacks = attacks_module.build_voicemark_valid_attacks(SAMPLE_RATE)
    if args.skip_codecs:
        attacks = [attack for attack in attacks if not attack[2]]

    watermarker = ExternalWatermarker(
        args.backend,
        device,
        args.wavmark_root,
        need_generator=False,
        need_detector=True,
        wavmark_checkpoint=args.wavmark_checkpoint,
    )
    cache_dir = args.output_dir / "attack_results"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}

    for attack_name, attack_fn, is_codec in attacks:
        cache_path = cache_dir / f"{slugify(attack_name)}.json"
        if cache_path.is_file() and not args.force:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("count") == len(pairs):
                print(f"Attack cached: {attack_name}", flush=True)
                results[attack_name] = cached
                continue

        wm_rows: List[Dict[str, Any]] = []
        clean_rows: List[Dict[str, Any]] = []
        attack_error = None
        iterator = tqdm(pairs, desc=f"Attack {attack_name}", dynamic_ncols=True)
        for utt_id, clean_path, wm_path, payload in iterator:
            try:
                clean = load_mono_16k(clean_path, device)
                wm = load_mono_16k(wm_path, device)
                seed = stable_seed(args.backend, attack_name, utt_id)
                random.seed(seed)
                np.random.seed(seed & 0xFFFFFFFF)
                torch.manual_seed(seed)
                attacked_wm = attack_fn(wm)
                random.seed(seed)
                np.random.seed(seed & 0xFFFFFFFF)
                torch.manual_seed(seed)
                attacked_clean = attack_fn(clean)
                if isinstance(attacked_wm, tuple):
                    attacked_wm = attacked_wm[0]
                if isinstance(attacked_clean, tuple):
                    attacked_clean = attacked_clean[0]
                wm_prob, wm_acc, wm_decoded = watermarker.detect(attacked_wm, payload)
                clean_prob, clean_acc, clean_decoded = watermarker.detect(
                    attacked_clean, payload
                )
                wm_rows.append(
                    {
                        "utterance_id": utt_id,
                        "prob": wm_prob,
                        "bit_acc": wm_acc,
                        "decoded_bits": wm_decoded,
                    }
                )
                clean_rows.append(
                    {
                        "utterance_id": utt_id,
                        "prob": clean_prob,
                        "bit_acc": clean_acc,
                        "decoded_bits": clean_decoded,
                    }
                )
            except Exception as exc:  # preserve partial progress and report the row
                if is_codec and not wm_rows:
                    attack_error = f"{type(exc).__name__}: {exc}"
                    iterator.write(f"Codec attack unavailable: {attack_name}: {attack_error}")
                    break
                iterator.write(f"Skip {utt_id}: {type(exc).__name__}: {exc}")

        if is_codec:
            attacks_module.release_codec_models()
        if attack_error is not None:
            stats: Dict[str, Any] = {
                "attack": attack_name,
                "is_codec": is_codec,
                "available": False,
                "error": attack_error,
                "count": 0,
                "expected_count": len(pairs),
                "details": [],
            }
        elif not wm_rows:
            raise RuntimeError(f"No successful rows for attack: {attack_name}")
        else:
            wm_prob = mean(row["prob"] for row in wm_rows)
            clean_prob = mean(row["prob"] for row in clean_rows)
            stats = {
                "attack": attack_name,
                "is_codec": is_codec,
                "available": True,
                "count": len(wm_rows),
                "expected_count": len(pairs),
                "detect_acc": (wm_prob + (1.0 - clean_prob)) / 2.0,
                "wm_bit_acc": mean(row["bit_acc"] for row in wm_rows),
                "wm_prob": wm_prob,
                "clean_bit_acc": mean(row["bit_acc"] for row in clean_rows),
                "clean_prob": clean_prob,
                "details": [
                    {"wm": wm_row, "clean": clean_row}
                    for wm_row, clean_row in zip(wm_rows, clean_rows)
                ],
            }
        cache_path.write_text(
            json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        results[attack_name] = stats

    aggregate_results = {
        name: {key: value for key, value in stats.items() if key != "details"}
        for name, stats in results.items()
    }
    summary = {
        "watermark_backend": args.backend,
        "count": len(pairs),
        "bits_per_message": BITS_PER_MESSAGE,
        "bit_accuracy_unit": "binary_bit",
        "bit_accuracy_version": 2,
        "detection_threshold": 0.5 if args.backend == "audioseal" else "exact_16_bit_sync_pattern",
        "attack_suite": "VoiceMark valid.py DSP plus Encodec/DAC/SNAC bandwidth attacks",
        "attack_result_dir": str(cache_dir.resolve()),
        "attacks": aggregate_results,
    }
    summary_path = args.output_dir / "watermark_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    table = format_robustness_table(args.backend, len(pairs), results)
    (args.output_dir / "watermark_validation_table.txt").write_text(
        table, encoding="utf-8"
    )
    print(table, flush=True)
    return summary


def format_robustness_table(
    backend: str, count: int, results: Dict[str, Dict[str, Any]]
) -> str:
    width = 138
    lines = [
        "=" * width,
        f"Watermark model: {backend} | evaluated pairs: {count} | payload: 16 binary bits",
        "=" * width,
        f"{'Attack Type':<43} | {'Detect ACC':>10} | {'WM Bit Acc':>10} | {'WM Prob':>9} | {'Clean B.Acc':>11} | {'Clean Prob':>10} | {'N':>5}",
        "-" * width,
    ]
    for attack_name, stats in results.items():
        if not stats.get("available", True):
            lines.append(f"{attack_name:<43} | {'N/A':>10} | {'N/A':>10} | {'N/A':>9} | {'N/A':>11} | {'N/A':>10} | {0:>5}")
            lines.append(f"  unavailable: {stats.get('error', 'unknown error')}")
            continue
        lines.append(
            f"{attack_name:<43} | {stats['detect_acc']:>10.4f} | "
            f"{stats['wm_bit_acc']:>10.4f} | {stats['wm_prob']:>9.4f} | "
            f"{stats['clean_bit_acc']:>11.4f} | {stats['clean_prob']:>10.4f} | "
            f"{stats['count']:>5d}"
        )
    lines.extend(
        [
            "=" * width,
            "Detect ACC = (mean WM detection probability + 1 - mean clean false-positive probability) / 2.",
            "Bit Acc is the binary accuracy of the same deterministic 16-bit payload for both models.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args():
    recipe_dir = Path(__file__).resolve().parent
    projects_root = recipe_dir.parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("audioseal", "wavmark"), required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stages",
        default="embed,quality,robustness",
        help="Comma-separated subset of embed,quality,robustness.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--skip-codecs", action="store_true")
    parser.add_argument("--skip-utmos", action="store_true")
    parser.add_argument("--utmos-repo", default=DEFAULT_UTMOS_REPO)
    parser.add_argument("--utmos-model", default=DEFAULT_UTMOS_MODEL)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--recipe-dir", type=Path, default=recipe_dir)
    parser.add_argument("--vall-e-root", type=Path, default=projects_root / "vall-e")
    parser.add_argument("--wavmark-root", type=Path, default=projects_root / "wavmark")
    parser.add_argument("--wavmark-checkpoint", default="default")
    args = parser.parse_args()
    args.source_dir = args.source_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.recipe_dir = args.recipe_dir.expanduser().resolve()
    args.vall_e_root = args.vall_e_root.expanduser().resolve()
    args.wavmark_root = args.wavmark_root.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    unknown = args.stages - {"embed", "quality", "robustness"}
    if unknown:
        parser.error(f"Unknown stages: {sorted(unknown)}")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"Backend    : {args.backend}", flush=True)
    print(f"Source     : {args.source_dir}", flush=True)
    print(f"Output     : {args.output_dir}", flush=True)
    print(f"Stages     : {','.join(sorted(args.stages))}", flush=True)
    print(f"Device     : {device}", flush=True)
    if "embed" in args.stages:
        embed_all(args, device)
    if "quality" in args.stages:
        evaluate_quality(args, device)
    if "robustness" in args.stages:
        evaluate_robustness(args, device)


if __name__ == "__main__":
    main()

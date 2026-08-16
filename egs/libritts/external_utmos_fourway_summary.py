#!/usr/bin/env python3
"""Combine UTMOS-only results for the four external-watermark cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


CASE_SPECS = (
    {
        "directory": "audioseal_seedtts_prompt",
        "display_name": "AudioSeal / Seed-TTS",
        "backend": "audioseal",
        "watermark_model": "audioseal_wm_16bits",
        "prompt_mode": "Seed-TTS per-row prompt",
    },
    {
        "directory": "audioseal_libritts_prompt",
        "display_name": "AudioSeal / Fixed",
        "backend": "audioseal",
        "watermark_model": "audioseal_wm_16bits",
        "prompt_mode": "fixed LibriTTS prompt",
    },
    {
        "directory": "wavmark_seedtts_prompt",
        "display_name": "WavMark / Seed-TTS",
        "backend": "wavmark",
        "watermark_model": "WavMark default checkpoint",
        "prompt_mode": "Seed-TTS per-row prompt",
    },
    {
        "directory": "wavmark_libritts_prompt",
        "display_name": "WavMark / Fixed",
        "backend": "wavmark",
        "watermark_model": "WavMark default checkpoint",
        "prompt_mode": "fixed LibriTTS prompt",
    },
)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_case(root: Path, spec: Dict[str, str]) -> Dict[str, Any]:
    case_dir = root / spec["directory"]
    utmos_path = case_dir / "utmos_clean_wm.json"
    embedding_path = case_dir / "embedding_summary.json"
    utmos = read_json(utmos_path)
    embedding = read_json(embedding_path)

    if embedding.get("backend") != spec["backend"]:
        raise ValueError(
            f"{case_dir}: expected backend={spec['backend']}, "
            f"embedding metadata says {embedding.get('backend')!r}"
        )
    expected_pairs = embedding.get("output_pair_count")
    input_pairs = utmos.get("input_pair_count")
    if expected_pairs != input_pairs:
        raise ValueError(
            f"{case_dir}: embedding has {expected_pairs} pairs but UTMOS "
            f"evaluated input contains {input_pairs} pairs"
        )
    evaluated = utmos.get("count")
    skipped = utmos.get("skipped_short_count", 0)
    if evaluated is None or evaluated + skipped != input_pairs:
        raise ValueError(f"{case_dir}: inconsistent UTMOS evaluated/skipped counts")

    source_dir = Path(embedding["source_dir"]).expanduser().resolve()
    source_metadata_path = source_dir / "watermark_summary.json"
    source_metadata = read_json(source_metadata_path)
    if source_metadata.get("watermark_backend") != "voicemark":
        raise ValueError(
            f"{source_metadata_path}: expected VoiceMark-aligned clean carrier"
        )

    return {
        **spec,
        "audio_dir": str(case_dir.resolve()),
        "clean_source_dir": str(source_dir),
        "clean_valle_checkpoint": source_metadata.get("checkpoint"),
        "input_pair_count": input_pairs,
        "evaluated_count": evaluated,
        "skipped_short_count": skipped,
        "avg_utmos_clean": utmos.get("avg_utmos_clean"),
        "avg_utmos_wm": utmos.get("avg_utmos_wm"),
        "avg_utmos_delta_wm_minus_clean": utmos.get(
            "avg_utmos_delta_wm_minus_clean"
        ),
        "utmos_json": str(utmos_path.resolve()),
        "utmos_model": utmos.get("utmos_model"),
    }


def fmt(value: Any, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def make_table(
    cases: List[Dict[str, Any]],
    fixed_prompt_audio: Path,
    fixed_prompt_text: str,
) -> str:
    width = 132
    utmos_model = cases[0]["utmos_model"] or {}
    lines = [
        "External-watermark four-way UTMOS-only summary",
        "=" * width,
        "Case mapping",
        "-" * width,
    ]
    for case in cases:
        lines.append(
            f"{case['display_name']:<24} watermark={case['watermark_model']}; "
            f"prompt={case['prompt_mode']}; clean VALL-E={case['clean_valle_checkpoint']}"
        )
    lines.extend(
        [
            f"Fixed prompt audio: {fixed_prompt_audio}",
            f"Fixed prompt text : {fixed_prompt_text}",
            "Seed-TTS prompt   : prompt_text and prompt_wav from each meta.lst row",
            (
                "UTMOS model      : "
                f"{utmos_model.get('repository')} / {utmos_model.get('entrypoint')}"
            ),
            "",
            "UTMOS predicted naturalness (higher is better)",
            "-" * width,
            (
                f"{'Case':<24} {'Input N':>8} {'Eval N':>8} {'Skip':>6} "
                f"{'UTMOS Clean':>13} {'UTMOS WM':>11} {'Delta WM-Clean':>15}"
            ),
            "-" * width,
        ]
    )
    for case in cases:
        lines.append(
            f"{case['display_name']:<24} "
            f"{case['input_pair_count']:>8d} "
            f"{case['evaluated_count']:>8d} "
            f"{case['skipped_short_count']:>6d} "
            f"{fmt(case['avg_utmos_clean']):>13} "
            f"{fmt(case['avg_utmos_wm']):>11} "
            f"{fmt(case['avg_utmos_delta_wm_minus_clean']):>15}"
        )
    lines.extend(
        [
            "=" * width,
            "Delta WM-Clean = watermarked UTMOS minus its paired clean-carrier UTMOS.",
            "Prompt refers to VALL-E clean-carrier synthesis; AudioSeal/WavMark are post-hoc embedders and do not consume the TTS prompt.",
            "This report contains UTMOS only; WER, SIM, PESQ, STOI, SI-SNR, and watermark detection are not run.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    recipe_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--fixed-prompt-audio",
        type=Path,
        default=recipe_dir / "prompts/8455_210777_000067_000000.wav",
    )
    parser.add_argument(
        "--fixed-prompt-text",
        default="This I read with great attention, while they sat silent.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "utmos_only_summary"
    )
    fixed_prompt_audio = args.fixed_prompt_audio.expanduser().resolve()
    cases = [load_case(root, spec) for spec in CASE_SPECS]
    first_model = cases[0]["utmos_model"]
    if any(case["utmos_model"] != first_model for case in cases[1:]):
        raise ValueError("The four cases were evaluated with different UTMOS models")

    summary = {
        "schema_version": 1,
        "metric_scope": "UTMOS only",
        "case_count": len(cases),
        "fixed_prompt_audio": str(fixed_prompt_audio),
        "fixed_prompt_text": args.fixed_prompt_text,
        "seed_tts_prompt": "prompt_text and prompt_wav from each meta.lst row",
        "prompt_scope": (
            "Prompt configures VALL-E clean-carrier synthesis; the external "
            "watermark embedder does not consume it."
        ),
        "utmos_model": first_model,
        "cases": cases,
    }
    table = make_table(cases, fixed_prompt_audio, args.fixed_prompt_text)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "external_utmos_fourway_summary.json"
    table_path = output_dir / "external_utmos_fourway_summary.txt"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    table_path.write_text(table, encoding="utf-8")
    print(table)
    print(f"Summary JSON : {json_path}")
    print(f"Summary table: {table_path}")


if __name__ == "__main__":
    main()

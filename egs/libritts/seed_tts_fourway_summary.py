#!/usr/bin/env python3
"""Combine four Seed-TTS clean/wm evaluations into one comparison report."""

import argparse
import json
from pathlib import Path


def nested(data, *keys):
    value = data
    for key in keys:
        if value is None or not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def fmt(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def normalize_checkpoint(checkpoint: str, recipe_dir: Path) -> Path:
    path = Path(checkpoint).expanduser()
    if not path.is_absolute():
        path = recipe_dir / path
    return path.resolve()


def load_case(fields: list[str], recipe_dir: Path) -> dict:
    name, backend, prompt_mode, expected_checkpoint, audio_dir, result_json = fields
    audio_dir_path = Path(audio_dir).expanduser().resolve()
    result_path = Path(result_json).expanduser().resolve()
    report = json.loads(result_path.read_text(encoding="utf-8"))

    source_watermark_path = audio_dir_path / "watermark_summary.json"
    if not source_watermark_path.is_file():
        raise FileNotFoundError(
            f"{name}: generation-time watermark summary is missing: "
            f"{source_watermark_path}"
        )
    source_watermark = json.loads(source_watermark_path.read_text(encoding="utf-8"))
    actual_backend = source_watermark.get("watermark_backend")
    if actual_backend != backend:
        raise ValueError(
            f"{name}: expected backend={backend}, metadata says {actual_backend}"
        )

    expected_path = normalize_checkpoint(expected_checkpoint, recipe_dir)
    actual_checkpoint = source_watermark.get("checkpoint")
    if not actual_checkpoint:
        raise ValueError(f"{name}: source metadata has no VALL-E checkpoint")
    actual_path = normalize_checkpoint(actual_checkpoint, recipe_dir)
    if actual_path != expected_path:
        raise ValueError(
            f"{name}: VALL-E checkpoint mismatch: expected {expected_path}, "
            f"metadata says {actual_path}"
        )

    coverage = report.get("coverage")
    if coverage is not None and coverage.get("prompt_mode") != prompt_mode:
        raise ValueError(
            f"{name}: expected prompt_mode={prompt_mode}, "
            f"evaluation says {coverage.get('prompt_mode')}"
        )

    clean = report.get("clean") or {}
    wm = report.get("wm") or {}
    impact = report.get("watermark_impact") or {}
    quality = report.get("audio_quality_clean_vs_wm") or {}
    utmos = report.get("utmos") or {}
    robustness = report.get("watermark_robustness") or {}
    if backend == "traceablespeech" and robustness:
        if not (
            robustness.get("bit_accuracy_unit") == "binary_bit"
            and robustness.get("bit_accuracy_version", 0) >= 2
            and robustness.get("bits_per_message") == 16
        ):
            raise ValueError(
                f"{name}: TraceableSpeech result is legacy hexadecimal-symbol "
                "accuracy; rerun watermark evaluation for 16-bit accuracy."
            )
    identity = nested(robustness, "attacks", "Clean (Identity)") or {}

    clean_wer = nested(clean, "wer", "macro_wer_percent")
    wm_wer = nested(wm, "wer", "macro_wer_percent")
    clean_prompt_sim = nested(clean, "sim", "mean")
    wm_prompt_sim = nested(wm, "sim", "mean")
    clean_wm_sim = nested(impact, "clean_wm_sim", "mean")

    return {
        "name": name,
        "backend": backend,
        "prompt_mode": prompt_mode,
        "expected_valle_checkpoint": str(expected_path),
        "metadata_valle_checkpoint": str(actual_path),
        "audio_dir": str(audio_dir_path),
        "individual_result_json": str(result_path),
        "individual_result_table": str(result_path.with_suffix(".txt")),
        "generation_pair_count": source_watermark.get("count"),
        "wer_sim_pair_count": (
            coverage.get("evaluated_common_rows") if coverage is not None else None
        ),
        "utmos_pair_count": utmos.get("count"),
        "utmos_skipped_short_count": utmos.get("skipped_short_count", 0),
        "metrics": {
            "clean_wer_percent": clean_wer,
            "wm_wer_percent": wm_wer,
            "clean_prompt_sim": clean_prompt_sim,
            "wm_prompt_sim": wm_prompt_sim,
            "clean_wm_sim": clean_wm_sim,
            "prompt_sim_drop": impact.get("prompt_sim_drop"),
            "pesq_wb_clean_vs_wm": quality.get("avg_pesq_wb"),
            "stoi_clean_vs_wm": quality.get("avg_stoi"),
            "si_snr_db_clean_vs_wm": quality.get("avg_si_snr_db"),
            "utmos_clean": utmos.get("avg_utmos_clean"),
            "utmos_wm": utmos.get("avg_utmos_wm"),
            "utmos_delta_wm_minus_clean": utmos.get(
                "avg_utmos_delta_wm_minus_clean"
            ),
            "clean_attack_detect_acc": identity.get("detect_acc"),
            "clean_wm_bit_accuracy": (
                robustness.get("wm_bit_accuracy")
                if robustness.get("wm_bit_accuracy") is not None
                else identity.get("wm_bit_acc")
            ),
            "all_attack_bit_accuracy": robustness.get("attack_overall_accuracy"),
        },
        "modules": report.get("modules"),
    }


def make_table(cases: list[dict]) -> str:
    lines = [
        "Seed-TTS four-way evaluation",
        "=" * 116,
        "Case mapping and validated VALL-E checkpoint",
        "-" * 116,
    ]
    for case in cases:
        lines.append(
            f"{case['name']}: backend={case['backend']}, "
            f"prompt={case['prompt_mode']}, "
            f"VALL-E={case['expected_valle_checkpoint']}"
        )

    lines.extend(
        [
            "",
            "WER and speaker similarity",
            "-" * 116,
            f"{'Case':<24} {'N':>6} {'Clean WER%':>11} {'WM WER%':>10} "
            f"{'Clean-P SIM':>12} {'WM-P SIM':>10} {'Clean-WM SIM':>13}",
            "-" * 116,
        ]
    )
    for case in cases:
        metrics = case["metrics"]
        n = case["wer_sim_pair_count"]
        lines.append(
            f"{case['name']:<24} {str(n) if n is not None else 'N/A':>6} "
            f"{fmt(metrics['clean_wer_percent'], 3):>11} "
            f"{fmt(metrics['wm_wer_percent'], 3):>10} "
            f"{fmt(metrics['clean_prompt_sim'], 6):>12} "
            f"{fmt(metrics['wm_prompt_sim'], 6):>10} "
            f"{fmt(metrics['clean_wm_sim'], 6):>13}"
        )

    lines.extend(
        [
            "",
            "Watermark robustness and clean-vs-watermarked audio quality",
            "-" * 116,
            f"{'Case':<24} {'N':>6} {'PESQ-WB':>9} {'STOI':>9} {'SI-SNR dB':>10} "
            f"{'Clean Det':>10} {'WM Bit Acc':>11} {'All Atk Acc':>11}",
            "-" * 116,
        ]
    )
    for case in cases:
        metrics = case["metrics"]
        n = case["generation_pair_count"]
        lines.append(
            f"{case['name']:<24} {str(n) if n is not None else 'N/A':>6} "
            f"{fmt(metrics['pesq_wb_clean_vs_wm']):>9} "
            f"{fmt(metrics['stoi_clean_vs_wm']):>9} "
            f"{fmt(metrics['si_snr_db_clean_vs_wm']):>10} "
            f"{fmt(metrics['clean_attack_detect_acc']):>10} "
            f"{fmt(metrics['clean_wm_bit_accuracy']):>11} "
            f"{fmt(metrics['all_attack_bit_accuracy']):>11}"
        )
    lines.extend(
        [
            "",
            "UTMOS predicted naturalness (higher is better)",
            "-" * 116,
            f"{'Case':<24} {'N':>6} {'UTMOS Clean':>13} {'UTMOS WM':>11} "
            f"{'Delta WM-Clean':>15}",
            "-" * 116,
        ]
    )
    for case in cases:
        metrics = case["metrics"]
        n = case["utmos_pair_count"]
        lines.append(
            f"{case['name']:<24} {str(n) if n is not None else 'N/A':>6} "
            f"{fmt(metrics['utmos_clean']):>13} "
            f"{fmt(metrics['utmos_wm']):>11} "
            f"{fmt(metrics['utmos_delta_wm_minus_clean']):>15}"
        )
    lines.extend(
        [
            "=" * 116,
            "Clean-P/WM-P SIM use the prompt selected for each case; fixed cases use the fixed LibriTTS prompt.",
            "Each individual result directory contains the complete DSP/Encodec/DAC/SNAC attack table.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recipe-dir", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        nargs=6,
        metavar=(
            "NAME",
            "BACKEND",
            "PROMPT_MODE",
            "EXPECTED_PT",
            "AUDIO_DIR",
            "RESULT_JSON",
        ),
        required=True,
    )
    args = parser.parse_args()

    recipe_dir = args.recipe_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [load_case(fields, recipe_dir) for fields in args.case]
    if len(cases) != 4:
        raise ValueError(f"Exactly four cases are required; got {len(cases)}")

    summary = {
        "case_count": len(cases),
        "checkpoint_validation": "passed",
        "cases": cases,
    }
    table = make_table(cases)
    (output_dir / "fourway_metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "fourway_metrics_summary.txt").write_text(
        table, encoding="utf-8"
    )
    print(table)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare and summarize clean/watermarked Seed-TTS metric evaluation."""

import argparse
import json
from pathlib import Path


def load_manifest(path: Path):
    rows = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split("|")]
        if len(fields) not in (4, 5):
            raise ValueError(
                f"{path}:{line_number}: expected 4 or 5 pipe-separated fields, "
                f"got {len(fields)}"
            )
        utterance_id, prompt_text, prompt_wav, target_text = fields[:4]
        prompt_path = Path(prompt_wav).expanduser()
        if not prompt_path.is_absolute():
            prompt_path = (path.parent / prompt_path).resolve()
        rows.append(
            {
                "utterance_id": Path(utterance_id).stem,
                "prompt_text": prompt_text,
                "prompt_wav": prompt_path,
                "target_text": target_text,
            }
        )
    return rows


def prepare(args):
    manifest = args.manifest.expanduser().resolve()
    audio_dir = args.audio_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    prompt_override = (
        args.prompt_wav.expanduser().resolve() if args.prompt_wav is not None else None
    )
    if prompt_override is not None and not prompt_override.is_file():
        raise FileNotFoundError(f"Fixed prompt WAV not found: {prompt_override}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest)
    manifest_ids = {row["utterance_id"] for row in rows}
    clean_ids = {
        path.name[: -len("_clean.wav")]
        for path in audio_dir.glob("*_clean.wav")
    }
    wm_ids = {
        path.name[: -len("_wm.wav")]
        for path in audio_dir.glob("*_wm.wav")
    }
    paired_ids = clean_ids & wm_ids
    common = []
    missing_clean = []
    missing_wm = []
    missing_prompt = []
    for row in rows:
        utterance_id = row["utterance_id"]
        clean = audio_dir / f"{utterance_id}_clean.wav"
        wm = audio_dir / f"{utterance_id}_wm.wav"
        prompt_wav = prompt_override or row["prompt_wav"]
        if not clean.is_file():
            missing_clean.append(utterance_id)
        if not wm.is_file():
            missing_wm.append(utterance_id)
        if not prompt_wav.is_file():
            missing_prompt.append(utterance_id)
        if clean.is_file() and wm.is_file() and prompt_wav.is_file():
            eval_row = dict(row)
            eval_row["prompt_wav"] = prompt_wav
            common.append((eval_row, clean, wm))

    if not common:
        raise RuntimeError("No common clean/wm/prompt triples were found.")

    for variant, index in (("clean", 1), ("wm", 2)):
        pair_path = output_dir / f"{variant}_pairs.lst"
        with pair_path.open("w", encoding="utf-8") as stream:
            for row, clean, wm in common:
                generated = (clean, wm)[index - 1]
                # This is the official Seed-TTS SIM/metadata layout:
                # generated_wav|prompt_wav|target_text
                stream.write(
                    f"{generated}|{row['prompt_wav']}|{row['target_text']}\n"
                )

    # Watermark speaker-transparency pairing. The official SIM runner only
    # consumes the first two fields, so this evaluates clean <-> watermarked
    # with exactly the same WavLM-large cosine-similarity implementation.
    clean_wm_path = output_dir / "clean_wm_pairs.lst"
    with clean_wm_path.open("w", encoding="utf-8") as stream:
        for row, clean, wm in common:
            stream.write(f"{clean}|{wm}|{row['target_text']}\n")

    coverage = {
        "manifest": str(manifest),
        "audio_dir": str(audio_dir),
        "manifest_rows": len(rows),
        "clean_wav_count": len(clean_ids),
        "wm_wav_count": len(wm_ids),
        "paired_wav_count": len(paired_ids),
        "evaluated_common_rows": len(common),
        "prompt_mode": "fixed" if prompt_override is not None else "manifest",
        "fixed_prompt_wav": str(prompt_override) if prompt_override is not None else None,
        "missing_clean": missing_clean,
        "missing_wm": missing_wm,
        "missing_prompt": missing_prompt,
        "clean_without_wm": sorted(clean_ids - wm_ids),
        "wm_without_clean": sorted(wm_ids - clean_ids),
        "paired_ids_missing_from_manifest": sorted(paired_ids - manifest_ids),
        "policy": "Only IDs having clean, wm, and prompt WAVs are evaluated in both variants.",
    }
    (output_dir / "coverage.json").write_text(
        json.dumps(coverage, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(coverage, indent=2, ensure_ascii=False))


def read_sim(path: Path):
    scores = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("avg score:"):
            continue
        fields = line.rsplit("\t", 1)
        if len(fields) == 2:
            scores.append(float(fields[1]))
    if not scores:
        raise RuntimeError(f"No SIM scores found in {path}")
    mean = sum(scores) / len(scores)
    variance = sum((value - mean) ** 2 for value in scores) / len(scores)
    return {"count": len(scores), "mean": mean, "variance": variance}


def summarize(args):
    output_dir = args.output_dir.expanduser().resolve()
    coverage = None
    variants = {}
    clean_wm_sim = None
    prompt_sim_drop = None
    if not args.skip_wer_sim:
        coverage = json.loads(
            (output_dir / "coverage.json").read_text(encoding="utf-8")
        )
        for variant in ("clean", "wm"):
            wer = json.loads(
                (output_dir / variant / "wer_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            sim = read_sim(output_dir / variant / "sim_raw.tsv")
            variants[variant] = {"wer": wer, "sim": sim}

        clean_wm_sim = read_sim(output_dir / "clean_wm" / "sim_raw.tsv")
        prompt_sim_drop = (
            variants["clean"]["sim"]["mean"]
            - variants["wm"]["sim"]["mean"]
        )

    audio_quality = None
    if args.audio_quality_json is not None:
        audio_quality_path = args.audio_quality_json.expanduser().resolve()
        audio_quality = json.loads(audio_quality_path.read_text(encoding="utf-8"))

    utmos = None
    if args.utmos_json is not None:
        utmos_path = args.utmos_json.expanduser().resolve()
        utmos = json.loads(utmos_path.read_text(encoding="utf-8"))
    elif audio_quality is not None and audio_quality.get("avg_utmos_wm") is not None:
        # External-watermark evaluation stores UTMOS in its quality JSON.
        utmos = audio_quality

    watermark = None
    watermark_table = None
    if args.watermark_summary_json is not None:
        watermark_summary_path = args.watermark_summary_json.expanduser().resolve()
        raw_watermark = json.loads(watermark_summary_path.read_text(encoding="utf-8"))
        # Per-sample details can be tens of MB.  Keep the complete source file
        # beside this report and embed all aggregate fields in the combined JSON.
        watermark = {
            key: value for key, value in raw_watermark.items() if key != "details"
        }
        watermark["full_summary_file"] = str(watermark_summary_path)
        watermark["detail_count"] = len(raw_watermark.get("details", []))
    if args.watermark_table is not None:
        watermark_table_path = args.watermark_table.expanduser().resolve()
        watermark_table = watermark_table_path.read_text(encoding="utf-8").strip()

    summary = {
        "metric_protocol": {
            "wer": "Seed-TTS official English protocol: Whisper-large-v3; macro utterance WER.",
            "prompt_sim": "Seed-TTS official protocol: WavLM-large cosine similarity of each generated variant to its prompt.",
            "clean_wm_sim": "WavLM-large cosine similarity between paired clean and watermarked outputs.",
            "prompt_sim_drop": "Clean Prompt SIM minus Watermarked Prompt SIM; lower is better.",
            "utmos": "SpeechMOS UTMOS22 strong predicted naturalness MOS for clean and watermarked audio; higher is better.",
        },
        "modules": {
            "wer_sim": not args.skip_wer_sim,
            "watermark_quality": audio_quality is not None or watermark is not None,
            "utmos": utmos is not None,
        },
        "coverage": coverage,
        "clean": variants.get("clean"),
        "wm": variants.get("wm"),
        "watermark_impact": (
            {
                "clean_wm_sim": clean_wm_sim,
                "prompt_sim_drop": prompt_sim_drop,
            }
            if not args.skip_wer_sim
            else None
        ),
        "audio_quality_clean_vs_wm": (
            {key: value for key, value in audio_quality.items() if key != "details"}
            if audio_quality is not None
            else None
        ),
        "utmos": (
            {key: value for key, value in utmos.items() if key != "details"}
            if utmos is not None
            else None
        ),
        "watermark_robustness": watermark,
    }
    (output_dir / "seed_tts_clean_wm_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    audio_dir = (
        args.audio_dir.expanduser().resolve()
        if args.audio_dir is not None
        else Path(coverage["audio_dir"] if coverage else output_dir.parent)
    )
    lines = [f"Combined evaluation: {audio_dir.name}", "=" * 72]
    if not args.skip_wer_sim:
        lines.extend(
            [
                f"Common evaluated samples: {coverage['evaluated_common_rows']} / {coverage['manifest_rows']}",
                "",
                f"{'Variant':<12} {'WER (%)':>12} {'Prompt SIM':>12} {'WER N':>10} {'SIM N':>10}",
                "-" * 72,
            ]
        )
        for variant in ("clean", "wm"):
            wer = variants[variant]["wer"]
            sim = variants[variant]["sim"]
            lines.append(
                f"{variant:<12} {wer['macro_wer_percent']:>12.3f} "
                f"{sim['mean']:>12.6f} {wer['count']:>10d} {sim['count']:>10d}"
            )
        lines.extend(
            [
                "-" * 72,
                f"Clean-WM SIM:    {clean_wm_sim['mean']:.6f}  (N={clean_wm_sim['count']})",
                f"Prompt SIM Drop: {prompt_sim_drop:.6f}  (Clean Prompt SIM - WM Prompt SIM)",
                "=" * 72,
                "WER: Whisper-large-v3, lower is better.",
                "Prompt SIM: generated audio vs prompt, higher is better.",
                "Clean-WM SIM: clean vs watermarked output, higher is better.",
                "Prompt SIM Drop: watermark-induced prompt-SIM decrease, lower is better.",
            ]
        )
    else:
        lines.append("WER/SIM module: skipped (RUN_WER_SIM=false)")

    if audio_quality is not None:
        def metric(value, digits=4, suffix=""):
            return "N/A" if value is None else f"{value:.{digits}f}{suffix}"

        lines.extend(
            [
                "",
                "Audio quality / watermark transparency (clean vs wm)",
                "-" * 72,
                f"Pairs:    {audio_quality['count']}",
                f"PESQ-WB:  {metric(audio_quality.get('avg_pesq_wb'))}",
                f"STOI:     {metric(audio_quality.get('avg_stoi'))}",
                f"SI-SNR:   {metric(audio_quality.get('avg_si_snr_db'), suffix=' dB')}",
                f"ViSQOL:   {metric(audio_quality.get('avg_visqol_moslqo'))}",
            ]
        )

    if utmos is not None:
        def utmos_metric(value):
            return "N/A" if value is None else f"{value:.4f}"

        lines.extend(
            [
                "",
                "UTMOS predicted naturalness (higher is better)",
                "-" * 72,
                f"Input pairs: {utmos.get('input_pair_count', utmos['count'])}",
                f"Evaluated:   {utmos['count']}",
                f"Skipped:     {utmos.get('skipped_short_count', 0)} short pairs",
                f"UTMOS clean: {utmos_metric(utmos.get('avg_utmos_clean'))}",
                f"UTMOS wm:    {utmos_metric(utmos.get('avg_utmos_wm'))}",
                "UTMOS delta: "
                f"{utmos_metric(utmos.get('avg_utmos_delta_wm_minus_clean'))} (wm - clean)",
            ]
        )

    if watermark_table:
        lines.extend(["", "Watermark detection and attack robustness", watermark_table])
    table = "\n".join(lines) + "\n"
    (output_dir / "seed_tts_clean_wm_metrics.txt").write_text(table, encoding="utf-8")
    print(table)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare")
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--audio-dir", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument(
        "--prompt-wav",
        type=Path,
        default=None,
        help="Use one fixed prompt WAV for every Prompt-SIM pair.",
    )
    prep.set_defaults(func=prepare)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--output-dir", type=Path, required=True)
    summary.add_argument("--audio-dir", type=Path, default=None)
    summary.add_argument("--skip-wer-sim", action="store_true")
    summary.add_argument("--audio-quality-json", type=Path, default=None)
    summary.add_argument("--utmos-json", type=Path, default=None)
    summary.add_argument("--watermark-summary-json", type=Path, default=None)
    summary.add_argument("--watermark-table", type=Path, default=None)
    summary.set_defaults(func=summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

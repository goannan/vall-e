#!/usr/bin/env python3
"""Benchmark watermark embedding/extraction runtime and combine all backends."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(device: torch.device, function):
    synchronize(device)
    start = time.perf_counter()
    result = function()
    synchronize(device)
    return result, time.perf_counter() - start


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def benchmark_external(args) -> Dict[str, Any]:
    device = torch.device(args.device)
    carrier = json.loads(args.carrier_timing_json.read_text(encoding="utf-8"))
    carrier_rows = carrier.get("details") or []
    if not carrier_rows:
        raise RuntimeError(f"No carrier rows in {args.carrier_timing_json}")

    external = load_module(
        "runtime_external_watermark_eval",
        args.recipe_dir / "external_watermark_eval.py",
    )
    (watermarker, setup_seconds) = timed_call(
        device,
        lambda: external.ExternalWatermarker(
            backend=args.backend,
            device=device,
            wavmark_root=args.wavmark_root,
            need_generator=True,
            need_detector=True,
            wavmark_checkpoint=args.wavmark_checkpoint,
        ),
    )

    details: List[Dict[str, Any]] = []
    synchronize(device)
    loop_start = time.perf_counter()
    for index, carrier_row in enumerate(carrier_rows, start=1):
        utterance_id = carrier_row["utterance_id"]
        clean_path = args.clean_dir / carrier_row["clean"]
        if not clean_path.is_file():
            raise FileNotFoundError(f"Missing clean carrier: {clean_path}")
        clean = external.load_mono_16k(clean_path, device)
        audio_seconds = clean.shape[-1] / external.SAMPLE_RATE

        payload, payload_seconds = timed_call(
            device, lambda: external.payload_for(utterance_id)
        )
        try:
            (watermarked, _embed_info), backend_embed_seconds = timed_call(
                device, lambda: watermarker.embed(clean, payload)
            )
            detect_input = watermarked
            detection_padding_samples = 0
            if args.backend == "wavmark" and detect_input.shape[-1] < external.SAMPLE_RATE:
                detection_padding_samples = external.SAMPLE_RATE - detect_input.shape[-1]
                detect_input = torch.nn.functional.pad(
                    detect_input, (0, detection_padding_samples)
                )
            (probability, bit_accuracy, _), extract_seconds = timed_call(
                device, lambda: watermarker.detect(detect_input, payload)
            )
        except Exception as exc:
            raise RuntimeError(
                f"{args.backend} runtime benchmark failed for {clean_path}"
            ) from exc

        details.append(
            {
                "index": index - 1,
                "utterance_id": utterance_id,
                "clean": clean_path.name,
                "audio_duration_seconds": audio_seconds,
                "payload_generation_seconds": payload_seconds,
                "backend_embedding_seconds": backend_embed_seconds,
                "watermark_embedding_seconds": (
                    payload_seconds + backend_embed_seconds
                ),
                "watermark_extract_seconds": extract_seconds,
                "detection_padding_samples": detection_padding_samples,
                "detection_probability": float(probability),
                "bit_accuracy": float(bit_accuracy),
            }
        )
        if index == 1 or index % 25 == 0 or index == len(carrier_rows):
            print(
                f"{args.backend}: {index}/{len(carrier_rows)} | "
                f"embed={backend_embed_seconds:.4f}s extract={extract_seconds:.4f}s",
                flush=True,
            )

    synchronize(device)
    loop_wall_seconds = time.perf_counter() - loop_start

    def total(key: str) -> float:
        return float(sum(row[key] for row in details))

    audio_seconds = total("audio_duration_seconds")
    payload_seconds = total("payload_generation_seconds")
    backend_embed_seconds = total("backend_embedding_seconds")
    embedding_seconds = total("watermark_embedding_seconds")
    extraction_seconds = total("watermark_extract_seconds")
    summary = {
        "schema_version": 2,
        "benchmark_kind": "posthoc_watermark",
        "timing_scope": (
            "CUDA-synchronized backend operations; model loading, waveform loading, "
            "and waveform writes are excluded."
        ),
        "watermark_backend": args.backend,
        "hardware": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "CPU"
            ),
            "torch_version": str(torch.__version__),
        },
        "configuration": {
            "wavmark_checkpoint": (
                args.wavmark_checkpoint if args.backend == "wavmark" else None
            ),
            "bits_per_message": external.BITS_PER_MESSAGE,
        },
        "clean_dir": str(args.clean_dir.resolve()),
        "carrier_timing_json": str(args.carrier_timing_json.resolve()),
        "count": len(details),
        "setup_seconds_excluded_from_totals": {
            "watermark_models_load_seconds": setup_seconds
        },
        "loop_wall_seconds_including_io_and_cpu_overhead": loop_wall_seconds,
        "totals": {
            "audio_duration_seconds": audio_seconds,
            "payload_generation_seconds": payload_seconds,
            "backend_embedding_seconds": backend_embed_seconds,
            "watermark_embedding_seconds": embedding_seconds,
            "watermark_incremental_over_clean_seconds": embedding_seconds,
            "watermark_extract_seconds": extraction_seconds,
            "watermark_extract_count": len(details),
            "watermark_extract_skipped_count": 0,
            "watermark_embedding_rtf": (
                embedding_seconds / audio_seconds if audio_seconds > 0 else None
            ),
            "watermark_extract_rtf": (
                extraction_seconds / audio_seconds if audio_seconds > 0 else None
            ),
        },
        "details": details,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Runtime JSON: {args.output_json}", flush=True)
    return summary


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_report(
    report: Dict[str, Any], path: Path, benchmark_kind: str, backend: str
) -> None:
    if report.get("schema_version", 0) < 2:
        raise ValueError(
            f"Outdated runtime report: {path}. Rerun the SLURM script with "
            "FORCE=true."
        )
    if report.get("benchmark_kind") != benchmark_kind:
        raise ValueError(
            f"Unexpected benchmark_kind in {path}: "
            f"{report.get('benchmark_kind')!r}"
        )
    if report.get("watermark_backend") != backend:
        raise ValueError(
            f"Unexpected watermark backend in {path}: "
            f"{report.get('watermark_backend')!r}"
        )


def integrated_rows(report: Dict[str, Any], display_name: str) -> List[Dict[str, Any]]:
    totals = report["totals"]
    count = report["count"]
    setup = report.get("setup_seconds_excluded_from_totals") or {}
    setup_total = float(sum(setup.values()))
    clean_seconds = totals["clean_synthesis_seconds"]
    embedding_seconds = totals["watermark_embedding_path_seconds"]
    incremental_seconds = totals["watermark_incremental_over_clean_seconds"]
    extraction_seconds = totals["watermark_extract_seconds"]
    extract_count = totals.get("watermark_extract_count", count)
    return [
        {
            "method": f"VALL-E clean ({display_name} codec)",
            "backend": report["watermark_backend"],
            "kind": "clean_baseline",
            "requested_count": report.get("requested_samples", count),
            "count": count,
            "synthesis_skipped_count": report.get("skipped_count", 0),
            "audio_duration_seconds": totals["audio_duration_seconds"],
            "clean_synthesis_seconds": clean_seconds,
            "watermark_embedding_seconds": None,
            "watermark_incremental_seconds": None,
            "watermarked_pipeline_seconds": None,
            "watermark_extract_seconds": None,
            "watermark_extract_count": None,
            "embedding_rtf": None,
            "extract_rtf": None,
            "setup_seconds_excluded": setup_total,
        },
        {
            "method": f"{display_name} integrated",
            "backend": report["watermark_backend"],
            "kind": "integrated_watermark",
            "requested_count": report.get("requested_samples", count),
            "count": count,
            "synthesis_skipped_count": report.get("skipped_count", 0),
            "audio_duration_seconds": totals["audio_duration_seconds"],
            "clean_synthesis_seconds": clean_seconds,
            "watermark_embedding_seconds": embedding_seconds,
            "watermark_incremental_seconds": incremental_seconds,
            "watermarked_pipeline_seconds": totals[
                "watermarked_synthesis_seconds"
            ],
            "watermark_extract_seconds": extraction_seconds,
            "watermark_extract_count": extract_count,
            "embedding_rtf": totals["watermark_embedding_path_rtf"],
            "incremental_rtf": totals[
                "watermark_incremental_over_clean_rtf"
            ],
            "extract_rtf": totals["watermark_extract_rtf"],
            "setup_seconds_excluded": setup_total,
        },
    ]


def posthoc_row(
    report: Dict[str, Any], carrier: Dict[str, Any], display_name: str
) -> Dict[str, Any]:
    totals = report["totals"]
    carrier_totals = carrier["totals"]
    if report["count"] != carrier["count"]:
        raise ValueError(
            f"{display_name}: post-hoc count {report['count']} does not match "
            f"carrier count {carrier['count']}"
        )
    report_ids = [row["utterance_id"] for row in report.get("details", [])]
    carrier_ids = [row["utterance_id"] for row in carrier.get("details", [])]
    if report_ids != carrier_ids:
        raise ValueError(
            f"{display_name}: post-hoc utterance IDs do not exactly match the "
            "clean carrier report"
        )
    clean_seconds = carrier_totals["clean_synthesis_seconds"]
    embedding_seconds = totals["watermark_embedding_seconds"]
    setup = report.get("setup_seconds_excluded_from_totals") or {}
    return {
        "method": f"{display_name} post-hoc on VoiceMark clean",
        "backend": report["watermark_backend"],
        "kind": "posthoc_watermark",
        "requested_count": carrier.get("requested_samples", carrier["count"]),
        "count": report["count"],
        "synthesis_skipped_count": carrier.get("skipped_count", 0),
        "audio_duration_seconds": totals["audio_duration_seconds"],
        "clean_synthesis_seconds": clean_seconds,
        "watermark_embedding_seconds": embedding_seconds,
        "watermark_incremental_seconds": embedding_seconds,
        "watermarked_pipeline_seconds": clean_seconds + embedding_seconds,
        "watermark_extract_seconds": totals["watermark_extract_seconds"],
        "watermark_extract_count": totals.get(
            "watermark_extract_count", report["count"]
        ),
        "embedding_rtf": totals["watermark_embedding_rtf"],
        "incremental_rtf": totals["watermark_embedding_rtf"],
        "extract_rtf": totals["watermark_extract_rtf"],
        "setup_seconds_excluded": float(sum(setup.values())),
    }


def format_number(value, digits=3, width=12) -> str:
    if value is None:
        return f"{'N/A':>{width}}"
    return f"{value:>{width}.{digits}f}"


def make_table(rows: List[Dict[str, Any]]) -> str:
    width = 190
    lines = [
        "Watermark synthesis / embedding / extraction runtime benchmark",
        "=" * width,
        (
            f"{'Method':<43} {'Req.N':>6} {'Synth.N':>7} {'Skip':>5} "
            f"{'Audio s':>11} {'Clean total s':>14} "
            f"{'WM embed/path s':>16} {'Extra vs clean s':>16} "
            f"{'WM pipeline s':>14} {'Extract s':>12} {'Ext.N':>7}"
        ),
        "-" * width,
    ]
    for row in rows:
        lines.append(
            f"{row['method']:<43} {row['requested_count']:>6d} "
            f"{row['count']:>7d} {row['synthesis_skipped_count']:>5d} "
            f"{format_number(row['audio_duration_seconds'], width=11)} "
            f"{format_number(row['clean_synthesis_seconds'], width=14)} "
            f"{format_number(row['watermark_embedding_seconds'], width=16)} "
            f"{format_number(row['watermark_incremental_seconds'], width=16)} "
            f"{format_number(row['watermarked_pipeline_seconds'], width=14)} "
            f"{format_number(row['watermark_extract_seconds'], width=12)} "
            f"{str(row['watermark_extract_count']) if row['watermark_extract_count'] is not None else 'N/A':>7}"
        )

    lines.extend(
        [
            "=" * width,
            "Normalized watermark runtime (model setup excluded)",
            "-" * width,
            (
                f"{'Method':<43} {'Path ms/utt':>14} {'Extra ms/utt':>14} "
                f"{'Extract ms/utt':>16} {'Path RTF':>12} {'Extra RTF':>12} "
                f"{'Extract RTF':>12} {'Setup s excluded':>17}"
            ),
            "-" * width,
        ]
    )
    for row in rows:
        embed = row["watermark_embedding_seconds"]
        incremental = row["watermark_incremental_seconds"]
        extract = row["watermark_extract_seconds"]
        extract_count = row["watermark_extract_count"]
        embed_ms = embed * 1000.0 / row["count"] if embed is not None else None
        incremental_ms = (
            incremental * 1000.0 / row["count"]
            if incremental is not None
            else None
        )
        extract_ms = (
            extract * 1000.0 / extract_count
            if extract is not None and extract_count
            else None
        )
        lines.append(
            f"{row['method']:<43} "
            f"{format_number(embed_ms, width=14)} "
            f"{format_number(incremental_ms, width=14)} "
            f"{format_number(extract_ms, width=16)} "
            f"{format_number(row['embedding_rtf'], digits=6, width=12)} "
            f"{format_number(row.get('incremental_rtf'), digits=6, width=12)} "
            f"{format_number(row['extract_rtf'], digits=6, width=12)} "
            f"{format_number(row['setup_seconds_excluded'], width=17)}"
        )
    lines.extend(
        [
            "=" * width,
            "Timing uses device synchronization. Model loading, generated audio writes, quality metrics, attacks, WER, and SIM are excluded.",
            "The integrated fixed prompt is loaded/encoded once and included in both clean and WM pipeline totals; post-hoc carrier loading is excluded.",
            "Integrated WM embed/path includes message generation plus the complete watermarked codec decoder path.",
            "Extra vs clean is the integrated WM path minus the clean decoder; for post-hoc methods it equals embedding time.",
            "Small negative Extra vs clean values can occur from runtime variance when clean and WM decoder paths have nearly identical cost.",
            "WM pipeline is an alternative end-to-end path: prompt preparation + VALL-E token generation + WM path; it does not add clean decoding.",
            "AudioSeal/WavMark use the exact VoiceMark-run clean carriers; their WM pipeline equals reused clean synthesis + post-hoc embedding.",
            "All operation totals cover Synth.N successfully generated utterances; Skip records VALL-E early-EOS rows with no audio to benchmark.",
            "Ext.N is the number of utterances whose detector inference actually ran; Extract s includes every invocation, including short-audio eligibility checks.",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize(args) -> Dict[str, Any]:
    voicemark = read_json(args.voicemark_json)
    traceablespeech = read_json(args.traceablespeech_json)
    audioseal = read_json(args.audioseal_json)
    wavmark = read_json(args.wavmark_json)
    validate_report(
        voicemark,
        args.voicemark_json,
        "integrated_valle_watermark",
        "voicemark",
    )
    validate_report(
        traceablespeech,
        args.traceablespeech_json,
        "integrated_valle_watermark",
        "traceablespeech",
    )
    validate_report(audioseal, args.audioseal_json, "posthoc_watermark", "audioseal")
    validate_report(wavmark, args.wavmark_json, "posthoc_watermark", "wavmark")
    if voicemark["requested_samples"] != traceablespeech["requested_samples"]:
        raise ValueError(
            "VoiceMark and TraceableSpeech reports requested different sample counts"
        )
    rows = []
    rows.extend(integrated_rows(voicemark, "VoiceMark"))
    rows.extend(integrated_rows(traceablespeech, "TraceableSpeech"))
    rows.append(posthoc_row(audioseal, voicemark, "AudioSeal"))
    rows.append(posthoc_row(wavmark, voicemark, "WavMark"))
    summary = {
        "schema_version": 2,
        "timing_scope": (
            "CUDA-synchronized operations; setup, generated output writes, metrics, "
            "and attacks excluded. Integrated fixed-prompt preparation is included "
            "once; post-hoc clean-carrier loading is excluded."
        ),
        "sources": {
            "voicemark": str(args.voicemark_json.resolve()),
            "traceablespeech": str(args.traceablespeech_json.resolve()),
            "audioseal": str(args.audioseal_json.resolve()),
            "wavmark": str(args.wavmark_json.resolve()),
        },
        "rows": rows,
    }
    table = make_table(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "watermark_runtime_summary.json"
    table_path = args.output_dir / "watermark_runtime_summary.txt"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    table_path.write_text(table, encoding="utf-8")
    print(table)
    print(f"Summary JSON : {json_path}")
    print(f"Summary table: {table_path}")
    return summary


def main() -> None:
    recipe_dir = Path(__file__).resolve().parent
    projects_root = recipe_dir.parents[2]
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    external = subparsers.add_parser("external")
    external.add_argument("--backend", choices=("audioseal", "wavmark"), required=True)
    external.add_argument("--clean-dir", type=Path, required=True)
    external.add_argument("--carrier-timing-json", type=Path, required=True)
    external.add_argument("--output-json", type=Path, required=True)
    external.add_argument("--recipe-dir", type=Path, default=recipe_dir)
    external.add_argument("--wavmark-root", type=Path, default=projects_root / "wavmark")
    external.add_argument("--wavmark-checkpoint", default="default")
    external.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    combine = subparsers.add_parser("summarize")
    combine.add_argument("--voicemark-json", type=Path, required=True)
    combine.add_argument("--traceablespeech-json", type=Path, required=True)
    combine.add_argument("--audioseal-json", type=Path, required=True)
    combine.add_argument("--wavmark-json", type=Path, required=True)
    combine.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "external":
        args.clean_dir = args.clean_dir.expanduser().resolve()
        args.carrier_timing_json = args.carrier_timing_json.expanduser().resolve()
        args.output_json = args.output_json.expanduser().resolve()
        args.recipe_dir = args.recipe_dir.expanduser().resolve()
        args.wavmark_root = args.wavmark_root.expanduser().resolve()
        benchmark_external(args)
    else:
        args.voicemark_json = args.voicemark_json.expanduser().resolve()
        args.traceablespeech_json = args.traceablespeech_json.expanduser().resolve()
        args.audioseal_json = args.audioseal_json.expanduser().resolve()
        args.wavmark_json = args.wavmark_json.expanduser().resolve()
        args.output_dir = args.output_dir.expanduser().resolve()
        summarize(args)


if __name__ == "__main__":
    main()

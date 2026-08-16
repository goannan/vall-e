#!/usr/bin/env python3
"""Evaluate watermark detection/robustness on existing clean/wm WAV pairs.

The generation pipeline stores the target watermark message in
``<utterance_id>_wm.json``.  This evaluator reuses those targets, so it can run
the same DSP and neural-codec attack table as ``bin/infer.py`` without
resynthesizing speech.
"""

import argparse
import json
from pathlib import Path

import torch

from valle.bin.attacks import build_voicemark_valid_attacks
from valle.bin.infer import (
    evaluate_voicemark_attacks,
    make_voicemark_table,
    summarize_attacks,
    traceable_bits_to_symbols,
    traceable_symbols_to_bits,
)
from valle.data import AudioTokenizer


def detect_backend(audio_dir: Path, records: list[dict]) -> tuple[str, dict]:
    source_summary_path = audio_dir / "watermark_summary.json"
    source_summary = {}
    if source_summary_path.is_file():
        source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
        backend = source_summary.get("watermark_backend")
        if backend in {"traceablespeech", "voicemark"}:
            return backend, source_summary

    if records[0].get("watermark_symbols") is not None:
        return "traceablespeech", source_summary
    values = torch.as_tensor(records[0]["watermark_bits"]).reshape(-1)
    # Legacy TraceableSpeech JSON stores four hexadecimal symbols. New JSON is
    # disambiguated by ``watermark_symbols``; VoiceMark stores 16 binary bits.
    backend = "traceablespeech" if values.numel() == 4 else "voicemark"
    return backend, source_summary


def load_records(audio_dir: Path) -> list[dict]:
    records = []
    missing_metadata = []
    missing_wm = []
    for clean_path in sorted(audio_dir.glob("*_clean.wav")):
        utterance_id = clean_path.name[: -len("_clean.wav")]
        wm_path = audio_dir / f"{utterance_id}_wm.wav"
        meta_path = audio_dir / f"{utterance_id}_wm.json"
        if not wm_path.is_file():
            missing_wm.append(wm_path.name)
            continue
        if not meta_path.is_file():
            missing_metadata.append(meta_path.name)
            continue
        record = json.loads(meta_path.read_text(encoding="utf-8"))
        if "watermark_bits" not in record and "watermark_symbols" not in record:
            missing_metadata.append(
                f"{meta_path.name}:watermark_bits/watermark_symbols"
            )
            continue
        record["clean"] = clean_path.name
        record["watermarked"] = wm_path.name
        record.setdefault("utterance_id", utterance_id)
        record.setdefault("index", len(records))
        # Never mutate the generation-time attack dictionary in the source JSON.
        record["attacks"] = {}
        records.append(record)

    if missing_wm:
        raise RuntimeError(
            f"Found clean WAVs without wm WAVs ({len(missing_wm)}): "
            + ", ".join(missing_wm[:10])
        )
    if missing_metadata:
        raise RuntimeError(
            "Watermark bit metadata is required to calculate bit accuracy; "
            f"missing/invalid entries ({len(missing_metadata)}): "
            + ", ".join(missing_metadata[:10])
        )
    if not records:
        raise RuntimeError(f"No *_clean.wav/*_wm.wav pairs found in {audio_dir}")
    return records


def normalize_messages(records: list[dict], backend: str) -> None:
    """Upgrade legacy TS metadata to explicit symbols plus 16 binary bits."""

    for record in records:
        if backend == "traceablespeech":
            if record.get("watermark_symbols") is not None:
                symbols = torch.as_tensor(record["watermark_symbols"]).long()
            else:
                stored = torch.as_tensor(record["watermark_bits"]).long()
                symbols = (
                    stored
                    if stored.shape[-1] == 4
                    else traceable_bits_to_symbols(stored)
                )
            record["watermark_symbols"] = symbols.int().tolist()
            record["watermark_bits"] = (
                traceable_symbols_to_bits(symbols).int().tolist()
            )
        record["bit_accuracy_unit"] = "binary_bit"


def load_quality(path: Path | None) -> tuple[dict, list[dict]]:
    if path is None:
        return {}, []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in data.get("details", []):
        pesq = item.get("pesq_wb")
        stoi = item.get("stoi")
        si_snr = item.get("si_snr_db")
        rows.append(
            {
                # VoiceMark valid.py records failed utterance-level quality
                # metrics as zero; preserve that table policy here.
                "pesq_wb": 0.0 if pesq is None else pesq,
                "stoi": 0.0 if stoi is None else stoi,
                "si_snr": 0.0 if si_snr is None else si_snr,
            }
        )
    return data, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all existing Seed-TTS clean/wm pairs with the valid.py attack suite."
    )
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--watermark-backend",
        choices=["auto", "traceablespeech", "voicemark"],
        default="auto",
    )
    parser.add_argument("--audio-quality-json", type=Path, default=None)
    parser.add_argument("--skip-codecs", action="store_true")
    parser.add_argument(
        "--ts-checkpoint",
        default="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000",
    )
    parser.add_argument(
        "--ts-config",
        default="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json",
    )
    parser.add_argument(
        "--voicemark-root",
        default="/home/wu25/mrnas04home/projects/VoiceMark",
    )
    parser.add_argument(
        "--voicemark-config",
        default="STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json",
    )
    parser.add_argument(
        "--voicemark-st-checkpoint",
        default="STmodels/pretrained_model/SpeechTokenizer.pt",
    )
    parser.add_argument(
        "--voicemark-checkpoint",
        default="train/Log/spt_base/20260601-123358/WatermarkTrainer_final_00150000.pt",
    )
    args = parser.parse_args()

    audio_dir = args.audio_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    quality_path = (
        args.audio_quality_json.expanduser().resolve()
        if args.audio_quality_json is not None
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(audio_dir)
    detected_backend, source_summary = detect_backend(audio_dir, records)
    backend = detected_backend if args.watermark_backend == "auto" else args.watermark_backend
    if args.watermark_backend != "auto" and backend != detected_backend:
        raise ValueError(
            f"Requested backend {backend}, but metadata indicates {detected_backend}."
        )
    normalize_messages(records, backend)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AudioTokenizer(
        device=device,
        enable_ts=backend == "traceablespeech",
        ts_checkpoint=args.ts_checkpoint,
        ts_config=args.ts_config,
        watermark_backend=backend,
        voicemark_root=args.voicemark_root,
        voicemark_config=args.voicemark_config,
        voicemark_st_checkpoint=args.voicemark_st_checkpoint,
        voicemark_checkpoint=args.voicemark_checkpoint,
        voicemark_embed_vq1=True,
    )
    if not tokenizer.has_watermark_decoder:
        raise RuntimeError(f"Failed to load the {backend} watermark detector.")

    attacks = build_voicemark_valid_attacks(tokenizer.sample_rate)
    if args.skip_codecs:
        attacks = [item for item in attacks if not item[2]]

    with torch.no_grad():
        attack_stats = evaluate_voicemark_attacks(
            tokenizer,
            audio_dir,
            records,
            attacks,
            tokenizer.sample_rate,
            device,
        )
    attack_summary = summarize_attacks(attack_stats)
    quality, quality_rows = load_quality(quality_path)

    clean_stats = attack_summary.get("Clean (Identity)", {})
    total_correct = sum(row["bits_correct"] for row in attack_summary.values())
    total_bits = sum(row["bits_total"] for row in attack_summary.values())
    model_label = source_summary.get("checkpoint", f"existing {backend} WAVs")
    table = make_voicemark_table(model_label, attack_summary, quality_rows)

    summary = {
        "count": len(records),
        "sample_rate": tokenizer.sample_rate,
        "checkpoint": model_label,
        "watermark_backend": backend,
        "bit_accuracy_unit": "binary_bit",
        "bit_accuracy_version": 2,
        "bits_per_message": 16,
        "native_message_format": (
            "4 hexadecimal symbols expanded MSB-first to 16 binary bits"
            if backend == "traceablespeech"
            else "16 binary bits"
        ),
        "source_audio_dir": str(audio_dir),
        "avg_pesq_wb_clean_vs_wm": quality.get("avg_pesq_wb"),
        "avg_stoi_clean_vs_wm": quality.get("avg_stoi"),
        "avg_si_snr_clean_vs_wm": quality.get("avg_si_snr_db"),
        "wm_bit_accuracy": clean_stats.get("wm_bit_acc"),
        "wm_bits_correct": clean_stats.get("bits_correct", 0),
        "wm_bits_total": clean_stats.get("bits_total", 0),
        "attack_overall_accuracy": total_correct / total_bits if total_bits else None,
        "attack_bits_correct": total_correct,
        "attack_bits_total": total_bits,
        "attacks": attack_summary,
        "table_file": str(output_dir / "watermark_validation_table.txt"),
        "details": records,
    }
    (output_dir / "watermark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "watermark_validation_table.txt").write_text(
        table + "\n", encoding="utf-8"
    )
    print(table)


if __name__ == "__main__":
    main()

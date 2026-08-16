#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torchaudio
from encodec.utils import convert_audio

from batch_watermark_quality import evaluate_pairs
from compare_audio_quality import compute_metrics
from valle.bin.attacks import AudioEffects
from valle.data import AudioTokenizer


def numeric_prefix(path: Path) -> int:
    return int(path.stem.split("_", 1)[0])


def load_source_meta(source_dir: Path, index: int) -> dict:
    meta_path = source_dir / f"{index}_wm.json"
    if not meta_path.is_file():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def tensor_only(value):
    if isinstance(value, tuple):
        return value[0]
    return value


def resample_waveform(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, orig_sr, target_sr)


def match_length(wav: torch.Tensor, length: int) -> torch.Tensor:
    if wav.shape[-1] > length:
        return wav[..., :length]
    if wav.shape[-1] < length:
        return torch.nn.functional.pad(wav, (0, length - wav.shape[-1]))
    return wav


def hex_symbols_to_bits(symbols: torch.Tensor) -> torch.Tensor:
    """Expand TraceableSpeech 0..15 watermark symbols into binary bits."""
    symbols = symbols.detach().long()
    shifts = torch.tensor([3, 2, 1, 0], device=symbols.device, dtype=torch.long)
    bits = (symbols.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(*symbols.shape[:-1], symbols.shape[-1] * 4)


def bit_stats(
    pred: torch.Tensor, target: torch.Tensor
) -> tuple[int, int, float, torch.Tensor, torch.Tensor]:
    pred_bits = hex_symbols_to_bits(pred).to(target.device)
    target_bits = hex_symbols_to_bits(target)
    correct = int((pred_bits == target_bits).sum().item())
    total = int(target_bits.numel())
    return correct, total, correct / total if total else 0.0, pred_bits, target_bits


def detect_record(
    tokenizer: AudioTokenizer, wav: torch.Tensor, sign: torch.Tensor, wav_sr: Optional[int] = None
) -> Optional[dict]:
    if wav_sr is not None and wav_sr != tokenizer.sample_rate:
        wav = resample_waveform(wav.detach().cpu(), wav_sr, tokenizer.sample_rate)
    detection = tokenizer.detect_watermark(wav)
    if detection is None:
        return None
    detect_prob, pred, _ = detection
    correct, total, acc, pred_bits, _ = bit_stats(pred, sign)
    return {
        "detect_prob": float(detect_prob.mean().item()),
        "accuracy": acc,
        "bits_correct": correct,
        "bits_total": total,
        "predicted_bits": pred_bits.detach().cpu().int().tolist(),
    }


def attack_functions(sample_rate: int) -> list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]]:
    return [
        ("speed", lambda wav: AudioEffects.speed(wav, speed_range=(1.2, 1.2), sample_rate=sample_rate)),
        ("updownresample", lambda wav: AudioEffects.updownresample(wav, sample_rate=sample_rate)),
        ("echo", lambda wav: AudioEffects.echo(wav, sample_rate=sample_rate)),
        ("random_noise", lambda wav: AudioEffects.random_noise(wav)),
        ("pink_noise", lambda wav: AudioEffects.pink_noise(wav)),
        ("lowpass_filter", lambda wav: AudioEffects.lowpass_filter(wav, sample_rate=sample_rate)),
        ("highpass_filter", lambda wav: AudioEffects.highpass_filter(wav, sample_rate=sample_rate)),
        ("bandpass_filter", lambda wav: AudioEffects.bandpass_filter(wav, sample_rate=sample_rate)),
        ("smooth", lambda wav: AudioEffects.smooth(wav)),
        ("boost_audio", lambda wav: AudioEffects.boost_audio(wav)),
        ("duck_audio", lambda wav: AudioEffects.duck_audio(wav)),
        ("identity", lambda wav: AudioEffects.identity(wav)),
        ("shush", lambda wav: AudioEffects.shush(wav)),
        ("encodec", lambda wav: AudioEffects.encodec(wav, sample_rate=sample_rate)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TraceableSpeech checkpoint inference on existing clean wavs and write VoiceMark-style eval outputs."
    )
    parser.add_argument("--source-dir", type=Path, default=Path("infer/wm_eval_epoch40_100"))
    parser.add_argument("--output-dir", type=Path, default=Path("infer/ts_pt_eval_epoch40_100"))
    parser.add_argument("--checkpoint", default="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000")
    parser.add_argument("--config", default="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--eval-sr", type=int, default=16000)
    parser.add_argument(
        "--output-sr",
        type=int,
        default=None,
        help="Final wav sample rate. Defaults to TraceableSpeech's native sample rate.",
    )
    args = parser.parse_args()

    tokenizer = AudioTokenizer(
        watermark_backend="traceablespeech",
        enable_ts=True,
        ts_checkpoint=args.checkpoint,
        ts_config=args.config,
    )
    ts_sample_rate = tokenizer.sample_rate
    output_sample_rate = args.output_sr or ts_sample_rate
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(args.source_dir.glob("*_clean.wav"), key=numeric_prefix)[: args.max_samples]
    attacks = attack_functions(output_sample_rate)
    attack_stats = {name: {"count": 0, "bits_correct": 0, "bits_total": 0} for name, _ in attacks}
    records = []
    wm_bits_correct = 0
    wm_bits_total = 0

    with torch.no_grad():
        for source_path in source_files:
            index = numeric_prefix(source_path)
            source_meta = load_source_meta(args.source_dir, index)
            source_wav, source_sr = torchaudio.load(source_path)
            ts_wav = convert_audio(source_wav, source_sr, ts_sample_rate, tokenizer.channels).unsqueeze(0).to(tokenizer.device)

            frames = tokenizer.encode(ts_wav)
            watermark = tokenizer.random_watermark(ts_wav.shape[0])
            if watermark is None:
                continue
            clean_ts, watermarked_ts = tokenizer.decode_pair(frames, watermark_sign=watermark)
            if clean_ts is None or watermarked_ts is None:
                continue
            clean_out = resample_waveform(
                clean_ts[0].detach().cpu(), ts_sample_rate, output_sample_rate
            )
            watermarked_out = resample_waveform(
                watermarked_ts[0].detach().cpu(), ts_sample_rate, output_sample_rate
            )
            watermarked_out = match_length(watermarked_out, clean_out.shape[-1])
            watermarked_eval = watermarked_out.unsqueeze(0)

            clean_name = f"{index}_clean.wav"
            wm_name = f"{index}_wm.wav"
            clean_path = args.output_dir / clean_name
            wm_path = args.output_dir / wm_name
            torchaudio.save(str(clean_path), clean_out.cpu(), output_sample_rate)
            torchaudio.save(str(wm_path), watermarked_out.cpu(), output_sample_rate)
            pair_metrics = compute_metrics(clean_path, wm_path, args.eval_sr)

            record = {
                "index": index,
                "text": source_meta.get("text"),
                "sample_rate": output_sample_rate,
                "clean": clean_name,
                "watermarked": wm_name,
                "watermark_symbols": watermark.detach().cpu().int().tolist(),
                "watermark_bits": hex_symbols_to_bits(watermark).detach().cpu().int().tolist(),
                "bit_accuracy_unit": "binary_bit",
            }
            if pair_metrics.get("pesq_wb") is not None:
                record["pesq_wb_clean_vs_wm"] = pair_metrics["pesq_wb"]
            if pair_metrics.get("stoi") is not None:
                record["stoi_clean_vs_wm"] = pair_metrics["stoi"]

            detection = detect_record(tokenizer, watermarked_eval, watermark, output_sample_rate)
            if detection is not None:
                record.update(detection)
                wm_bits_correct += detection["bits_correct"]
                wm_bits_total += detection["bits_total"]

            record["attacks"] = {}
            for attack_name, attack_fn in attacks:
                try:
                    attacked = tensor_only(attack_fn(watermarked_eval.clone().to(tokenizer.device)))
                    attacked_detection = detect_record(tokenizer, attacked, watermark, output_sample_rate)
                except Exception as exc:
                    record["attacks"][attack_name] = {"error": str(exc)}
                    continue
                if attacked_detection is None:
                    continue
                bits_correct = attacked_detection["bits_correct"]
                bits_total = attacked_detection["bits_total"]
                attack_stats[attack_name]["count"] += 1
                attack_stats[attack_name]["bits_correct"] += bits_correct
                attack_stats[attack_name]["bits_total"] += bits_total
                record["attacks"][attack_name] = {
                    "accuracy": attacked_detection["accuracy"],
                    "bits_correct": bits_correct,
                    "bits_total": bits_total,
                }

            (args.output_dir / f"{index}_wm.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            records.append(record)
            print(
                f"{index}: wm_acc={record.get('accuracy')} "
                f"bits={record.get('bits_correct')}/{record.get('bits_total')}"
            )

    quality = evaluate_pairs(args.output_dir, args.eval_sr)
    (args.output_dir / "metrics.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    total_attack_correct = sum(stats["bits_correct"] for stats in attack_stats.values())
    total_attack_bits = sum(stats["bits_total"] for stats in attack_stats.values())
    summary = {
        "count": len(records),
        "sample_rate": output_sample_rate,
        "source_dir": str(args.source_dir),
        "checkpoint": args.checkpoint,
        "watermark_backend": "traceablespeech",
        "bit_accuracy_unit": "binary_bit",
        "bit_accuracy_version": 2,
        "bits_per_message": 16,
        "native_message_format": "4 hexadecimal symbols expanded MSB-first to 16 binary bits",
        "avg_pesq_wb_clean_vs_wm": quality.get("avg_pesq_wb"),
        "avg_stoi_clean_vs_wm": quality.get("avg_stoi"),
        "wm_bit_accuracy": wm_bits_correct / wm_bits_total if wm_bits_total else None,
        "wm_bits_correct": int(wm_bits_correct),
        "wm_bits_total": int(wm_bits_total),
        "attack_overall_accuracy": total_attack_correct / total_attack_bits if total_attack_bits else None,
        "attack_bits_correct": int(total_attack_correct),
        "attack_bits_total": int(total_attack_bits),
        "attacks": {
            name: {
                "accuracy": stats["bits_correct"] / stats["bits_total"] if stats["bits_total"] else None,
                "bits_correct": int(stats["bits_correct"]),
                "bits_total": int(stats["bits_total"]),
                "count": int(stats["count"]),
            }
            for name, stats in attack_stats.items()
        },
        "details": records,
    }
    (args.output_dir / "watermark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in [
        "count",
        "sample_rate",
        "avg_pesq_wb_clean_vs_wm",
        "avg_stoi_clean_vs_wm",
        "wm_bit_accuracy",
        "attack_overall_accuracy",
    ]}, indent=2))


if __name__ == "__main__":
    main()

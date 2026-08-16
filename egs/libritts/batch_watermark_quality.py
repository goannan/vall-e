import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from compare_audio_quality import compute_metrics


def evaluate_pairs(
    output_dir: Path,
    target_sr: int,
    utmos_scorer: Optional[Callable[[Path], float]] = None,
) -> Dict[str, float]:
    """Compute intrusive metrics and optional UTMOS for clean/watermarked pairs.

    Supported naming conventions:
      - <id>_clean.wav paired with <id>_wm.wav (current VALL-E/VoiceMark)
      - <id>.wav paired with <id>.pre.wm.wav or <id>.wm.wav (legacy scripts)
    """
    results: List[Dict[str, float]] = []
    missing = []
    acc_values = []

    for wav in sorted(output_dir.glob("*.wav")):
        if (
            wav.name.endswith("_wm.wav")
            or wav.name.endswith(".wm.wav")
            or wav.name.endswith(".pre.wm.wav")
        ):
            continue
        # Seed-TTS mode also exposes <utterance_id>.wav for official WER/SIM.
        # It is a hard link/copy of one variant, not another clean reference.
        if not wav.name.endswith("_clean.wav") and wav.with_name(
            f"{wav.stem}_wm.wav"
        ).is_file():
            continue
        if wav.name.endswith("_clean.wav"):
            wm = wav.with_name(wav.name[: -len("_clean.wav")] + "_wm.wav")
        else:
            wm_pre = wav.with_suffix(".pre.wm.wav")
            wm = wm_pre if wm_pre.is_file() else wav.with_suffix(".wm.wav")
        if not wm.is_file():
            missing.append(wm)
            continue
        metrics = compute_metrics(wav, wm, target_sr)
        metrics.update({"ref": wav.name, "deg": wm.name})
        if utmos_scorer is not None:
            utmos_clean = float(utmos_scorer(wav))
            utmos_wm = float(utmos_scorer(wm))
            metrics.update(
                {
                    "utmos_clean": utmos_clean,
                    "utmos_wm": utmos_wm,
                    "utmos_delta_wm_minus_clean": utmos_wm - utmos_clean,
                }
            )

        meta_path = wm.with_suffix(".json")
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if "accuracy" in meta:
                    metrics["wm_accuracy"] = float(meta["accuracy"])
                    acc_values.append(metrics["wm_accuracy"])
            except Exception:
                pass

        results.append(metrics)

    if not results:
        raise RuntimeError("No clean/watermarked pairs found under output_dir")

    pesq_vals = [m["pesq_wb"] for m in results if m.get("pesq_wb") is not None]
    stoi_vals = [m["stoi"] for m in results if m.get("stoi") is not None]
    si_snr_vals = [
        m["si_snr_db"] for m in results if m.get("si_snr_db") is not None
    ]
    visqol_vals = [
        m["visqol_moslqo"]
        for m in results
        if m.get("visqol_moslqo") is not None
    ]
    avg_pesq = sum(pesq_vals) / len(pesq_vals) if pesq_vals else None
    avg_stoi = sum(stoi_vals) / len(stoi_vals) if stoi_vals else None
    avg_si_snr = sum(si_snr_vals) / len(si_snr_vals) if si_snr_vals else None
    avg_visqol = sum(visqol_vals) / len(visqol_vals) if visqol_vals else None
    utmos_clean_vals = [
        m["utmos_clean"] for m in results if m.get("utmos_clean") is not None
    ]
    utmos_wm_vals = [
        m["utmos_wm"] for m in results if m.get("utmos_wm") is not None
    ]
    utmos_delta_vals = [
        m["utmos_delta_wm_minus_clean"]
        for m in results
        if m.get("utmos_delta_wm_minus_clean") is not None
    ]
    avg_utmos_clean = (
        sum(utmos_clean_vals) / len(utmos_clean_vals) if utmos_clean_vals else None
    )
    avg_utmos_wm = (
        sum(utmos_wm_vals) / len(utmos_wm_vals) if utmos_wm_vals else None
    )
    avg_utmos_delta = (
        sum(utmos_delta_vals) / len(utmos_delta_vals) if utmos_delta_vals else None
    )
    avg_acc = sum(acc_values) / len(acc_values) if acc_values else None

    summary = {
        "count": len(results),
        "avg_pesq_wb": avg_pesq,
        "avg_stoi": avg_stoi,
        "avg_si_snr_db": avg_si_snr,
        "avg_visqol_moslqo": avg_visqol,
        "avg_utmos_clean": avg_utmos_clean,
        "avg_utmos_wm": avg_utmos_wm,
        "avg_utmos_delta_wm_minus_clean": avg_utmos_delta,
        "avg_wm_accuracy": avg_acc,
        "missing_pairs": [str(p) for p in missing],
        "details": results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Batch quality evaluation: compare watermarked wavs to clean references in the same directory."
    )
    parser.add_argument("--dir", type=Path, required=True, help="Directory containing <id>.wav and <id>.wm.wav pairs")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate for PESQ/STOI (8k or 16k)")
    parser.add_argument("--json", type=Path, help="Optional path to save full metrics as JSON")
    args = parser.parse_args()

    summary = evaluate_pairs(args.dir, args.sr)

    print(f"Evaluated pairs : {summary['count']}")
    print(f"Avg PESQ (wb)   : {summary['avg_pesq_wb']:.3f}" if summary['avg_pesq_wb'] is not None else "Avg PESQ (wb)   : N/A")
    print(f"Avg STOI        : {summary['avg_stoi']:.3f}" if summary['avg_stoi'] is not None else "Avg STOI        : N/A")
    print(f"Avg SI-SNR (dB) : {summary['avg_si_snr_db']:.3f}" if summary['avg_si_snr_db'] is not None else "Avg SI-SNR (dB) : N/A")
    if summary["avg_visqol_moslqo"] is not None:
        print(f"Avg ViSQOL      : {summary['avg_visqol_moslqo']:.3f}")
    if summary["avg_utmos_wm"] is not None:
        print(f"Avg UTMOS clean : {summary['avg_utmos_clean']:.3f}")
        print(f"Avg UTMOS wm    : {summary['avg_utmos_wm']:.3f}")
        print(
            "Avg UTMOS delta : "
            f"{summary['avg_utmos_delta_wm_minus_clean']:+.3f} (wm - clean)"
        )
    if summary["missing_pairs"]:
        print("Missing wm files:")
        for p in summary["missing_pairs"]:
            print(f"  {p}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Full metrics saved to {args.json}")


if __name__ == "__main__":
    main()

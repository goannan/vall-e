import argparse
import json
from pathlib import Path
from typing import Dict, List

from compare_audio_quality import compute_metrics


def evaluate_pairs(output_dir: Path, target_sr: int) -> Dict[str, float]:
    """Compute PESQ/STOI for all (clean, watermarked) pairs under output_dir.

    We treat <name>.wav as the clean reference and <name>.pre.wm.wav as the
    pre-attack watermarked counterpart if present; otherwise fall back to
    <name>.wm.wav.
    """
    results: List[Dict[str, float]] = []
    missing = []
    acc_values = []

    for wav in sorted(output_dir.glob("*.wav")):
        if wav.name.endswith(".wm.wav"):
            continue
        wm_pre = wav.with_suffix(".pre.wm.wav")
        wm = wm_pre if wm_pre.is_file() else wav.with_suffix(".wm.wav")
        if not wm.is_file():
            missing.append(wm)
            continue
        metrics = compute_metrics(wav, wm, target_sr)
        metrics.update({"ref": wav.name, "deg": wm.name})

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
    avg_pesq = sum(pesq_vals) / len(pesq_vals) if pesq_vals else None
    avg_stoi = sum(stoi_vals) / len(stoi_vals) if stoi_vals else None
    avg_acc = sum(acc_values) / len(acc_values) if acc_values else None

    summary = {
        "count": len(results),
        "avg_pesq_wb": avg_pesq,
        "avg_stoi": avg_stoi,
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

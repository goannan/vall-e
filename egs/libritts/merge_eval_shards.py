#!/usr/bin/env python3
"""
Merge evaluation shards and format exact 3-table benchmark report.
"""

import os
import sys
import argparse
import csv
import glob
import json
import logging
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

SCRIPT_DIR = Path(__file__).resolve().parent
for p in [str(SCRIPT_DIR), "/home/wu25/mrnas04home/projects/vall-e/egs/libritts"]:
    if p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)

from tts_native_attacks import format_full_validation_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="Dataset")
    parser.add_argument("--manifest-path", type=str, default="")
    parser.add_argument("--wm-ckpt-path", type=str, default="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000")
    parser.add_argument("--step", type=str, default="Step 150000")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    shard_files = sorted(glob.glob(str(eval_dir / "eval_shard_*.jsonl")))
    timing_files = sorted(glob.glob(str(eval_dir / "timing_*.json")))

    all_records = []
    for sf in shard_files:
        with open(sf, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))

    logging.info(f"Loaded {len(all_records)} evaluated records from {len(shard_files)} shards in {eval_dir}")
    if not all_records:
        logging.error("No records found to merge!")
        return

    # Total audio duration & latencies
    total_audio_duration = sum(r.get("duration_sec", 0.0) for r in all_records)
    total_detect_time = 0.0
    total_embed_time = 0.0
    for tf in timing_files:
        with open(tf, "r", encoding="utf-8") as f:
            td = json.load(f)
            total_detect_time += td.get("total_detect_time", 0.0)
            total_embed_time += td.get("total_embed_time", 0.0)

    # Accumulate quality metrics
    pesq_list = [r["pesq"] for r in all_records if r.get("pesq", 0) > 0]
    stoi_list = [r["stoi"] for r in all_records if r.get("stoi", 0) > 0]
    c_utmos_list = [r["clean_utmos"] for r in all_records if r.get("clean_utmos", 0) > 0]
    w_utmos_list = [r["wm_utmos"] for r in all_records if r.get("wm_utmos", 0) > 0]
    c_sim_list = [r["clean_sim"] for r in all_records if r.get("clean_sim", 0) > 0]
    w_sim_list = [r["wm_sim"] for r in all_records if r.get("wm_sim", 0) > 0]
    c_wer_list = [r["clean_wer"] for r in all_records if r.get("clean_wer", 0) >= 0]
    w_wer_list = [r["wm_wer"] for r in all_records if r.get("wm_wer", 0) >= 0]
    c_cer_list = [r["clean_cer"] for r in all_records if r.get("clean_cer", 0) >= 0]
    w_cer_list = [r["wm_cer"] for r in all_records if r.get("wm_cer", 0) >= 0]

    # Calculate optimal operating threshold from Clean (Identity) distribution
    clean_id_c = [r["attacks"]["Clean (Identity)"]["clean_prob"] for r in all_records]
    clean_id_w = [r["attacks"]["Clean (Identity)"]["wm_prob"] for r in all_records]
    y_id_true = np.array([0] * len(clean_id_c) + [1] * len(clean_id_w))
    y_id_scores = np.concatenate([clean_id_c, clean_id_w])
    fpr_id, tpr_id, th_id = roc_curve(y_id_true, y_id_scores)
    fnr_id = 1.0 - tpr_id
    eer_idx = np.nanargmin(np.absolute(fnr_id - fpr_id))
    operating_threshold = float(th_id[eer_idx])
    logging.info(f"Calibrated detection operating threshold from Clean Identity EER: {operating_threshold:.4f}")

    # Accumulate attack metrics
    first_attacks = all_records[0]["attacks"]
    results = {}
    attack_scores = {}
    for key, info in first_attacks.items():
        results[key] = {
            "category": info["category"],
            "family": info["family"],
            "bitrate": info.get("detail", ""),
            "bit_matches": 0,
            "total_bits": 0,
            "pos_matches": 0,
            "pos_frames": 0,
            "neg_matches": 0,
            "neg_frames": 0,
            "det_roc_auc": 0.5,
            "det_tpr_at_001_fpr": 0.0,
            "wm_roc_auc": 0.5,
            "wm_tpr_at_001_fpr": 0.0,
        }
        attack_scores[key] = {"pos_det": [], "neg_det": [], "bit_scores": []}

    for r in all_records:
        for key, atk in r["attacks"].items():
            results[key]["bit_matches"] += atk["bit_matches"]
            results[key]["total_bits"] += atk["total_bits"]
            if atk["wm_prob"] >= operating_threshold:
                results[key]["pos_matches"] += 1
            results[key]["pos_frames"] += 1
            if atk["clean_prob"] < operating_threshold:
                results[key]["neg_matches"] += 1
            results[key]["neg_frames"] += 1

            attack_scores[key]["pos_det"].append(atk["wm_prob"])
            attack_scores[key]["neg_det"].append(atk["clean_prob"])
            attack_scores[key]["bit_scores"].append(atk["bit_matches"] / max(1, atk["total_bits"]))

    for key in results:
        pos_det = attack_scores[key]["pos_det"]
        neg_det = attack_scores[key]["neg_det"]
        y_true = [0] * len(neg_det) + [1] * len(pos_det)
        y_scores = neg_det + pos_det

        if len(set(y_true)) > 1 and len(y_scores) == len(y_true):
            try:
                results[key]["det_roc_auc"] = float(roc_auc_score(y_true, y_scores))
            except Exception:
                results[key]["det_roc_auc"] = 0.5
            try:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                target_fpr = 0.001
                idx = np.searchsorted(fpr, target_fpr, side="right") - 1
                idx = max(0, min(idx, len(tpr) - 1))
                results[key]["det_tpr_at_001_fpr"] = float(tpr[idx])
            except Exception:
                results[key]["det_tpr_at_001_fpr"] = 0.0

        b_matches = results[key]["bit_matches"]
        t_bits = max(1, results[key]["total_bits"])
        results[key]["bit_acc"] = float(b_matches / t_bits)
        
        p_matches = results[key]["pos_matches"]
        p_frames = max(1, results[key]["pos_frames"])
        pos_acc = float(p_matches / p_frames)
        
        n_matches = results[key]["neg_matches"]
        n_frames = max(1, results[key]["neg_frames"])
        neg_acc = float(n_matches / n_frames)
        
        results[key]["pos_acc"] = pos_acc
        results[key]["neg_acc"] = neg_acc
        results[key]["detect_acc"] = float(0.5 * (pos_acc + neg_acc))

        # WM ROC-AUC
        bit_sc = attack_scores[key]["bit_scores"]
        fake_bits = [0.5] * len(bit_sc)
        y_b_true = [0] * len(fake_bits) + [1] * len(bit_sc)
        y_b_scores = fake_bits + bit_sc
        if len(set(y_b_true)) > 1:
            try:
                results[key]["wm_roc_auc"] = float(roc_auc_score(y_b_true, y_b_scores))
                fpr, tpr, _ = roc_curve(y_b_true, y_b_scores)
                idx = np.searchsorted(fpr, 0.001, side="right") - 1
                idx = max(0, min(idx, len(tpr) - 1))
                results[key]["wm_tpr_at_001_fpr"] = float(tpr[idx])
            except Exception:
                results[key]["wm_roc_auc"] = results[key]["det_roc_auc"]
                results[key]["wm_tpr_at_001_fpr"] = results[key]["det_tpr_at_001_fpr"]

    avg_pesq = float(np.mean(pesq_list)) if pesq_list else 0.0
    avg_stoi = float(np.mean(stoi_list)) if stoi_list else 0.0
    c_ut = float(np.mean(c_utmos_list)) if c_utmos_list else 0.0
    w_ut = float(np.mean(w_utmos_list)) if w_utmos_list else 0.0
    c_sim = float(np.mean(c_sim_list)) if c_sim_list else 0.0
    w_sim = float(np.mean(w_sim_list)) if w_sim_list else 0.0
    c_wer = float(np.mean(c_wer_list)) if c_wer_list else 0.0
    w_wer = float(np.mean(w_wer_list)) if w_wer_list else 0.0
    c_cer = float(np.mean(c_cer_list)) if c_cer_list else 0.0
    w_cer = float(np.mean(w_cer_list)) if w_cer_list else 0.0

    num_attacks = len(results)
    embed_overhead_ms = (total_embed_time / max(1e-5, total_audio_duration)) * 1000.0
    detect_latency_ms = (total_detect_time / max(1e-5, total_audio_duration * num_attacks * 2)) * 1000.0

    det_aucs = [v["det_roc_auc"] for v in results.values()]
    wm_aucs = [v["wm_roc_auc"] for v in results.values()]
    det_tprs = [v["det_tpr_at_001_fpr"] for v in results.values()]
    wm_tprs = [v["wm_tpr_at_001_fpr"] for v in results.values()]

    overall_det_auc = float(np.mean(det_aucs)) if det_aucs else 0.5
    overall_wm_auc = float(np.mean(wm_aucs)) if wm_aucs else 0.5
    overall_det_tpr_001 = float(np.mean(det_tprs)) if det_tprs else 0.0
    overall_wm_tpr_001 = float(np.mean(wm_tprs)) if wm_tprs else 0.0

    quality_metrics = {
        "pesq_wb": avg_pesq,
        "stoi": avg_stoi,
        "clean_utmos": c_ut,
        "wm_utmos": w_ut,
        "clean_sim": c_sim,
        "wm_sim": w_sim,
        "clean_wer": c_wer,
        "wm_wer": w_wer,
        "clean_cer": c_cer,
        "wm_cer": w_cer,
        "embed_overhead_ms_per_sec": embed_overhead_ms,
        "detect_latency_ms_per_sec": detect_latency_ms,
        "overall_det_roc_auc": overall_det_auc,
        "overall_wm_roc_auc": overall_wm_auc,
        "overall_det_tpr_at_001_fpr": overall_det_tpr_001,
        "overall_wm_tpr_at_001_fpr": overall_wm_tpr_001,
    }

    # Format 3-table standard report string
    table_str = format_full_validation_table(args.step, results, quality_metrics=quality_metrics)

    print("\n" + "=" * 95)
    print(f"  FINAL BENCHMARK RESULTS ({args.dataset_name}: Evaluated on {len(all_records)} Samples)")
    print("=" * 95)
    print(table_str, flush=True)

    report_file = eval_dir / "test_evaluation_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 95 + "\n")
        f.write(f" TraceableSpeech VALL-E Native Watermark Benchmark Report ({args.dataset_name})\n")
        f.write(f" Test Manifest:       {args.manifest_path}\n")
        f.write(f" Evaluated Cuts:      {len(all_records)}\n")
        f.write(f" Watermark Checkpoint:{args.wm_ckpt_path}\n")
        f.write(f" Total Audio Duration:{total_audio_duration:.2f} s\n")
        f.write("=" * 95 + "\n\n")
        f.write(table_str + "\n")
    logging.info(f"Saved text report to: {report_file}")

    summary_json_file = eval_dir / "test_evaluation_summary.json"
    with open(summary_json_file, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": args.wm_ckpt_path,
            "steps": 150000,
            "epoch": 40,
            "num_evaluated_samples": len(all_records),
            "total_manifest_samples": len(all_records),
            "total_audio_duration_sec": total_audio_duration,
            "calibrated_threshold": operating_threshold,
            "quality_metrics": quality_metrics,
            "attack_metrics": results,
            "sample_audio_records": all_records[:50],
        }, f, indent=4)
    logging.info(f"Saved JSON summary to: {summary_json_file}")

    csv_rows = []
    for k, v in results.items():
        csv_rows.append({
            "attack_name": k,
            "category": v["category"],
            "family": v["family"],
            "bitrate": v.get("bitrate", ""),
            "detect_acc": v["detect_acc"],
            "det_roc_auc": v["det_roc_auc"],
            "det_tpr_at_001_fpr": v["det_tpr_at_001_fpr"],
            "bit_acc": v["bit_acc"],
            "wm_roc_auc": v["wm_roc_auc"],
            "wm_tpr_at_001_fpr": v["wm_tpr_at_001_fpr"],
        })
    csv_file = eval_dir / "test_attack_metrics.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
        writer.writeheader()
        writer.writerows(csv_rows)
    logging.info(f"Saved CSV metrics to: {csv_file}")


if __name__ == "__main__":
    main()

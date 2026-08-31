with open("test_valle_native_watermark.py", "r", encoding="utf-8") as f:
    c = f.read()

# 1. Update attack_scores initialization
c = c.replace(
    'attack_scores = {key: {"pos_scores": [], "neg_scores": []} for key in results.keys()}',
    'attack_scores = {key: {"pos_det_scores": [], "neg_det_scores": [], "pos_wm_scores": [], "neg_wm_scores": []} for key in results.keys()}'
)

# 2. Update loop inside attacks
target_loop = """                try:
                    attacked_clean = atk_fn(clean_audio)
                except Exception:
                    attacked_clean = clean_audio

                t_det_1 = time.perf_counter()
                feat_atk_cl = generator.forward_feature(attacked_clean)
                prob_cl_t, _, _ = detector.detect_watermark(feat_atk_cl)
                total_detect_time += (time.perf_counter() - t_det_1)

                prob_cl = float(prob_cl_t.mean().item())
                clean_tp_flag = 1 if prob_cl >= 0.5 else 0
                tn_flag = 1 - clean_tp_flag

                results[key]["bit_matches"] += bit_matches
                results[key]["total_bits"] += 16
                results[key]["pos_matches"] += tp_flag
                results[key]["pos_frames"] += 1
                results[key]["neg_matches"] += tn_flag
                results[key]["neg_frames"] += 1

                attack_scores[key]["pos_scores"].append(prob_wm)
                attack_scores[key]["neg_scores"].append(prob_cl)"""

repl_loop = """                try:
                    attacked_clean = atk_fn(clean_audio)
                except Exception:
                    attacked_clean = clean_audio

                t_det_1 = time.perf_counter()
                feat_atk_cl = generator.forward_feature(attacked_clean)
                prob_cl_t, msg_out_cl, _ = detector.detect_watermark(feat_atk_cl)
                total_detect_time += (time.perf_counter() - t_det_1)

                prob_cl = float(prob_cl_t.mean().item())
                msg_pred_cl = msg_out_cl.squeeze(0).cpu().numpy().tolist()
                cl_bit_matches = sum(int(c1) == int(c2) for c1, c2 in zip(msg_pred_cl, msg_np))
                clean_tp_flag = 1 if prob_cl >= 0.5 else 0
                tn_flag = 1 - clean_tp_flag

                results[key]["bit_matches"] += bit_matches
                results[key]["total_bits"] += 16
                results[key]["pos_matches"] += tp_flag
                results[key]["pos_frames"] += 1
                results[key]["neg_matches"] += tn_flag
                results[key]["neg_frames"] += 1

                attack_scores[key]["pos_det_scores"].append(prob_wm)
                attack_scores[key]["neg_det_scores"].append(prob_cl)
                attack_scores[key]["pos_wm_scores"].append(bit_matches / 16.0)
                attack_scores[key]["neg_wm_scores"].append(cl_bit_matches / 16.0)"""

assert target_loop in c, "target_loop not found in test_valle_native_watermark.py"
c = c.replace(target_loop, repl_loop)

# 3. Update compute metrics block
target_calc = """    # 5. Compute Final Metrics & AUC
    summary = {}
    csv_rows = []
    all_y_true = []
    all_y_scores = []

    for key, stats in results.items():
        bit_acc = stats["bit_matches"] / max(1, stats["total_bits"])
        pos_acc = stats["pos_matches"] / max(1, stats["pos_frames"])
        neg_acc = stats["neg_matches"] / max(1, stats["neg_frames"])
        detect_acc = 0.5 * (pos_acc + neg_acc)

        pos_s = attack_scores[key]["pos_scores"]
        neg_s = attack_scores[key]["neg_scores"]
        y_true = [0] * len(neg_s) + [1] * len(pos_s)
        y_scores = neg_s + pos_s

        all_y_true.extend(y_true)
        all_y_scores.extend(y_scores)

        auc, tpr_001 = compute_auc_and_tpr_at_fpr(y_true, y_scores, target_fpr=0.001)

        summary[key] = {
            "category": stats["category"],
            "family": stats["family"],
            "bitrate": stats["bitrate"],
            "detect_acc": detect_acc,
            "bit_acc": bit_acc,
            "tpr": pos_acc,
            "tnr": neg_acc,
            "roc_auc": auc,
            "tpr_at_001_fpr": tpr_001,
        }

        csv_rows.append({
            "Attack": key,
            "Category": stats["category"],
            "Family": stats["family"],
            "Bitrate": stats["bitrate"],
            "Bit_Accuracy": f"{bit_acc:.4f}",
            "Detect_Accuracy": f"{detect_acc:.4f}",
            "TPR": f"{pos_acc:.4f}",
            "TNR": f"{neg_acc:.4f}",
            "ROC_AUC": f"{auc:.4f}",
            "TPR_at_001_FPR": f"{tpr_001:.4f}",
        })

    overall_auc, overall_tpr_001 = compute_auc_and_tpr_at_fpr(all_y_true, all_y_scores, target_fpr=0.001)"""

repl_calc = """    # 5. Compute Final Metrics & Dual AUC (Detection & Bit-Matching Extraction)
    summary = {}
    csv_rows = []
    all_det_true, all_det_scores = [], []
    all_wm_true, all_wm_scores = [], []

    for key, stats in results.items():
        bit_acc = stats["bit_matches"] / max(1, stats["total_bits"])
        pos_acc = stats["pos_matches"] / max(1, stats["pos_frames"])
        neg_acc = stats["neg_matches"] / max(1, stats["neg_frames"])
        detect_acc = 0.5 * (pos_acc + neg_acc)

        # 1. Detection ROC-AUC & TPR
        pos_d = attack_scores[key]["pos_det_scores"]
        neg_d = attack_scores[key]["neg_det_scores"]
        y_det_true = [0] * len(neg_d) + [1] * len(pos_d)
        y_det_scores = neg_d + pos_d
        all_det_true.extend(y_det_true)
        all_det_scores.extend(y_det_scores)
        det_auc, det_tpr_001 = compute_auc_and_tpr_at_fpr(y_det_true, y_det_scores, target_fpr=0.001)

        # 2. WM Bit-Matching Extraction ROC-AUC & TPR
        pos_w = attack_scores[key]["pos_wm_scores"]
        neg_w = attack_scores[key]["neg_wm_scores"]
        y_wm_true = [0] * len(neg_w) + [1] * len(pos_w)
        y_wm_scores = neg_w + pos_w
        all_wm_true.extend(y_wm_true)
        all_wm_scores.extend(y_wm_scores)
        wm_auc, wm_tpr_001 = compute_auc_and_tpr_at_fpr(y_wm_true, y_wm_scores, target_fpr=0.001)

        summary[key] = {
            "category": stats["category"],
            "family": stats["family"],
            "bitrate": stats["bitrate"],
            "detect_acc": detect_acc,
            "det_roc_auc": det_auc,
            "det_tpr_at_001_fpr": det_tpr_001,
            "bit_acc": bit_acc,
            "wm_roc_auc": wm_auc,
            "wm_tpr_at_001_fpr": wm_tpr_001,
            "tpr": pos_acc,
            "tnr": neg_acc,
        }

        csv_rows.append({
            "Attack": key,
            "Category": stats["category"],
            "Family": stats["family"],
            "Bitrate": stats["bitrate"],
            "Detect_Accuracy": f"{detect_acc:.4f}",
            "Det_ROC_AUC": f"{det_auc:.4f}",
            "Det_TPR_at_001_FPR": f"{det_tpr_001:.4f}",
            "WM_Bit_Accuracy": f"{bit_acc:.4f}",
            "WM_ROC_AUC": f"{wm_auc:.4f}",
            "WM_TPR_at_001_FPR": f"{wm_tpr_001:.4f}",
            "TPR": f"{pos_acc:.4f}",
            "TNR": f"{neg_acc:.4f}",
        })

    overall_det_auc, overall_det_tpr_001 = compute_auc_and_tpr_at_fpr(all_det_true, all_det_scores, target_fpr=0.001)
    overall_wm_auc, overall_wm_tpr_001 = compute_auc_and_tpr_at_fpr(all_wm_true, all_wm_scores, target_fpr=0.001)"""

assert target_calc in c, "target_calc not found in test_valle_native_watermark.py"
c = c.replace(target_calc, repl_calc)

c = c.replace('"overall_roc_auc": overall_auc', '"overall_det_roc_auc": overall_det_auc, "overall_wm_roc_auc": overall_wm_auc')
c = c.replace('"overall_tpr_at_001_fpr": overall_tpr_001', '"overall_det_tpr_at_001_fpr": overall_det_tpr_001, "overall_wm_tpr_at_001_fpr": overall_wm_tpr_001')

with open("test_valle_native_watermark.py", "w", encoding="utf-8") as f:
    f.write(c)
print("Updated test_valle_native_watermark.py with Dual ROC-AUC & TPR (Detect & Extraction)!")

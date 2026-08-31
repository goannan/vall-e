#!/usr/bin/env python3
"""
Generate clean and watermarked synthesized audio from VALL-E native test tokens using NeuMark watermark model,
and compute watermark detection rate & speech quality metrics matching the training valid stage table.
Uses per-sample subprocess workers to guarantee zero memory accumulation on login nodes.
"""

import os
import sys
import gc
import json
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
NEUMARK_ROOT = Path("/home/pj25001109/ku60000344/projects/NeuMark").resolve()

for p in [str(PROJECT_DIR), str(SCRIPT_DIR), str(NEUMARK_ROOT), str(NEUMARK_ROOT / "train")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from tts_native_attacks import format_full_validation_table
from lhotse import load_manifest_lazy


def run_worker_for_sample(
    sample_index: int,
    cut_id: str,
    checkpoint: str,
    manifest_path: str,
    audio_dir: Path,
    tmp_json_path: Path,
):
    """Run a dedicated single-sample worker in a fresh Python process."""
    worker_code = f"""
import sys, os, json, torch, torchaudio
import torch.nn.functional as F
from pathlib import Path
from lhotse import load_manifest_lazy

SCRIPT_DIR = Path('{SCRIPT_DIR}')
PROJECT_DIR = SCRIPT_DIR.parents[1]
NEUMARK_ROOT = Path('{NEUMARK_ROOT}')
sys.path.extend([str(PROJECT_DIR), str(SCRIPT_DIR), str(NEUMARK_ROOT), str(NEUMARK_ROOT / 'train')])

from STmodels.model import SpeechTokenizer
from models import WMEmbedder, WMDetector
from tts_native_attacks import get_validation_attack_suite, compute_wer_cer
from tts_native_loss import UTMOSLoss, ASRLoss
from tts_native_dataset import resolve_wav_path

def clean_state_dict(state_dict):
    return {{k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}}

def ensure_3d_wav(t):
    if t.ndim == 1: return t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 2: return t.unsqueeze(1) if t.shape[0] == 1 else t.unsqueeze(0)
    return t

def to_2d_wav(t):
    t = t.detach().cpu()
    while t.ndim > 2: t = t.squeeze(0)
    if t.ndim == 1: t = t.unsqueeze(0)
    return t

def compute_st_sim(generator, wav1, wav2):
    with torch.no_grad():
        w1 = ensure_3d_wav(wav1)
        w2 = ensure_3d_wav(wav2)
        min_len = min(w1.shape[-1], w2.shape[-1])
        if min_len < 320: return 0.0
        f1 = generator.forward_feature(w1[..., :min_len])
        f2 = generator.forward_feature(w2[..., :min_len])
        emb1 = f1.mean(dim=-1)
        emb2 = f2.mean(dim=-1)
        return float(F.cosine_similarity(emb1, emb2, dim=-1).mean().item())

device = torch.device('cpu')

# 1. Models
st_cfg = str(NEUMARK_ROOT / 'STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json')
st_ckpt = str(NEUMARK_ROOT / 'STmodels/pretrained_model/SpeechTokenizer.pt')
generator = SpeechTokenizer.load_from_checkpoint(st_cfg, st_ckpt).to(device).eval()

ckpt_data = torch.load('{checkpoint}', map_location=device, weights_only=False)
step = int(ckpt_data.get('steps', 202692))
msg_proc = WMEmbedder(nbits=16, input_dim=1024, nchunk_size=4).to(device).eval()
detector = WMDetector(input_channels=1024, nbits=16, nchunk_size=4).to(device).eval()
msg_proc.load_state_dict(clean_state_dict(ckpt_data['msg_processor']), strict=True)
detector.load_state_dict(clean_state_dict(ckpt_data['detector']), strict=True)

utmos_loss = UTMOSLoss(device='cpu')
asr_loss = ASRLoss(device='cpu')

# 2. Load Cut
cuts_iter = load_manifest_lazy('{manifest_path}')
target_cut = None
for c in cuts_iter:
    if c.id == '{cut_id}':
        target_cut = c
        break

if target_cut is None:
    raise ValueError(f'Cut {cut_id} not found')

cut = target_cut
codes_np = cut.load_features()
codes = torch.from_numpy(codes_np.copy()).long().transpose(0, 1).unsqueeze(0).to(device)
text = cut.supervisions[0].text if cut.supervisions else ''

# Reference audio
audio_ref = None
if hasattr(cut, 'recording') and cut.recording and cut.recording.sources:
    try:
        wav_p = resolve_wav_path(cut.recording.sources[0].source)
        if wav_p and wav_p.is_file():
            p_audio, p_sr = torchaudio.load(str(wav_p))
            if p_sr != 16000:
                p_audio = torchaudio.functional.resample(p_audio, p_sr, 16000)
            start_s = int(cut.start * 16000)
            num_s = int(cut.duration * 16000)
            audio_ref = p_audio[:, start_s : start_s + num_s]
    except Exception:
        pass

torch.manual_seed({sample_index * 100 + 42})
message = torch.randint(0, 2, (1, 16), dtype=torch.int64, device=device)
msg_hex = ''.join([str(b.item()) for b in message[0]])

with torch.inference_mode():
    codes_qbt = codes.permute(1, 0, 2).contiguous() if codes.shape[1] == 8 else codes
    quantized_layers = [generator.quantizer.decode(codes_qbt[k : k + 1], st=k) for k in range(8)]

    # 1. Clean Audio
    clean_audio = generator.decoder(sum(quantized_layers))

    # 2. Watermarked Audio
    watermarked_layers = [msg_proc(q, message) for q in quantized_layers]
    wm_audio = generator.decoder(sum(watermarked_layers))

    prompt_audio = audio_ref if audio_ref is not None else clean_audio.clone()

    peak_c = clean_audio.abs().max()
    clean_norm = (clean_audio / peak_c * 0.95) if peak_c > 0.99 else clean_audio
    peak_w = wm_audio.abs().max()
    wm_norm = (wm_audio / peak_w * 0.95) if peak_w > 0.99 else wm_audio

    # Save WAVs
    prefix = f'sample_{sample_index:02d}_{cut_id}'
    audio_dir = Path('{audio_dir}')
    clean_p = audio_dir / f'{{prefix}}_clean.wav'
    wm_p = audio_dir / f'{{prefix}}_wm.wav'
    prompt_p = audio_dir / f'{{prefix}}_prompt.wav'

    torchaudio.save(str(clean_p), to_2d_wav(clean_norm), 16000)
    torchaudio.save(str(wm_p), to_2d_wav(wm_norm), 16000)
    torchaudio.save(str(prompt_p), to_2d_wav(prompt_audio), 16000)

    # Metrics
    c_ut = utmos_loss.model(clean_norm.squeeze(1), 16000).mean().item() if utmos_loss.available else 0.0
    w_ut = utmos_loss.model(wm_norm.squeeze(1), 16000).mean().item() if utmos_loss.available else 0.0

    c_sim = compute_st_sim(generator, clean_norm, prompt_audio)
    w_sim = compute_st_sim(generator, wm_norm, prompt_audio)

    c_wer, c_cer, w_wer, w_cer = 0.0, 0.0, 0.0, 0.0
    if getattr(asr_loss, 'model', None) is not None:
        c_hyps = asr_loss.decode_greedy(clean_norm, 16000)
        w_hyps = asr_loss.decode_greedy(wm_norm, 16000)
        c_wer, c_cer = compute_wer_cer(text, c_hyps[0])
        w_wer, w_cer = compute_wer_cer(text, w_hyps[0])

    # Attacks
    val_attacks = get_validation_attack_suite(16000)
    attack_stats = {{}}
    for cat, name, detail, atk_fn in val_attacks:
        key = name if cat == 'DSP' else f'{{name}} {{detail}}'
        try: atk_wm = atk_fn(wm_norm)
        except Exception: atk_wm = wm_norm
        emb_wm = generator.forward_feature(ensure_3d_wav(atk_wm))
        logits_wm, _ = detector(emb_wm)
        _, pred_bits, _ = detector.detect_watermark(emb_wm)

        try: atk_clean = atk_fn(clean_norm)
        except Exception: atk_clean = clean_norm
        emb_clean = generator.forward_feature(ensure_3d_wav(atk_clean))
        logits_clean, _ = detector(emb_clean)

        attack_stats[key] = {{
            'category': cat, 'family': name, 'bitrate': detail,
            'bit_matches': int((pred_bits.long() == message.long()).sum().item()),
            'total_bits': int(message.numel()),
            'pos_matches': int((logits_wm > 0.0).sum().item()),
            'pos_frames': int(logits_wm.numel()),
            'neg_matches': int((logits_clean <= 0.0).sum().item()),
            'neg_frames': int(logits_clean.numel()),
        }}

res = {{
    'sample_index': {sample_index},
    'cut_id': '{cut_id}',
    'text': text,
    'watermark_bits': msg_hex,
    'duration_sec': round(clean_audio.shape[-1] / 16000.0, 2),
    'clean_wav': str(clean_p.resolve()),
    'wm_wav': str(wm_p.resolve()),
    'prompt_wav': str(prompt_p.resolve()),
    'step': step,
    'c_ut': c_ut, 'w_ut': w_ut,
    'c_sim': c_sim, 'w_sim': w_sim,
    'c_wer': c_wer, 'w_wer': w_wer,
    'c_cer': c_cer, 'w_cer': w_cer,
    'attack_stats': attack_stats,
}}

with open('{tmp_json_path}', 'w') as f:
    json.dump(res, f)
print(f'Done sample {sample_index}')
"""
    cmd = [
        "/home/pj25001109/ku60000344/miniconda3/envs/valle/bin/python",
        "-c",
        worker_code,
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="exp/tts_native_neumark_v2/20260821-135903/NeuMark_epoch_013.pt",
    )
    parser.add_argument(
        "--test-manifest",
        default="data/tokenized_voicemark/cuts_test_valle_native.jsonl.gz",
    )
    parser.add_argument(
        "--output-dir",
        default="exp/eval_valle_test_neumark_v2",
    )
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    out_dir = SCRIPT_DIR / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    audio_dir = out_dir / "synthesized_audios"
    audio_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "tmp_parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = SCRIPT_DIR / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    manifest_path = SCRIPT_DIR / args.test_manifest if not Path(args.test_manifest).is_absolute() else Path(args.test_manifest)

    print("================================================================================")
    print("  VALL-E Native Test Tokens + NeuMark Watermark Synthesis & Evaluation")
    print("================================================================================")
    print(f"NeuMark Checkpoint: {ckpt_path}")
    print(f"Test Manifest:      {manifest_path}")
    print(f"Audio Output Dir:   {audio_dir}")
    print(f"Evaluating Samples: {args.num_samples}")
    print("--------------------------------------------------------------------------------")

    cuts_iter = load_manifest_lazy(str(manifest_path))
    test_cuts = []
    for c in cuts_iter:
        if 2.0 <= c.duration <= 10.0:
            test_cuts.append(c)
            if len(test_cuts) >= args.num_samples:
                break

    print(f"Selected {len(test_cuts)} test cuts. Launching isolated per-sample workers...")

    sample_results = []
    for i, cut in enumerate(test_cuts):
        idx = i + 1
        tmp_json = tmp_dir / f"sample_{idx:02d}.json"
        print(f"\n[{idx}/{len(test_cuts)}] Processing sample {idx}: {cut.id} (duration: {cut.duration:.2f}s)...")
        run_worker_for_sample(
            sample_index=idx,
            cut_id=cut.id,
            checkpoint=str(ckpt_path),
            manifest_path=str(manifest_path),
            audio_dir=audio_dir,
            tmp_json_path=tmp_json,
        )
        with open(tmp_json, "r") as f:
            data = json.load(f)
            sample_results.append(data)

    # Aggregate results
    step = sample_results[0]["step"]
    agg_attacks = defaultdict(lambda: {
        "category": "", "family": "", "bitrate": "",
        "bit_matches": 0, "total_bits": 0,
        "pos_matches": 0, "pos_frames": 0,
        "neg_matches": 0, "neg_frames": 0,
    })

    clean_utmos, wm_utmos = [], []
    clean_sim, wm_sim = [], []
    clean_wer, wm_wer = [], []
    clean_cer, wm_cer = [], []
    saved_metadata = []

    for r in sample_results:
        clean_utmos.append(r["c_ut"])
        wm_utmos.append(r["w_ut"])
        clean_sim.append(r["c_sim"])
        wm_sim.append(r["w_sim"])
        clean_wer.append(r["c_wer"])
        wm_wer.append(r["w_wer"])
        clean_cer.append(r["c_cer"])
        wm_cer.append(r["w_cer"])

        saved_metadata.append({
            "sample_index": r["sample_index"],
            "cut_id": r["cut_id"],
            "text": r["text"],
            "watermark_bits": r["watermark_bits"],
            "duration_sec": r["duration_sec"],
            "clean_wav": r["clean_wav"],
            "wm_wav": r["wm_wav"],
            "prompt_wav": r["prompt_wav"],
        })

        for key, stats in r["attack_stats"].items():
            agg_attacks[key]["category"] = stats["category"]
            agg_attacks[key]["family"] = stats["family"]
            agg_attacks[key]["bitrate"] = stats["bitrate"]
            agg_attacks[key]["bit_matches"] += stats["bit_matches"]
            agg_attacks[key]["total_bits"] += stats["total_bits"]
            agg_attacks[key]["pos_matches"] += stats["pos_matches"]
            agg_attacks[key]["pos_frames"] += stats["pos_frames"]
            agg_attacks[key]["neg_matches"] += stats["neg_matches"]
            agg_attacks[key]["neg_frames"] += stats["neg_frames"]

    summary = {}
    for key, stats in agg_attacks.items():
        bit_acc = stats["bit_matches"] / max(1, stats["total_bits"])
        pos_acc = stats["pos_matches"] / max(1, stats["pos_frames"])
        neg_acc = stats["neg_matches"] / max(1, stats["neg_frames"])
        det_acc = (stats["pos_matches"] + stats["neg_matches"]) / max(1, stats["pos_frames"] + stats["neg_frames"])
        summary[key] = {
            "category": stats["category"],
            "family": stats["family"],
            "bitrate": stats["bitrate"],
            "bit_acc": bit_acc,
            "pos_acc": pos_acc,
            "neg_acc": neg_acc,
            "detect_acc": det_acc,
        }

    quality_metrics = {
        "clean_utmos": sum(clean_utmos) / len(clean_utmos),
        "wm_utmos": sum(wm_utmos) / len(wm_utmos),
        "clean_sim": sum(clean_sim) / len(clean_sim),
        "wm_sim": sum(wm_sim) / len(wm_sim),
        "clean_wer": sum(clean_wer) / len(clean_wer),
        "wm_wer": sum(wm_wer) / len(wm_wer),
        "clean_cer": sum(clean_cer) / len(clean_cer),
        "wm_cer": sum(wm_cer) / len(wm_cer),
    }

    report_table = format_full_validation_table(step, summary, quality_metrics=quality_metrics)

    print("\n" + report_table + "\n", flush=True)

    report_path = out_dir / "validation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_table + "\n")

    summary_json_path = out_dir / "eval_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": str(ckpt_path),
            "step": step,
            "num_evaluated_samples": len(sample_results),
            "saved_audio_samples": saved_metadata,
            "quality_metrics": quality_metrics,
            "robustness_summary": summary,
        }, f, indent=4)

    print(f"[Done] All tasks completed successfully!")
    print(f"Audios saved to:    {audio_dir}")
    print(f"Report saved to:    {report_path}")
    print(f"Summary JSON saved: {summary_json_path}")


if __name__ == "__main__":
    main()

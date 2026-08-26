#!/usr/bin/env python3
"""
Lightweight verification script to test VALL-E speech synthesis and diagnostic metrics
(ASR WER/CER, Speaker SIM, UTMOS) across different prompt and sampling configurations.
"""

import argparse
import importlib.machinery
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

# 1. Mock unneeded modules
for mod in [
    "k2", "k2.version", "kaldialign", "pypinyin", "pypinyin.contrib", "pypinyin.contrib.tone_convert",
    "phonemizer", "phonemizer.backend", "phonemizer.backend.espeak", "phonemizer.backend.espeak.language_switch",
    "phonemizer.backend.espeak.words_mismatch", "phonemizer.punctuation", "phonemizer.separator",
    "traceableSpeech", "traceableSpeech.env", "traceableSpeech.meldataset", "traceableSpeech.models", "traceableSpeech.watermark",
]:
    if mod not in sys.modules:
        m = MagicMock()
        m.__spec__ = importlib.machinery.ModuleSpec(mod, None)
        sys.modules[mod] = m

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = SCRIPT_DIR.parents[3]
NEUMARK_ROOT = (PROJECT_DIR.parent / "NeuMark").resolve()
ICEFALL_ROOT = (PROJECT_DIR.parent / "icefall").resolve()

for p in [
    str(PROJECT_DIR),
    str(SCRIPT_DIR),
    str(ICEFALL_ROOT),
    str(NEUMARK_ROOT),
    str(NEUMARK_ROOT / "train"),
    str(NEUMARK_ROOT / "STmodels"),
]:
    if p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)

from icefall.utils import AttributeDict
from lhotse import load_manifest_lazy
from STmodels.model import SpeechTokenizer
from valle.data.collation import get_text_token_collater
from valle.models import get_model

# Metric utilities
from tts_native_attacks import compute_wer_cer
from tts_native_loss import ASRLoss, SpeakerSimLoss, UTMOSLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Verify VALL-E Token Synthesis Quality")
    parser.add_argument("--valle-checkpoint", type=str, default="exp/valle_voicemark/epoch-40.pt")
    parser.add_argument("--st-config", type=str, default="STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json")
    parser.add_argument("--st-checkpoint", type=str, default="STmodels/pretrained_model/SpeechTokenizer.pt")
    parser.add_argument("--wavlm-checkpoint", type=str, default="models/wavlm_large_finetune.pth")
    parser.add_argument("--manifest", type=str, default="data/tokenized_voicemark/cuts_dev.jsonl.gz")
    parser.add_argument("--output-dir", type=str, default="exp/test_valle_synthesis")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of pairs to test")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_fixed_prompt(st_model, prompt_wav_path: Path, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load and tokenize fixed reference prompt audio."""
    wav, sr = torchaudio.load(str(prompt_wav_path))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav_t = wav.unsqueeze(0).to(device)  # [1, 1, T]
    with torch.no_grad():
        codes = st_model.encode(wav_t)  # [8, 1, T_frames] or [1, 8, T_frames]
        if codes.shape[0] == 8:
            codes = codes.permute(1, 2, 0)  # [1, T_frames, 8]
        elif codes.shape[1] == 8:
            codes = codes.permute(0, 2, 1)  # [1, T_frames, 8]
    return codes, wav


# Minimum token frames needed to survive SpeechTokenizer decoder convolutions
MIN_DECODE_FRAMES = 10


def decode_tokens(st_model, codes_th: torch.Tensor) -> torch.Tensor:
    """
    Decode [1, T, 8] acoustic codes to waveform [1, 1, T_samples] via SpeechTokenizer.
    Returns None if the token sequence is too short to decode.
    """
    T = codes_th.shape[1]
    if T < MIN_DECODE_FRAMES:
        logging.warning(
            f"decode_tokens: token length {T} < {MIN_DECODE_FRAMES}, "
            f"padding to minimum (output will be low quality)."
        )
        # Repeat-pad to minimum length
        repeat_factor = (MIN_DECODE_FRAMES + T - 1) // T
        codes_th = codes_th.repeat(1, repeat_factor, 1)[:, :MIN_DECODE_FRAMES, :]
    # SpeechTokenizer quantizer decode expects [8, 1, T]
    codes_qbt = codes_th.permute(2, 0, 1).contiguous()
    quantized_layers = [st_model.quantizer.decode(codes_qbt[k : k + 1], st=k) for k in range(8)]
    z = sum(quantized_layers)
    wav = st_model.decoder(z)
    return wav


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logging.info(f"Using device: {device}")

    # 1. Load SpeechTokenizer
    logging.info("Loading SpeechTokenizer...")
    st_cfg_path = NEUMARK_ROOT / args.st_config if not os.path.exists(args.st_config) else Path(args.st_config)
    st_ckpt_path = NEUMARK_ROOT / args.st_checkpoint if not os.path.exists(args.st_checkpoint) else Path(args.st_checkpoint)
    with open(st_cfg_path) as f:
        st_cfg = json.load(f)
    st_model = SpeechTokenizer(st_cfg)
    st_ckpt = torch.load(str(st_ckpt_path), map_location="cpu", weights_only=False)
    st_model.load_state_dict(st_ckpt if "model" not in st_ckpt else st_ckpt["model"])
    st_model.to(device)
    st_model.eval()

    # 2. Load VALL-E Model
    logging.info(f"Loading VALL-E from {args.valle_checkpoint}...")
    ckpt_data = torch.load(args.valle_checkpoint, map_location="cpu", weights_only=False)
    model_args = AttributeDict(ckpt_data)
    valle_model = get_model(model_args)
    valle_model.load_state_dict(ckpt_data["model"], strict=True)
    valle_model.to(device)
    valle_model.eval()

    # Log model configuration for diagnostics
    logging.info(
        f"VALL-E config: prefix_mode={valle_model.prefix_mode}, "
        f"ar_audio_prepend_bos={valle_model.ar_audio_prepend_bos}, "
        f"num_quantizers={valle_model.num_quantizers}"
    )

    text_tokens_file = "data/tokenized_voicemark/unique_text_tokens.k2symbols"
    text_collater = get_text_token_collater(text_tokens_file)

    # 3. Load Evaluation Models (ASR, SIM, UTMOS)
    logging.info("Loading Evaluation Models (Wav2Vec2 ASR, WavLM SIM, UTMOS)...")
    asr_model = ASRLoss(device=str(device))
    wavlm_path = SCRIPT_DIR / args.wavlm_checkpoint
    sim_model = SpeakerSimLoss(checkpoint_path=str(wavlm_path), device=str(device))
    utmos_model = UTMOSLoss(device=str(device))

    # 4. Load Test Cuts
    logging.info(f"Loading test cuts from {args.manifest}...")
    cuts = list(load_manifest_lazy(args.manifest))
    # Group by speaker to form pairs
    spk_map = {}
    for c in cuts:
        if c.supervisions and c.duration >= 3.0:
            spk = c.supervisions[0].speaker
            spk_map.setdefault(spk, []).append(c)

    test_pairs = []
    for spk, spk_cuts in spk_map.items():
        if len(spk_cuts) >= 2:
            test_pairs.append((spk_cuts[0], spk_cuts[1]))
            if len(test_pairs) >= args.num_samples:
                break

    logging.info(f"Selected {len(test_pairs)} test pairs for evaluation.\n")

    # Configurations to benchmark
    configs = [
        {
            "name": "Config A (Full Prompt Aligned, temp=0.8, top_k=25)",
            "temp": 0.8,
            "top_k": 25,
            "truncate_prompt": False,
            "proper_enroll": True,
        },
        {
            "name": "Config B (Full Prompt Aligned, temp=1.0, top_k=50)",
            "temp": 1.0,
            "top_k": 50,
            "truncate_prompt": False,
            "proper_enroll": True,
        },
        {
            "name": "Config C (Old Buggy Baseline: Trunc 150 frames + raw enroll, temp=1.0, top_k=-100)",
            "temp": 1.0,
            "top_k": -100,
            "truncate_prompt": True,
            "proper_enroll": False,
        },
    ]

    all_results = {}

    for cfg in configs:
        cfg_name = cfg["name"]
        logging.info(f"=== Testing {cfg_name} ===")
        wers, cers, sims, utmos_scores = [], [], [], []

        for i, (prompt_cut, target_cut) in enumerate(test_pairs):
            p_text = prompt_cut.supervisions[0].text
            t_text = target_cut.supervisions[0].text
            p_phonemes = prompt_cut.supervisions[0].custom["tokens"]["text"]
            t_phonemes = target_cut.supervisions[0].custom["tokens"]["text"]

            p_codes_np = prompt_cut.load_features()  # [T_p, 8]
            if cfg["truncate_prompt"]:
                p_codes_slice = p_codes_np[:150]
            else:
                p_codes_slice = p_codes_np

            audio_prompt_tokens = torch.from_numpy(p_codes_slice).long().unsqueeze(0).to(device)

            # Diagnostics for first sample of first config
            if cfg == configs[0] and i == 0:
                logging.info(
                    f"  DIAG: p_codes_np shape={p_codes_np.shape}, dtype={p_codes_np.dtype}, "
                    f"min={p_codes_np.min()}, max={p_codes_np.max()}"
                )
                logging.info(
                    f"  DIAG: p_phonemes len={len(p_phonemes)}, first5={p_phonemes[:5]}"
                )
                logging.info(
                    f"  DIAG: t_phonemes len={len(t_phonemes)}, first5={t_phonemes[:5]}"
                )
                logging.info(
                    f"  DIAG: audio_prompt_tokens shape={audio_prompt_tokens.shape}, "
                    f"min={audio_prompt_tokens.min().item()}, max={audio_prompt_tokens.max().item()}"
                )

            full_phonemes = p_phonemes + ["_"] + t_phonemes
            text_tokens_idx, text_tokens_lens = text_collater([full_phonemes])
            text_tokens_idx = text_tokens_idx.to(device)
            text_tokens_lens = text_tokens_lens.to(device)

            if cfg["proper_enroll"]:
                _, enroll_x_lens = text_collater([p_phonemes])
                enroll_x_lens = enroll_x_lens.to(device)
            else:
                enroll_x_lens = torch.tensor([len(p_phonemes)], device=device)

            with torch.no_grad():
                logging.info(
                    f"  Inference input shapes: text={text_tokens_idx.shape}, "
                    f"audio_prompt={audio_prompt_tokens.shape}, "
                    f"enroll_x_lens={enroll_x_lens}"
                )
                gen_tokens = valle_model.inference(
                    text_tokens_idx,
                    text_tokens_lens,
                    audio_prompt_tokens,
                    enroll_x_lens=enroll_x_lens,
                    top_k=cfg["top_k"],
                    temperature=cfg["temp"],
                )
                # gen_tokens: [1, T_gen, 8]
                logging.info(f"  Generated tokens shape: {gen_tokens.shape}")

                if gen_tokens.shape[1] < 3:
                    logging.warning(
                        f"  SKIP: generated only {gen_tokens.shape[1]} frames (early EOS). "
                        f"This indicates prompt/text mismatch."
                    )
                    wers.append(1.0)
                    cers.append(1.0)
                    sims.append(0.0)
                    utmos_scores.append(0.0)
                    continue

                wav_gen = decode_tokens(st_model, gen_tokens)  # [1, 1, T_samples]

            # 1. ASR
            asr_hyp = asr_model.decode_greedy(wav_gen, 16000)[0]
            wer, cer = compute_wer_cer(t_text, asr_hyp)
            wers.append(wer)
            cers.append(cer)

            # 2. Speaker SIM
            try:
                # Load ground truth prompt audio
                p_audio_np = prompt_cut.load_audio()
                p_audio = torch.from_numpy(p_audio_np).float().to(device)
                if p_audio.ndim == 2 and p_audio.shape[0] > 1:
                    p_audio = p_audio.mean(dim=0, keepdim=True)
                elif p_audio.ndim == 1:
                    p_audio = p_audio.unsqueeze(0)
                sim_val = sim_model.get_similarity(wav_gen, p_audio.unsqueeze(0), 16000)
            except Exception:
                sim_val = 0.0
            sims.append(sim_val)

            # 3. UTMOS
            try:
                utmos_val = utmos_model.model(wav_gen.squeeze(1), 16000).mean().item()
            except Exception:
                utmos_val = 0.0
            utmos_scores.append(utmos_val)

            # Save sample audio for Config A
            if cfg == configs[0] and i < 3:
                out_path = out_dir / f"sample_{i}_gen.wav"
                torchaudio.save(str(out_path), wav_gen[0].cpu(), 16000)

            logging.info(
                f"[{i+1}/{len(test_pairs)}] Target: \"{t_text[:40]}...\" | "
                f"ASR: \"{asr_hyp[:40]}...\" | WER={wer:.3f}, SIM={sim_val:.3f}, UTMOS={utmos_val:.3f}"
            )

        mean_wer = sum(wers) / len(wers) if wers else 0.0
        mean_cer = sum(cers) / len(cers) if cers else 0.0
        mean_sim = sum(sims) / len(sims) if sims else 0.0
        mean_utmos = sum(utmos_scores) / len(utmos_scores) if utmos_scores else 0.0

        all_results[cfg_name] = {
            "WER": mean_wer,
            "CER": mean_cer,
            "SIM": mean_sim,
            "UTMOS": mean_utmos,
        }

    print("\n" + "=" * 90)
    print("  VALL-E Synthesis Configuration Benchmark Summary")
    print("=" * 90)
    print(f"{'Configuration':<60} | {'WER':<8} | {'CER':<8} | {'SIM':<8} | {'UTMOS':<8}")
    print("-" * 90)
    for name, res in all_results.items():
        print(f"{name:<60} | {res['WER']:<8.4f} | {res['CER']:<8.4f} | {res['SIM']:<8.4f} | {res['UTMOS']:<8.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()

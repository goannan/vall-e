#!/usr/bin/env python3
# Copyright (c) 2026
# Direct End-to-End TTS Inference Test: Audio + Text -> Speech Synthesis

import argparse
import json
import logging
import os
import sys
import types
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

# 1. Clean Mocks to avoid ABI / missing dependency issues
for mod in ["k2", "k2.version", "kaldialign"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

icefall_mock = types.ModuleType("icefall")
icefall_utils = types.ModuleType("icefall.utils")

def make_pad_mask(lengths, max_len=0):
    import torch
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    return seq_range.unsqueeze(0).expand(batch_size, max_len) >= lengths.unsqueeze(-1)

def str2bool(val):
    if isinstance(val, bool): return val
    return str(val).lower() in ("y", "yes", "t", "true", "on", "1")

class AttributeDict(dict):
    def __getattr__(self, key):
        try: return self[key]
        except KeyError: raise AttributeError(key)
    def __setattr__(self, key, value): self[key] = value

icefall_utils.make_pad_mask = make_pad_mask
icefall_utils.str2bool = str2bool
icefall_utils.AttributeDict = AttributeDict
icefall_mock.utils = icefall_utils
sys.modules["icefall"] = icefall_mock
sys.modules["icefall.utils"] = icefall_utils

import numpy as np
import torch
import torchaudio

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]

def find_neumark_root() -> Path:
    candidates = [
        os.environ.get("NEUMARK_ROOT"),
        SCRIPT_DIR.parent.parent.parent / "NeuMark",
        PROJECT_DIR.parent / "NeuMark",
        Path("/home/wu25/mrnas04home/projects/NeuMark"),
        Path.cwd() / "NeuMark",
    ]
    for c in candidates:
        if c:
            p = Path(c).resolve()
            if p.is_dir():
                return p
    return (PROJECT_DIR.parent / "NeuMark").resolve()

NEUMARK_ROOT = find_neumark_root()
for p in [str(PROJECT_DIR), str(SCRIPT_DIR), str(NEUMARK_ROOT), str(NEUMARK_ROOT / "train")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from STmodels.model import SpeechTokenizer
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from valle.data.collation import get_text_token_collater
from valle.models import get_model

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def phonemize_text(backend: EspeakBackend, text: str) -> List[str]:
    """Phonemize text into list of individual IPA phonemes and word boundary underscores."""
    separator = Separator(word="_", syllable="-", phone="|")
    res = backend.phonemize([text.strip()], separator=separator)
    raw = res[0].strip()
    tokens = []
    for word in raw.split("_"):
        for phone in word.split("|"):
            if phone:
                tokens.append(phone)
        tokens.append("_")
    if tokens and tokens[-1] == "_":
        tokens.pop()
    return tokens


def parse_args():
    parser = argparse.ArgumentParser(
        description="Direct End-to-End VALL-E TTS Synthesis from Audio Prompt + Text"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="exp/valle_voicemark/epoch-40.pt",
        help="Path to VALL-E model checkpoint (.pt)",
    )
    parser.add_argument(
        "--prompt-wav",
        type=str,
        default="prompts/8455_210777_000067_000000.wav",
        help="Path to reference prompt audio (.wav)",
    )
    parser.add_argument(
        "--prompt-text",
        type=str,
        default=None,
        help="Transcript of the prompt audio (if None, attempts reading .txt with same stem)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Artificial intelligence has revolutionized modern speech synthesis and voice cloning.",
        help="Target text sentence to synthesize",
    )
    parser.add_argument(
        "--text-tokens",
        type=str,
        default="data/tokenized_voicemark/unique_text_tokens.k2symbols",
        help="Path to text symbol table (.k2symbols)",
    )
    parser.add_argument(
        "--st-config",
        type=str,
        default=str(NEUMARK_ROOT / "STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json"),
        help="Path to SpeechTokenizer config JSON",
    )
    parser.add_argument(
        "--st-checkpoint",
        type=str,
        default=str(NEUMARK_ROOT / "STmodels/pretrained_model/SpeechTokenizer.pt"),
        help="Path to SpeechTokenizer weights (.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp/direct_tts_test_output",
        help="Output directory to save generated audios and summary",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-100,
        help="Top-K for AR sampling (-100 disables top-k)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for AR sampling (e.g. 1.0, 0.8, 0.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="Computation device (e.g. cuda:0, cuda:1, or cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_wav_path = Path(args.prompt_wav).resolve()
    if not prompt_wav_path.exists():
        logging.error(f"Prompt WAV not found: {prompt_wav_path}")
        sys.exit(1)

    # Determine prompt text
    prompt_text = args.prompt_text
    if not prompt_text:
        txt_path = prompt_wav_path.with_suffix(".txt")
        if txt_path.exists():
            prompt_text = txt_path.read_text(encoding="utf-8").strip()
            logging.info(f"Loaded prompt text from {txt_path}: \"{prompt_text}\"")
        else:
            logging.error(f"Prompt text not specified and {txt_path} not found.")
            sys.exit(1)

    target_text = args.text.strip()

    logging.info("=" * 70)
    logging.info(" Direct End-to-End VALL-E TTS Synthesis & PT Quality Check ")
    logging.info(f" VALL-E Checkpoint:  {args.checkpoint}")
    logging.info(f" Prompt Audio:       {prompt_wav_path}")
    logging.info(f" Prompt Text:        \"{prompt_text}\"")
    logging.info(f" Target Text:        \"{target_text}\"")
    logging.info(f" Temperature/Top-K:  Temp={args.temperature}, Top-K={args.top_k}")
    logging.info(f" Output Directory:   {out_dir}")
    logging.info(f" Device:             {device}")
    logging.info("=" * 70)

    # 1. Load SpeechTokenizer (Codec)
    logging.info("[1/4] Loading SpeechTokenizer codec model...")
    with open(args.st_config, "r") as f:
        st_cfg = json.load(f)
    st_model = SpeechTokenizer(st_cfg)
    st_state = torch.load(args.st_checkpoint, map_location="cpu")
    st_model.load_state_dict(st_state)
    st_model.to(device).eval()

    # 2. Load VALL-E Neural Model
    logging.info(f"[2/4] Loading VALL-E Model from {args.checkpoint}...")
    ckpt_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = AttributeDict(ckpt_data)
    valle_model = get_model(model_args)
    valle_model.load_state_dict(ckpt_data["model"], strict=True)
    valle_model.to(device).eval()

    text_tokens_file = args.text_tokens if os.path.exists(args.text_tokens) else model_args.text_tokens
    text_collater = get_text_token_collater(text_tokens_file)
    espeak_backend = EspeakBackend("en-us", preserve_punctuation=True, with_stress=False)

    # 3. Audio & Text Processing
    logging.info("[3/4] Tokenizing Prompt Audio and Text Sequences...")
    
    # Load and resample prompt audio to 16kHz
    prompt_wav, sr = torchaudio.load(str(prompt_wav_path))
    if prompt_wav.shape[0] > 1:
        prompt_wav = prompt_wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        prompt_wav = torchaudio.functional.resample(prompt_wav, sr, 16000)
    prompt_wav = prompt_wav.to(device)

    # Encode prompt audio using SpeechTokenizer -> [8, 1, T] -> [1, T, 8]
    with torch.no_grad():
        prompt_codes_qbt = st_model.encode(prompt_wav.unsqueeze(0))  # [8, 1, T]
        prompt_tokens = prompt_codes_qbt.permute(1, 2, 0)           # [1, T, 8]
        prompt_len = prompt_tokens.shape[1]

        # Mode A: Baseline Codec Reconstruction of the prompt itself (Zero VALL-E)
        reconstructed_prompt_wav = st_model.decode(prompt_codes_qbt).squeeze(0).cpu()

    # Phonemize prompt text and target text
    prompt_phonemes = phonemize_text(espeak_backend, prompt_text)
    target_phonemes = phonemize_text(espeak_backend, target_text)

    full_phonemes = prompt_phonemes + ["_"] + target_phonemes
    text_tokens_idx, text_tokens_lens = text_collater([full_phonemes])
    _, enroll_x_lens = text_collater([prompt_phonemes])

    text_tokens_idx = text_tokens_idx.to(device)
    text_tokens_lens = text_tokens_lens.to(device)
    enroll_x_lens = enroll_x_lens.to(device)

    logging.info(f"  Prompt Audio Frames:  {prompt_len} ({prompt_len * 0.01333:.2f}s)")
    logging.info(f"  Prompt Phonemes:      {len(prompt_phonemes)} -> {prompt_phonemes[:10]}...")
    logging.info(f"  Target Phonemes:      {len(target_phonemes)} -> {target_phonemes[:10]}...")
    logging.info(f"  Total Text Tokens:    {text_tokens_lens.item()} (Enroll prefix: {enroll_x_lens.item()})")

    # 4. Run VALL-E Inference
    logging.info("[4/4] Running VALL-E AR + NAR Neural Speech Synthesis...")
    with torch.no_grad():
        # Generation 1: default sampling (temperature, top_k)
        gen_tokens = valle_model.inference(
            text_tokens_idx,
            text_tokens_lens,
            prompt_tokens,
            enroll_x_lens=enroll_x_lens,
            top_k=args.top_k,
            temperature=args.temperature,
        )
        target_tokens = gen_tokens[0, prompt_len:, :].cpu()  # [T_gen, 8]
        logging.info(f"  Generated Target Acoustic Frames: {target_tokens.shape[0]} ({target_tokens.shape[0]*0.01333:.2f}s)")

        # Decode generated target tokens to waveform
        target_codes_qbt = target_tokens.permute(1, 0).unsqueeze(1).to(device)
        synth_wav = st_model.decode(target_codes_qbt).squeeze(0).cpu()

        # Generation 2: Conservative / stable sampling (temp=0.8, top_k=10)
        gen_tokens_clean = valle_model.inference(
            text_tokens_idx,
            text_tokens_lens,
            prompt_tokens,
            enroll_x_lens=enroll_x_lens,
            top_k=10,
            temperature=0.8,
        )
        target_tokens_clean = gen_tokens_clean[0, prompt_len:, :].cpu()
        target_codes_clean_qbt = target_tokens_clean.permute(1, 0).unsqueeze(1).to(device)
        synth_clean_wav = st_model.decode(target_codes_clean_qbt).squeeze(0).cpu()

    # 5. Save all audio outputs
    f_orig_prompt = out_dir / "01_prompt_original.wav"
    f_recon_prompt = out_dir / "02_prompt_codec_reconstructed.wav"
    f_synth = out_dir / f"03_valle_synth_temp{args.temperature}_topk{args.top_k}.wav"
    f_synth_conservative = out_dir / "04_valle_synth_temp0.8_topk10.wav"

    torchaudio.save(str(f_orig_prompt), prompt_wav.cpu(), 16000)
    torchaudio.save(str(f_recon_prompt), reconstructed_prompt_wav, 16000)
    torchaudio.save(str(f_synth), synth_wav, 16000)
    torchaudio.save(str(f_synth_conservative), synth_clean_wav, 16000)

    # Save summary report
    summary = {
        "checkpoint": str(args.checkpoint),
        "prompt_wav": str(prompt_wav_path),
        "prompt_text": prompt_text,
        "target_text": target_text,
        "prompt_duration_sec": prompt_wav.shape[-1] / 16000.0,
        "prompt_reconstructed_file": str(f_recon_prompt.name),
        "valle_synth_file": str(f_synth.name),
        "valle_synth_duration_sec": synth_wav.shape[-1] / 16000.0,
        "valle_conservative_file": str(f_synth_conservative.name),
        "valle_conservative_duration_sec": synth_clean_wav.shape[-1] / 16000.0,
    }
    with open(out_dir / "direct_tts_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info("=" * 70)
    logging.info(" ALL AUDIOS SUCCESSFULLY GENERATED! ")
    logging.info(f" 1. Original Prompt:            {f_orig_prompt}")
    logging.info(f" 2. Codec Reconstructed Prompt: {f_recon_prompt} (SpeechTokenizer Upper Bound)")
    logging.info(f" 3. VALL-E Synthesized:         {f_synth} (Temp={args.temperature}, TopK={args.top_k})")
    logging.info(f" 4. VALL-E Synthesized (T=0.8): {f_synth_conservative} (Temp=0.8, TopK=10)")
    logging.info(f" Summary JSON:                  {out_dir / 'direct_tts_summary.json'}")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()

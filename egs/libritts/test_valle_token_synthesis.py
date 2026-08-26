#!/usr/bin/env python3
"""Minimal VALL-E token-synthesis smoke test.

This test deliberately excludes watermark, ASR, speaker-similarity, and UTMOS
models.  It checks only the path needed before watermark training:

  checkpoint text vocabulary + aligned prompt -> VALL-E tokens -> SpeechTokenizer

For every selected pair it writes the reference-token codec reconstruction and
the VALL-E synthesis next to a JSON report, so the two WAV files can be compared
by listening before a large token dataset is generated.
"""

import argparse
import hashlib
import importlib.machinery
import json
import logging
import os
import random
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from unittest.mock import MagicMock

# The inference path does not use these optional dependencies, but imports in the
# training tree expect them to exist.
for module_name in [
    "k2",
    "k2.version",
    "kaldialign",
    "pypinyin",
    "pypinyin.contrib",
    "pypinyin.contrib.tone_convert",
    "phonemizer",
    "phonemizer.backend",
    "phonemizer.backend.espeak",
    "phonemizer.backend.espeak.language_switch",
    "phonemizer.backend.espeak.words_mismatch",
    "phonemizer.punctuation",
    "phonemizer.separator",
    "traceableSpeech",
    "traceableSpeech.env",
    "traceableSpeech.meldataset",
    "traceableSpeech.models",
    "traceableSpeech.watermark",
]:
    if module_name not in sys.modules:
        module = MagicMock()
        module.__spec__ = importlib.machinery.ModuleSpec(module_name, None)
        sys.modules[module_name] = module

import numpy as np
import torch
import torchaudio

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]


def find_directory(env_name: str, candidates: Sequence[Path]) -> Path:
    env_value = os.environ.get(env_name)
    paths = ([Path(env_value)] if env_value else []) + list(candidates)
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved.is_dir():
            return resolved
    raise FileNotFoundError(
        f"Could not find {env_name}. Checked: " + ", ".join(str(p) for p in paths)
    )


NEUMARK_ROOT = find_directory(
    "NEUMARK_ROOT",
    [PROJECT_DIR.parent / "NeuMark", SCRIPT_DIR.parents[2] / "NeuMark"],
)
ICEFALL_ROOT = find_directory(
    "ICEFALL_ROOT",
    [PROJECT_DIR.parent / "icefall", SCRIPT_DIR.parents[2] / "icefall"],
)

for import_path in [
    PROJECT_DIR,
    SCRIPT_DIR,
    ICEFALL_ROOT,
    NEUMARK_ROOT,
    NEUMARK_ROOT / "train",
]:
    value = str(import_path)
    if value not in sys.path:
        sys.path.insert(0, value)

from icefall.utils import AttributeDict
from lhotse import load_manifest_lazy
from STmodels.model import SpeechTokenizer
from valle.data.collation import get_text_token_collater
from valle.models import get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test aligned VALL-E token synthesis")
    parser.add_argument(
        "--valle-checkpoint", default="exp/valle_voicemark/epoch-40.pt"
    )
    parser.add_argument(
        "--manifest", default="data/tokenized_voicemark/cuts_dev.jsonl.gz"
    )
    parser.add_argument(
        "--text-tokens",
        default=None,
        help="Optional override; default uses the vocabulary stored in the checkpoint.",
    )
    parser.add_argument(
        "--st-config",
        default="STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json",
    )
    parser.add_argument(
        "--st-checkpoint", default="STmodels/pretrained_model/SpeechTokenizer.pt"
    )
    parser.add_argument("--output-dir", default="exp/test_valle_tokens_short")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--min-duration", type=float, default=3.0)
    parser.add_argument("--max-duration", type=float, default=10.0)
    parser.add_argument(
        "--prompt-cut-id",
        default="1462_170142_000021_000003-163",
        help="Default is a 2.00-second same-speaker prompt.",
    )
    parser.add_argument(
        "--target-cut-id",
        default="1462_170142_000038_000001-219",
        help='Default is the 0.83-second target "I was wrong."',
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-100,
        help=(
            "Use the checkpoint's original unrestricted sampling by default. "
            "Greedy top-k=1 can get stuck without emitting EOS."
        ),
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
        help="VALL-E inference precision; fp32 matches the validated synthesis path.",
    )
    parser.add_argument(
        "--sample-on-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep VALL-E on GPU and run only AR multinomial sampling on CPU.",
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--min-frame-ratio", type=float, default=0.5)
    parser.add_argument("--max-frame-ratio", type=float, default=1.5)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def resolve_file(path_value: str, bases: Sequence[Path]) -> Path:
    path = Path(path_value).expanduser()
    candidates = [path] if path.is_absolute() else [path] + [base / path for base in bases]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("File not found. Checked: " + ", ".join(map(str, candidates)))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_speaker_pairs(cuts, min_duration: float, max_duration: float):
    speakers = defaultdict(list)
    for cut in cuts:
        if cut.supervisions and min_duration <= cut.duration <= max_duration:
            speakers[cut.supervisions[0].speaker].append(cut)

    pairs = []
    for speaker_cuts in speakers.values():
        if len(speaker_cuts) < 2:
            continue
        for index, target_cut in enumerate(speaker_cuts):
            prompt_cut = speaker_cuts[(index + 1) % len(speaker_cuts)]
            pairs.append((prompt_cut, target_cut))
    return pairs


def select_pairs(cuts, args: argparse.Namespace):
    if bool(args.prompt_cut_id) != bool(args.target_cut_id):
        raise ValueError("--prompt-cut-id and --target-cut-id must be supplied together")
    if args.prompt_cut_id:
        by_id = {cut.id: cut for cut in cuts}
        try:
            prompt_cut = by_id[args.prompt_cut_id]
            target_cut = by_id[args.target_cut_id]
        except KeyError as error:
            raise KeyError(f"Cut ID not found in manifest: {error.args[0]}") from error
        prompt_speaker = prompt_cut.supervisions[0].speaker
        target_speaker = target_cut.supervisions[0].speaker
        if prompt_speaker != target_speaker:
            raise ValueError(
                f"Prompt/target speakers differ: {prompt_speaker} != {target_speaker}"
            )
        return [(prompt_cut, target_cut)]

    pairs = build_speaker_pairs(cuts, args.min_duration, args.max_duration)
    if not pairs:
        raise RuntimeError("No same-speaker prompt/target pairs satisfy the duration limits")
    return pairs[: args.num_samples]


def supervision_phonemes(cut) -> List[str]:
    try:
        return list(cut.supervisions[0].custom["tokens"]["text"])
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError(f"Cut {cut.id} has no supervision phoneme tokens") from error


def decode_codes(st_model, codes_btq: torch.Tensor) -> torch.Tensor:
    if codes_btq.ndim != 3 or codes_btq.shape[0] != 1 or codes_btq.shape[2] != 8:
        raise ValueError(f"Expected codes [1, T, 8], got {tuple(codes_btq.shape)}")
    if codes_btq.shape[1] == 0:
        raise ValueError("Cannot decode an empty token sequence")
    codes_qbt = codes_btq.permute(2, 0, 1).long().contiguous()
    return st_model.decode(codes_qbt)


def save_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    audio = waveform.detach().float().cpu()
    if audio.ndim == 3:
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(str(path), audio, sample_rate)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint_path = resolve_file(args.valle_checkpoint, [SCRIPT_DIR, PROJECT_DIR])
    manifest_path = resolve_file(args.manifest, [SCRIPT_DIR, PROJECT_DIR])
    st_config_path = resolve_file(args.st_config, [NEUMARK_ROOT, SCRIPT_DIR])
    st_checkpoint_path = resolve_file(args.st_checkpoint, [NEUMARK_ROOT, SCRIPT_DIR])
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading VALL-E checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False, mmap=True
    )
    model_args = AttributeDict(checkpoint)
    checkpoint_vocab_value = getattr(model_args, "text_tokens", None)
    if not checkpoint_vocab_value and not args.text_tokens:
        raise ValueError("Checkpoint has no text_tokens entry; pass --text-tokens")
    checkpoint_vocab_path: Optional[Path] = None
    if checkpoint_vocab_value:
        checkpoint_vocab_path = resolve_file(
            checkpoint_vocab_value, [SCRIPT_DIR, PROJECT_DIR]
        )
    text_tokens_path = (
        resolve_file(args.text_tokens, [SCRIPT_DIR, PROJECT_DIR])
        if args.text_tokens
        else checkpoint_vocab_path
    )
    assert text_tokens_path is not None

    if checkpoint_vocab_path and sha256(text_tokens_path) != sha256(checkpoint_vocab_path):
        logging.warning(
            "Vocabulary override differs from checkpoint vocabulary. This is expected to "
            "produce incorrect phoneme IDs unless the model was remapped."
        )
    logging.info(
        "Text vocabulary: %s (sha256=%s, source=%s)",
        text_tokens_path,
        sha256(text_tokens_path)[:12],
        "override" if args.text_tokens else "checkpoint",
    )
    text_collater = get_text_token_collater(str(text_tokens_path))

    valle_model = get_model(model_args)
    valle_model.load_state_dict(checkpoint["model"], strict=True)
    valle_model.to(device).eval()
    del checkpoint

    logging.info("Loading SpeechTokenizer: %s", st_checkpoint_path)
    st_model = SpeechTokenizer.load_from_checkpoint(
        str(st_config_path), str(st_checkpoint_path)
    ).to(device)
    st_model.eval()

    cuts = list(load_manifest_lazy(manifest_path))
    pairs = select_pairs(cuts, args)
    reports = []
    all_length_checks_passed = True

    with torch.inference_mode():
        for sample_index, (prompt_cut, target_cut) in enumerate(pairs):
            prompt_phonemes = supervision_phonemes(prompt_cut)
            target_phonemes = supervision_phonemes(target_cut)
            unknown = sorted(
                (set(prompt_phonemes) | set(target_phonemes) | {"_"})
                - set(text_collater.token2idx)
            )
            if unknown:
                raise ValueError(
                    f"Pair {prompt_cut.id} -> {target_cut.id} contains phonemes absent "
                    f"from the checkpoint vocabulary: {unknown}"
                )

            prompt_np = prompt_cut.load_features()
            target_np = target_cut.load_features()
            prompt_tokens = torch.from_numpy(prompt_np).long().unsqueeze(0).to(device)
            target_tokens = torch.from_numpy(target_np).long().unsqueeze(0).to(device)

            full_phonemes = prompt_phonemes + ["_"] + target_phonemes
            text_ids, text_lens = text_collater([full_phonemes])
            _, enroll_lens = text_collater([prompt_phonemes])

            logging.info(
                "[%d/%d] %s -> %s | prompt=%d frames, target=%d frames",
                sample_index + 1,
                len(pairs),
                prompt_cut.id,
                target_cut.id,
                prompt_tokens.shape[1],
                target_tokens.shape[1],
            )
            amp_context = nullcontext()
            if device.type == "cuda" and args.precision != "fp32":
                amp_dtype = (
                    torch.bfloat16
                    if args.precision == "bf16"
                    else torch.float16
                )
                amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype)
            with amp_context:
                generated = valle_model.inference(
                    text_ids.to(device),
                    text_lens.to(device),
                    prompt_tokens,
                    enroll_x_lens=enroll_lens.to(device),
                    top_k=args.top_k,
                    temperature=args.temperature,
                    sample_on_cpu=args.sample_on_cpu,
                )
            if isinstance(generated, tuple):
                generated = generated[0]
            if generated.ndim != 3 or generated.shape[2] != 8:
                raise RuntimeError(
                    f"Unexpected VALL-E output shape: {tuple(generated.shape)}"
                )

            sample_dir = output_dir / f"sample_{sample_index:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            reference_wav_path = sample_dir / "01_reference_tokens_codec.wav"
            generated_wav_path = sample_dir / "02_valle_generated.wav"
            generated_tokens_path = sample_dir / "02_valle_generated_tokens.npy"

            reference_wav = decode_codes(st_model, target_tokens)
            save_wav(reference_wav_path, reference_wav, st_model.sample_rate)
            np.save(generated_tokens_path, generated[0].cpu().numpy().astype(np.int16))

            generated_frames = int(generated.shape[1])
            frame_ratio = generated_frames / int(target_tokens.shape[1])
            length_ok = (
                generated_frames > 0
                and args.min_frame_ratio <= frame_ratio <= args.max_frame_ratio
            )
            all_length_checks_passed &= length_ok
            if generated_frames:
                generated_wav = decode_codes(st_model, generated)
                save_wav(generated_wav_path, generated_wav, st_model.sample_rate)

            report = {
                "prompt_cut_id": prompt_cut.id,
                "target_cut_id": target_cut.id,
                "speaker": prompt_cut.supervisions[0].speaker,
                "prompt_text": prompt_cut.supervisions[0].text,
                "target_text": target_cut.supervisions[0].text,
                "prompt_frames": int(prompt_tokens.shape[1]),
                "reference_target_frames": int(target_tokens.shape[1]),
                "generated_frames": generated_frames,
                "generated_to_reference_frame_ratio": frame_ratio,
                "length_check_passed": length_ok,
                "reference_codec_wav": str(reference_wav_path),
                "generated_wav": str(generated_wav_path) if generated_frames else None,
                "generated_tokens": str(generated_tokens_path),
            }
            reports.append(report)
            logging.info(
                "Generated %d frames (ratio=%.3f, length_check=%s): %s",
                generated_frames,
                frame_ratio,
                "PASS" if length_ok else "FAIL",
                generated_wav_path,
            )

    summary = {
        "valle_checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "text_tokens": str(text_tokens_path),
        "text_tokens_sha256": sha256(text_tokens_path),
        "checkpoint_train_stage": getattr(model_args, "train_stage", None),
        "prefix_mode": getattr(model_args, "prefix_mode", None),
        "top_k": args.top_k,
        "temperature": args.temperature,
        "precision": args.precision,
        "sample_on_cpu": args.sample_on_cpu,
        "seed": args.seed,
        "length_ratio_bounds": [args.min_frame_ratio, args.max_frame_ratio],
        "all_length_checks_passed": all_length_checks_passed,
        "samples": reports,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    logging.info("Summary: %s", summary_path)
    logging.info(
        "Listen to 01_reference_tokens_codec.wav first. If it is intelligible but "
        "02_valle_generated.wav is not, the codec path is healthy and VALL-E remains the issue."
    )
    return 0 if all_length_checks_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# Copyright (c) 2026
# Pre-generate VALL-E TTS-Native 8-layer acoustic tokens using Speaker-Paired Full Sentences (3-10s)

import argparse
import hashlib
import importlib.machinery
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import h5py
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

# Mock unneeded C++ and unused watermark dependencies
for mod in ["k2", "k2.version", "kaldialign", "pypinyin", "pypinyin.contrib", "pypinyin.contrib.tone_convert",
            "phonemizer", "phonemizer.backend", "phonemizer.backend.espeak", "phonemizer.backend.espeak.language_switch",
            "phonemizer.backend.espeak.words_mismatch", "phonemizer.punctuation", "phonemizer.separator",
            "traceableSpeech", "traceableSpeech.env", "traceableSpeech.meldataset", "traceableSpeech.models", "traceableSpeech.watermark"]:
    if mod not in sys.modules:
        m = MagicMock()
        m.__spec__ = importlib.machinery.ModuleSpec(mod, None)
        sys.modules[mod] = m

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent.parent

def find_neumark_root() -> Path:
    candidates = [
        os.environ.get("NEUMARK_ROOT"),
        SCRIPT_DIR.parent.parent.parent / "NeuMark",
        PROJECT_DIR.parent / "NeuMark",
        Path.cwd() / "NeuMark",
        Path.cwd().parent / "NeuMark",
        Path.home() / "projects" / "NeuMark",
    ]
    for c in candidates:
        if c:
            p = Path(c).resolve()
            if p.is_dir():
                return p
    return (PROJECT_DIR.parent / "NeuMark").resolve()

def find_icefall_root() -> Optional[Path]:
    candidates = [
        os.environ.get("ICEFALL_ROOT"),
        PROJECT_DIR.parent / "icefall",
        SCRIPT_DIR.parent.parent.parent / "icefall",
        Path.home() / "projects" / "icefall",
    ]
    for c in candidates:
        if c:
            p = Path(c).resolve()
            if p.is_dir():
                return p
    return None

NEUMARK_ROOT = find_neumark_root()
ICEFALL_ROOT = find_icefall_root()

paths_to_add = [
    str(PROJECT_DIR),
    str(SCRIPT_DIR),
    str(NEUMARK_ROOT),
    str(NEUMARK_ROOT / "train"),
]
if ICEFALL_ROOT:
    paths_to_add.append(str(ICEFALL_ROOT))

for p in paths_to_add:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from icefall.utils import AttributeDict
except ImportError:
    class AttributeDict(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)
        def __setattr__(self, key, value):
            self[key] = value
from lhotse import CutSet, MonoCut, SupervisionSegment, load_manifest_lazy
from lhotse.features import Features
from STmodels.model import SpeechTokenizer
from valle.data.collation import get_text_token_collater
from valle.models import get_model

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-generate VALL-E TTS-Native Token Dataset (3-10s Full Sentences)")
    parser.add_argument("--valle-checkpoint", type=str, default="exp/valle_voicemark/epoch-40.pt")
    parser.add_argument("--input-manifest", type=str, default="data/tokenized_voicemark/cuts_train.jsonl.gz")
    parser.add_argument("--output-manifest", type=str, default="data/tokenized_voicemark/cuts_train_valle_native_v4.jsonl.gz")
    parser.add_argument("--output-h5", type=str, default="data/tokenized_voicemark/libritts_valle_native_train_v4.h5")
    parser.add_argument(
        "--st-config",
        type=str,
        default="STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json",
        help="SpeechTokenizer config, resolved relative to NEUMARK_ROOT when needed.",
    )
    parser.add_argument(
        "--st-checkpoint",
        type=str,
        default="STmodels/pretrained_model/SpeechTokenizer.pt",
        help="SpeechTokenizer checkpoint used to encode every prompt WAV independently.",
    )
    parser.add_argument(
        "--text-tokens",
        type=str,
        default=None,
        help=(
            "Text vocabulary override. By default the exact vocabulary recorded "
            "in the VALL-E checkpoint is used."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=-1, help="Max pairs to generate (-1 for all)")
    parser.add_argument("--min-duration", type=float, default=3.0, help="Min cut duration in seconds")
    parser.add_argument("--max-duration", type=float, default=10.0, help="Max cut duration in seconds")
    parser.add_argument(
        "--prompt-max-frames",
        type=int,
        default=-1,
        help=(
            "Maximum prompt frames; -1 keeps the complete prompt (recommended). "
            "Truncating audio while retaining the complete prompt transcript is unaligned."
        ),
    )
    parser.add_argument("--top-k", type=int, default=-100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
        help="VALL-E inference precision; fp32 is the validated setting for this checkpoint.",
    )
    parser.add_argument(
        "--sample-on-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep VALL-E on GPU but sample each 1025-way AR distribution on CPU. "
            "This preserves the synthesis trajectory validated for this checkpoint."
        ),
    )
    parser.add_argument(
        "--min-generated-duration-ratio",
        type=float,
        default=0.25,
        help="Reject generations shorter than this multiple of the target recording duration.",
    )
    parser.add_argument(
        "--max-generated-duration-ratio",
        type=float,
        default=3.0,
        help="Reject generations longer than this multiple of the target recording duration.",
    )
    parser.add_argument(
        "--max-generation-attempts",
        type=int,
        default=3,
        help="Retry empty or implausibly long/short stochastic generations.",
    )
    parser.add_argument(
        "--preview-dir",
        type=str,
        default=None,
        help="Optional directory for prompt/reference/generated WAV previews.",
    )
    parser.add_argument(
        "--max-previews",
        type=int,
        default=10,
        help="Maximum preview triplets to save when --preview-dir is supplied.",
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def build_speaker_pairs(cuts, min_dur: float, max_dur: float):
    """Group cuts by speaker and pair each target cut with a different prompt cut from the same speaker."""
    spk_map = defaultdict(list)
    for c in cuts:
        if min_dur <= c.duration <= max_dur and c.supervisions:
            spk_id = c.supervisions[0].speaker
            spk_map[spk_id].append(c)

    pairs = []
    for spk_id, spk_cuts in spk_map.items():
        n = len(spk_cuts)
        if n < 2:
            continue
        for i, target_cut in enumerate(spk_cuts):
            prompt_cut = spk_cuts[(i + 1) % n]  # pick next utterance of same speaker as prompt
            pairs.append((prompt_cut, target_cut))
    return pairs


def resolve_text_tokens_path(path_value: str) -> Path:
    """Resolve a checkpoint-relative text vocabulary from common launch directories."""
    path = Path(path_value).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([SCRIPT_DIR / path, PROJECT_DIR / path])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Text vocabulary not found. Checked: {checked}")


def resolve_file(path_value: str, bases) -> Path:
    """Resolve a required file against explicit base directories."""
    path = Path(path_value).expanduser()
    candidates = [path] if path.is_absolute() else [path] + [base / path for base in bases]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Required file not found. Checked: {checked}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_cut_audio(cut) -> Path:
    """Relocate manifests whose recording paths still point to the old cluster home."""
    if not cut.has_recording or not cut.recording.sources:
        raise ValueError(f"Cut {cut.id} has no recording source for prompt encoding")

    source = Path(cut.recording.sources[0].source).expanduser()
    if source.is_file():
        return source.resolve()

    marker = "LibriTTS/"
    source_text = str(source)
    relative = Path(source_text.split(marker, 1)[1]) if marker in source_text else None
    workspace_root = PROJECT_DIR.parents[1]
    roots = []
    if os.environ.get("LIBRITTS_ROOT"):
        roots.append(Path(os.environ["LIBRITTS_ROOT"]).expanduser())
    roots.extend(
        [
            workspace_root / "dataset" / "libriTTS" / "LibriTTS",
            workspace_root / "dataset" / "LibriTTS",
            SCRIPT_DIR / "download" / "LibriTTS",
        ]
    )
    if relative is not None:
        for root in roots:
            candidate = root / relative
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot relocate prompt audio for cut {cut.id}: original source={source}"
    )


def load_prompt_waveform(cut, sample_rate: int, device: torch.device):
    """Load exactly the cut span and return [1, 1, samples] plus its source path."""
    audio_path = resolve_cut_audio(cut)
    waveform, source_rate = torchaudio.load(str(audio_path))
    start_sample = round(cut.start * source_rate)
    num_samples = round(cut.duration * source_rate)
    waveform = waveform[:, start_sample : start_sample + num_samples].float()
    if waveform.shape[-1] == 0:
        raise ValueError(f"Prompt cut {cut.id} resolved to an empty waveform")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, source_rate, sample_rate
        )
    return waveform.unsqueeze(0).to(device), audio_path


def normalize_st_codes(codes: torch.Tensor) -> torch.Tensor:
    """Normalize SpeechTokenizer output to VALL-E layout [1, frames, 8]."""
    while isinstance(codes, (list, tuple)):
        codes = codes[0]
    if not isinstance(codes, torch.Tensor) or codes.ndim != 3:
        raise ValueError(f"Unexpected SpeechTokenizer output: {type(codes)}")
    if codes.shape[0] == 8:  # [8, B, T]
        codes = codes.permute(1, 2, 0)
    elif codes.shape[1] == 8:  # [B, 8, T]
        codes = codes.permute(0, 2, 1)
    elif codes.shape[2] != 8:
        raise ValueError(f"Cannot infer SpeechTokenizer code layout: {tuple(codes.shape)}")
    if codes.shape[0] != 1 or codes.shape[2] != 8:
        raise ValueError(f"Expected prompt codes [1, T, 8], got {tuple(codes.shape)}")
    return codes.long().contiguous()


def encode_prompt_independently(st_model, prompt_cut, device: torch.device):
    """Encode one prompt with no batch padding, matching Seed-TTS inference."""
    waveform, audio_path = load_prompt_waveform(
        prompt_cut, st_model.sample_rate, device
    )
    codes = normalize_st_codes(st_model.encode(waveform))
    return codes, audio_path


def deterministic_generation_seed(cut_key: str, attempt: int) -> int:
    base = int(hashlib.sha256(cut_key.encode("utf-8")).hexdigest()[:8], 16)
    return (base + attempt) % (2**31)


def save_waveform(path: Path, waveform: torch.Tensor, sample_rate: int):
    audio = waveform.detach().float().cpu()
    if audio.ndim == 3:
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(str(path), audio, sample_rate)


def save_preview(
    preview_root: Path,
    preview_index: int,
    prompt_cut,
    target_cut,
    generated_codes: np.ndarray,
    st_model,
    device: torch.device,
    generation_attempt: int,
):
    """Save a directly listenable prompt/reference/VALL-E triplet."""
    sample_dir = preview_root / f"sample_{preview_index:02d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    prompt_waveform, prompt_path = load_prompt_waveform(
        prompt_cut, st_model.sample_rate, device
    )
    target_waveform, target_path = load_prompt_waveform(
        target_cut, st_model.sample_rate, device
    )
    codes_qbt = (
        torch.from_numpy(generated_codes)
        .long()
        .to(device)
        .transpose(0, 1)
        .unsqueeze(1)
        .contiguous()
    )
    generated_waveform = st_model.decode(codes_qbt)

    prompt_wav = sample_dir / "00_prompt.wav"
    target_wav = sample_dir / "01_target_reference.wav"
    generated_wav = sample_dir / "02_valle_generated.wav"
    save_waveform(prompt_wav, prompt_waveform, st_model.sample_rate)
    save_waveform(target_wav, target_waveform, st_model.sample_rate)
    save_waveform(generated_wav, generated_waveform, st_model.sample_rate)

    metadata = {
        "prompt_cut_id": prompt_cut.id,
        "prompt_text": prompt_cut.supervisions[0].text,
        "prompt_source": str(prompt_path),
        "target_cut_id": target_cut.id,
        "target_text": target_cut.supervisions[0].text,
        "target_source": str(target_path),
        "target_reference_duration": target_cut.duration,
        "generated_frames": int(generated_codes.shape[0]),
        "generated_duration": float(generated_codes.shape[0] * 0.02),
        "generated_to_target_duration_ratio": float(
            generated_codes.shape[0] * 0.02 / target_cut.duration
        ),
        "generation_attempt": generation_attempt,
        "prompt_wav": str(prompt_wav),
        "target_reference_wav": str(target_wav),
        "valle_generated_wav": str(generated_wav),
    }
    (sample_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.max_generation_attempts < 1:
        raise ValueError("--max-generation-attempts must be at least 1")
    if args.max_previews < 0:
        raise ValueError("--max-previews cannot be negative")
    if not (
        0 < args.min_generated_duration_ratio
        <= args.max_generated_duration_ratio
    ):
        raise ValueError("Invalid generated-duration ratio bounds")

    # Enable TF32 and cuDNN optimizations
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    logging.info(f"Using device: {device}, Rank: {args.rank}/{args.world_size}")

    # 1. Load VALL-E Model
    valle_checkpoint_path = resolve_file(
        args.valle_checkpoint, [SCRIPT_DIR, PROJECT_DIR]
    )
    valle_checkpoint_stat = valle_checkpoint_path.stat()
    logging.info("Loading VALL-E model from %s...", valle_checkpoint_path)
    ckpt_data = torch.load(
        str(valle_checkpoint_path), map_location="cpu", weights_only=False
    )
    model_args = AttributeDict(ckpt_data)
    valle_model = get_model(model_args)
    valle_model.load_state_dict(ckpt_data["model"], strict=True)
    valle_model.to(device)
    valle_model.eval()

    # Prompts must follow the same single-WAV tokenization path as Seed-TTS
    # inference. Loading prompt features from the prepare H5 is intentionally
    # forbidden because those codes depend on batch padding.
    st_config_path = resolve_file(args.st_config, [NEUMARK_ROOT, SCRIPT_DIR])
    st_checkpoint_path = resolve_file(
        args.st_checkpoint, [NEUMARK_ROOT, SCRIPT_DIR]
    )
    st_checkpoint_stat = st_checkpoint_path.stat()
    st_config_sha256 = sha256(st_config_path)
    logging.info(
        "Loading SpeechTokenizer for independent prompt encoding: %s",
        st_checkpoint_path,
    )
    st_model = SpeechTokenizer.load_from_checkpoint(
        str(st_config_path), str(st_checkpoint_path)
    ).to(device)
    st_model.eval()

    # 2. Text Collater
    checkpoint_text_tokens = getattr(model_args, "text_tokens", None)
    text_tokens_value = args.text_tokens or checkpoint_text_tokens
    if not text_tokens_value:
        raise ValueError(
            "No text vocabulary was supplied and the checkpoint has no 'text_tokens' entry."
        )
    text_tokens_path = resolve_text_tokens_path(text_tokens_value)
    vocab_sha256 = hashlib.sha256(text_tokens_path.read_bytes()).hexdigest()
    logging.info(
        "Using text vocabulary: %s (sha256=%s%s)",
        text_tokens_path,
        vocab_sha256[:12],
        ", explicit override" if args.text_tokens else ", from checkpoint",
    )
    text_collater = get_text_token_collater(str(text_tokens_path))

    if args.prompt_max_frames > 0:
        logging.warning(
            "--prompt-max-frames=%d truncates prompt audio but not its transcript; "
            "this is intentionally non-default and can cause early EOS or repetitions.",
            args.prompt_max_frames,
        )

    # 3. Load Cuts & Build Pairs
    logging.info(f"Loading input cuts from {args.input_manifest} (Filtering: {args.min_duration}s - {args.max_duration}s)...")
    raw_cuts = load_manifest_lazy(args.input_manifest)
    pairs = build_speaker_pairs(raw_cuts, args.min_duration, args.max_duration)
    logging.info(f"Total valid same-speaker pairs found: {len(pairs)}")

    # Shard for distributed generation
    if args.world_size > 1:
        pairs = pairs[args.rank :: args.world_size]
        logging.info(f"Rank {args.rank} assigned {len(pairs)} pairs.")

    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]

    out_h5_path = Path(args.output_h5)
    if args.world_size > 1:
        out_h5_path = out_h5_path.with_name(f"{out_h5_path.stem}_rank{args.rank}{out_h5_path.suffix}")
    out_h5_path.parent.mkdir(parents=True, exist_ok=True)

    out_manifest_path = Path(args.output_manifest)
    if args.world_size > 1:
        out_manifest_path = out_manifest_path.with_name(f"{out_manifest_path.stem}_rank{args.rank}.jsonl.gz")

    preview_dir = None
    if args.preview_dir and args.max_previews > 0:
        preview_dir = Path(args.preview_dir).expanduser().resolve()
        preview_dir.mkdir(parents=True, exist_ok=True)

    # 4. Open H5 file and reject incompatible resume data.
    try:
        h5_file = h5py.File(str(out_h5_path), "a")
        existing_keys = set(h5_file.keys())
        expected_metadata = {
            "generation_version": 4,
            "prompt_token_source": "single_wav_speechtokenizer_encode",
            "valle_checkpoint_path": str(valle_checkpoint_path),
            "valle_checkpoint_size": valle_checkpoint_stat.st_size,
            "valle_checkpoint_mtime_ns": valle_checkpoint_stat.st_mtime_ns,
            "text_tokens_sha256": vocab_sha256,
            "st_config_sha256": st_config_sha256,
            "st_checkpoint_path": str(st_checkpoint_path),
            "st_checkpoint_size": st_checkpoint_stat.st_size,
            "st_checkpoint_mtime_ns": st_checkpoint_stat.st_mtime_ns,
            "prompt_max_frames": args.prompt_max_frames,
            "min_generated_duration_ratio": args.min_generated_duration_ratio,
            "max_generated_duration_ratio": args.max_generated_duration_ratio,
            "valle_precision": args.precision,
            "ar_sampling_device": "cpu" if args.sample_on_cpu else str(device.type),
        }
        if existing_keys:
            mismatches = {
                key: (h5_file.attrs.get(key), expected)
                for key, expected in expected_metadata.items()
                if h5_file.attrs.get(key) != expected
            }
            if mismatches:
                h5_file.close()
                raise RuntimeError(
                    f"Existing H5 dataset {out_h5_path} was generated with incompatible "
                    f"settings: {mismatches}. Use a new output path; do not mix old and "
                    "corrected tokens."
                )
        else:
            for key, value in expected_metadata.items():
                h5_file.attrs[key] = value
            h5_file.flush()
        logging.info(f"Found {len(existing_keys)} already generated entries in {out_h5_path} (resuming)...")
    except RuntimeError:
        raise
    except Exception as ex:
        raise RuntimeError(
            f"Could not open H5 dataset {out_h5_path}: {ex}. Use a new output path "
            "instead of overwriting an existing dataset."
        ) from ex

    generated_cuts = []
    stats = Counter()

    # 5. Generation Loop (Clean single-sample execution per GPU)
    with torch.inference_mode():
        for prompt_cut, target_cut in tqdm(pairs, desc=f"VALL-E Gen [Rank {args.rank}]"):
            cut_key = f"{target_cut.id}_paired_{prompt_cut.id}"
            prompt_audio_path = None
            generation_attempt = None

            if cut_key in existing_keys:
                dataset = h5_file[cut_key]
                gen_codes_np = dataset[:]
                prompt_frames = int(
                    dataset.attrs.get(
                        "prompt_frames", round(prompt_cut.duration * 50)
                    )
                )
                prompt_audio_path = dataset.attrs.get("prompt_audio_path", "")
                generation_attempt = int(dataset.attrs.get("generation_attempt", 0))
                stats["resumed"] += 1
            else:
                try:
                    audio_prompt_tokens, prompt_audio_path = (
                        encode_prompt_independently(st_model, prompt_cut, device)
                    )
                except Exception as ex:
                    stats["prompt_encode_error"] += 1
                    logging.warning(
                        "Skipping target %s: independent prompt encoding failed for %s: %s",
                        target_cut.id,
                        prompt_cut.id,
                        ex,
                    )
                    continue

                prompt_frames = int(audio_prompt_tokens.shape[1])
                if prompt_frames < 20:
                    stats["prompt_too_short"] += 1
                    continue

                # VALL-E requires prompt audio and prompt text to describe the same span.
                # Keep the full prompt unless an explicit diagnostic override was supplied.
                if args.prompt_max_frames > 0:
                    prompt_frames = min(args.prompt_max_frames, prompt_frames)
                    audio_prompt_tokens = audio_prompt_tokens[:, :prompt_frames]

                # Full sentences phonemes
                p_phonemes = prompt_cut.supervisions[0].custom["tokens"]["text"]
                # Target sentence full text
                t_phonemes = target_cut.supervisions[0].custom["tokens"]["text"]

                unknown_phonemes = sorted(
                    (set(p_phonemes) | set(t_phonemes) | {"_"})
                    - set(text_collater.token2idx)
                )
                if unknown_phonemes:
                    logging.warning(
                        "Skipping target %s: phonemes absent from checkpoint vocabulary: %s",
                        target_cut.id,
                        unknown_phonemes,
                    )
                    stats["unknown_phoneme"] += 1
                    continue

                full_phonemes = p_phonemes + ["_"] + t_phonemes
                text_tokens_idx, text_tokens_lens = text_collater([full_phonemes])
                text_tokens_idx = text_tokens_idx.to(device)
                text_tokens_lens = text_tokens_lens.to(device)

                # This length includes the same BOS/EOS convention used by the full text.
                _, enroll_x_lens = text_collater([p_phonemes])
                enroll_x_lens = enroll_x_lens.to(device)

                target_tokens = None
                last_rejection = None
                for attempt in range(args.max_generation_attempts):
                    seed = deterministic_generation_seed(cut_key, attempt)
                    torch.manual_seed(seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(seed)
                    try:
                        amp_context = nullcontext()
                        if device.type == "cuda" and args.precision != "fp32":
                            amp_dtype = (
                                torch.bfloat16
                                if args.precision == "bf16"
                                else torch.float16
                            )
                            amp_context = torch.autocast(
                                device_type="cuda", dtype=amp_dtype
                            )
                        with amp_context:
                            gen_tokens = valle_model.inference(
                                text_tokens_idx,
                                text_tokens_lens,
                                audio_prompt_tokens,
                                enroll_x_lens=enroll_x_lens,
                                top_k=args.top_k,
                                temperature=args.temperature,
                                sample_on_cpu=args.sample_on_cpu,
                            )
                        candidate = gen_tokens[0].cpu().numpy().astype(np.int16)
                        generated_duration = candidate.shape[0] * 0.02
                        duration_ratio = generated_duration / target_cut.duration
                        if candidate.shape[0] < 10:
                            last_rejection = f"only {candidate.shape[0]} frames"
                        elif not (
                            args.min_generated_duration_ratio
                            <= duration_ratio
                            <= args.max_generated_duration_ratio
                        ):
                            last_rejection = (
                                f"duration ratio {duration_ratio:.3f} outside "
                                f"[{args.min_generated_duration_ratio}, "
                                f"{args.max_generated_duration_ratio}]"
                            )
                        else:
                            target_tokens = candidate
                            generation_attempt = attempt + 1
                            break
                    except Exception as ex:
                        last_rejection = str(ex)

                    logging.warning(
                        "Rejected target %s with prompt %s, attempt %d/%d: %s",
                        target_cut.id,
                        prompt_cut.id,
                        attempt + 1,
                        args.max_generation_attempts,
                        last_rejection,
                    )

                if target_tokens is None:
                    stats["generation_rejected"] += 1
                    logging.warning(
                        "Skipping target %s after %d attempts: %s",
                        target_cut.id,
                        args.max_generation_attempts,
                        last_rejection,
                    )
                    continue

                # Save only accepted VALL-E target tokens. Prompt codes are never
                # written as training features; their provenance is stored as attrs.
                dataset = h5_file.create_dataset(
                    cut_key, data=target_tokens, compression="gzip"
                )
                dataset.attrs["prompt_cut_id"] = prompt_cut.id
                dataset.attrs["prompt_frames"] = prompt_frames
                dataset.attrs["prompt_audio_path"] = str(prompt_audio_path)
                dataset.attrs["prompt_token_source"] = (
                    "single_wav_speechtokenizer_encode"
                )
                dataset.attrs["generation_attempt"] = generation_attempt
                h5_file.flush()
                existing_keys.add(cut_key)
                gen_codes_np = target_tokens
                stats["generated"] += 1

            # Construct Lhotse Cut referencing generated tokens, target text, and GT recording
            # SpeechTokenizer strides are 8*5*4*2=320 samples at 16 kHz: 50 fps.
            gen_duration = float(gen_codes_np.shape[0] * 0.02)
            feat = Features(
                type="valle_native",
                num_frames=gen_codes_np.shape[0],
                num_features=8,
                frame_shift=0.02,
                sampling_rate=16000,
                start=0.0,
                duration=float(gen_codes_np.shape[0] * 0.02),
                storage_type="numpy_hdf5",
                storage_path=str(out_h5_path),
                storage_key=cut_key,
            )

            # Keep target supervision and audio recording for GT comparison
            new_supervision = target_cut.supervisions[0] if target_cut.supervisions else None
            new_cut = MonoCut(
                id=cut_key,
                start=0.0,
                duration=gen_duration,
                channel=0,
                features=feat,
                recording=target_cut.recording if target_cut.has_recording else None,
                supervisions=[new_supervision] if new_supervision else [],
                custom={
                    "prompt_cut_id": prompt_cut.id,
                    "target_cut_id": target_cut.id,
                    "speaker": prompt_cut.supervisions[0].speaker if prompt_cut.supervisions else "unknown",
                    "generation_version": 4,
                    "valle_precision": args.precision,
                    "ar_sampling_device": "cpu" if args.sample_on_cpu else device.type,
                    "prompt_token_source": "single_wav_speechtokenizer_encode",
                    "prompt_audio_path": str(prompt_audio_path),
                    "valle_checkpoint_path": str(valle_checkpoint_path),
                    "st_config_sha256": st_config_sha256,
                    "st_checkpoint_path": str(st_checkpoint_path),
                    "text_tokens_sha256": vocab_sha256,
                    "prompt_frames": prompt_frames,
                    "generation_attempt": generation_attempt,
                    "generated_to_target_duration_ratio": gen_duration
                    / target_cut.duration,
                }
            )
            generated_cuts.append(new_cut)
            stats["manifest_cuts"] += 1

            if preview_dir is not None and stats["previews"] < args.max_previews:
                try:
                    save_preview(
                        preview_dir,
                        stats["previews"],
                        prompt_cut,
                        target_cut,
                        gen_codes_np,
                        st_model,
                        device,
                        generation_attempt,
                    )
                    stats["previews"] += 1
                except Exception as ex:
                    stats["preview_errors"] += 1
                    logging.warning(
                        "Could not save preview for target %s: %s",
                        target_cut.id,
                        ex,
                    )

    h5_file.close()
    logging.info(f"Saving output manifest to {out_manifest_path} ({len(generated_cuts)} cuts)...")
    CutSet.from_cuts(generated_cuts).to_file(out_manifest_path)
    logging.info(f"Done! Saved {len(generated_cuts)} cuts to {out_manifest_path}")
    logging.info("Generation statistics: %s", dict(sorted(stats.items())))


if __name__ == "__main__":
    main()

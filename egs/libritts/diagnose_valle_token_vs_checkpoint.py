#!/usr/bin/env python3
"""Minimal diagnosis: SpeechTokenizer preparation vs. VALL-E checkpoint.

The script runs three controls on one same-speaker prompt/target pair:

1. Compare the prompt codes stored in the Lhotse H5 manifest with codes made by
   encoding exactly the same waveform as a single-item SpeechTokenizer batch.
2. Measure AR teacher-forcing NLL/accuracy/EOS behavior on stored target codes.
3. Run free synthesis twice, changing only the prompt codes (H5 vs. fresh).

The JSON verdict deliberately reports token and checkpoint failures separately,
because both problems may be present at the same time.
"""

import argparse
import hashlib
import json
import logging
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

# Reuse the smoke test's dependency setup and path-resolution helpers so this
# diagnostic exercises exactly the same model/text/tokenizer implementation.
import test_valle_token_synthesis as common


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distinguish SpeechTokenizer token-preparation errors from VALL-E checkpoint errors."
    )
    parser.add_argument("--valle-checkpoint", default="exp/valle_voicemark/epoch-40.pt")
    parser.add_argument("--manifest", default="data/tokenized_voicemark/cuts_dev.jsonl.gz")
    parser.add_argument(
        "--prompt-cut-id",
        default="1462_170142_000021_000003-163",
        help="Default is a 2.00-second prompt from the existing smoke test.",
    )
    parser.add_argument(
        "--target-cut-id",
        default="1462_170142_000038_000001-219",
        help='Default is the 0.83-second target "I was wrong.".',
    )
    parser.add_argument(
        "--text-tokens",
        default=None,
        help="Optional override; by default use the vocabulary recorded in the checkpoint.",
    )
    parser.add_argument(
        "--st-config",
        default="STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json",
    )
    parser.add_argument(
        "--st-checkpoint", default="STmodels/pretrained_model/SpeechTokenizer.pt"
    )
    parser.add_argument("--output-dir", default="exp/diagnose_valle_token_vs_checkpoint")
    parser.add_argument("--top-k", type=int, default=-100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16", "fp16"],
        default="bf16",
        help="VALL-E inference precision. The historical lab infer.py used fp32.",
    )
    parser.add_argument(
        "--sample-on-cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep VALL-E on GPU but run the 1025-way AR multinomial on CPU.",
    )
    parser.add_argument("--min-frame-ratio", type=float, default=0.5)
    parser.add_argument("--max-frame-ratio", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def streamed_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def amp_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def resolve_cut_audio(cut) -> Path:
    source = Path(cut.recording.sources[0].source).expanduser()
    if source.is_file():
        return source.resolve()

    source_text = str(source)
    marker = "LibriTTS/"
    relative = Path(source_text.split(marker, 1)[1]) if marker in source_text else None
    workspace_root = common.PROJECT_DIR.parents[1]
    candidates = [
        workspace_root / "dataset" / "libriTTS" / "LibriTTS",
        workspace_root / "dataset" / "LibriTTS",
        SCRIPT_DIR / "download" / "LibriTTS",
    ]
    if relative is not None:
        for root in candidates:
            candidate = root / relative
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot relocate audio for cut {cut.id}: original source={source}"
    )


def load_cut_waveform(cut, sample_rate: int, device: torch.device) -> torch.Tensor:
    audio_path = resolve_cut_audio(cut)
    waveform, source_rate = torchaudio.load(str(audio_path))
    start_sample = round(cut.start * source_rate)
    num_samples = round(cut.duration * source_rate)
    waveform = waveform[:, start_sample : start_sample + num_samples].float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, source_rate, sample_rate
        )
    return waveform.unsqueeze(0).to(device)  # [1, 1, samples]


def normalize_st_codes(codes: torch.Tensor) -> torch.Tensor:
    """Return SpeechTokenizer codes in VALL-E layout [1, frames, 8]."""
    while isinstance(codes, (list, tuple)):
        codes = codes[0]
    if not isinstance(codes, torch.Tensor) or codes.ndim != 3:
        raise ValueError(f"Unexpected SpeechTokenizer output: {type(codes)}")
    if codes.shape[0] == 8:  # [8, B, T]
        codes = codes.permute(1, 2, 0)
    elif codes.shape[1] == 8:  # [B, 8, T]
        codes = codes.permute(0, 2, 1)
    elif codes.shape[2] != 8:
        raise ValueError(f"Cannot infer code layout from {tuple(codes.shape)}")
    if codes.shape[0] != 1 or codes.shape[2] != 8:
        raise ValueError(f"Expected normalized codes [1, T, 8], got {tuple(codes.shape)}")
    return codes.long().contiguous()


def compare_codes(stored: torch.Tensor, fresh: torch.Tensor) -> Dict:
    stored_cpu = stored.detach().cpu()
    fresh_cpu = fresh.detach().cpu()
    common_frames = min(stored_cpu.shape[1], fresh_cpu.shape[1])
    if common_frames == 0:
        raise ValueError("Cannot compare empty token sequences")
    matches = stored_cpu[:, :common_frames] == fresh_cpu[:, :common_frames]
    per_quantizer = matches.float().mean(dim=(0, 1)).tolist()
    all_layers_per_frame = matches.all(dim=2).float().mean().item()
    exact = bool(
        stored_cpu.shape == fresh_cpu.shape and torch.equal(stored_cpu, fresh_cpu)
    )
    return {
        "stored_frames": int(stored_cpu.shape[1]),
        "fresh_frames": int(fresh_cpu.shape[1]),
        "common_frames": int(common_frames),
        "exactly_equal": exact,
        "overall_token_match": float(matches.float().mean().item()),
        "all_8_layers_frame_match": float(all_layers_per_frame),
        "per_quantizer_match": [float(value) for value in per_quantizer],
        "q0_match": float(per_quantizer[0]),
    }


def teacher_forcing_diagnostic(
    model,
    text_ids: torch.Tensor,
    text_lens: torch.Tensor,
    target_codes: torch.Tensor,
    device: torch.device,
    precision: str,
) -> Dict:
    captured = {}

    def capture_logits(_module, _inputs, output):
        captured["logits"] = output.detach().float().cpu()

    handle = model.ar_predict_layer.register_forward_hook(capture_logits)
    try:
        target_lens = torch.tensor(
            [target_codes.shape[1]], dtype=torch.int64, device=device
        )
        with amp_context(device, precision):
            model(
                text_ids.to(device),
                text_lens.to(device),
                target_codes,
                target_lens,
                reduction="mean",
                train_stage=1,
            )
    finally:
        handle.remove()

    logits = captured.get("logits")
    if logits is None:
        raise RuntimeError("Failed to capture AR logits during teacher forcing")
    if logits.ndim != 3:
        raise RuntimeError(f"Unexpected AR logits shape: {tuple(logits.shape)}")

    eos_id = logits.shape[-1] - 1
    frames = target_codes.shape[1]
    y_mask = torch.zeros((1, frames), dtype=torch.int64, device=device)
    _, labels = model.pad_y_eos(target_codes[..., 0], y_mask, eos_id=eos_id)
    labels = labels.detach().cpu().long()
    if logits.shape[:2] != labels.shape:
        raise RuntimeError(
            f"Teacher-forcing shape mismatch: logits={tuple(logits.shape)}, "
            f"labels={tuple(labels.shape)}"
        )

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    nll = float(F.cross_entropy(flat_logits, flat_labels, reduction="mean").item())
    predictions = logits.argmax(dim=-1)
    top1 = float((predictions == labels).float().mean().item())
    top10_ids = logits.topk(k=min(10, logits.shape[-1]), dim=-1).indices
    top10 = float((top10_ids == labels.unsqueeze(-1)).any(dim=-1).float().mean().item())

    eos_positions = (labels == eos_id).nonzero(as_tuple=False)
    if eos_positions.shape[0] != 1:
        raise RuntimeError(f"Expected one EOS target, found {eos_positions.shape[0]}")
    eos_position = int(eos_positions[0, 1].item())
    eos_logits = logits[0, eos_position]
    eos_probability = float(eos_logits.softmax(dim=-1)[eos_id].item())
    eos_rank = int((eos_logits > eos_logits[eos_id]).sum().item() + 1)

    non_eos = labels != eos_id
    premature_eos_argmax = float(
        ((predictions == eos_id) & non_eos).sum().item() / max(1, non_eos.sum().item())
    )
    random_nll = math.log(logits.shape[-1])
    # These thresholds only identify clearly random-like or clearly learned AR
    # behavior; intermediate results are deliberately marked inconclusive.
    if nll >= random_nll * 0.95 or top10 <= 0.02:
        learning_state = "RANDOM_LIKE_OR_WRONG_CHECKPOINT"
    elif nll <= random_nll * 0.75 and top10 >= 0.20:
        learning_state = "AR_HAS_LEARNED_STORED_TOKENS"
    else:
        learning_state = "INCONCLUSIVE"

    return {
        "num_frames": int(frames),
        "audio_vocab_size_including_eos": int(logits.shape[-1]),
        "nll": nll,
        "perplexity": float(math.exp(min(nll, 20.0))),
        "uniform_random_nll": float(random_nll),
        "top1_accuracy": top1,
        "top10_accuracy": top10,
        "true_eos_position": eos_position,
        "true_eos_probability": eos_probability,
        "true_eos_rank": eos_rank,
        "premature_eos_argmax_rate": premature_eos_argmax,
        "learning_state": learning_state,
    }


def free_synthesis(
    name: str,
    model,
    st_model,
    text_ids: torch.Tensor,
    text_lens: torch.Tensor,
    enroll_lens: torch.Tensor,
    prompt_codes: torch.Tensor,
    reference_frames: int,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> Tuple[Dict, torch.Tensor]:
    # Reset the RNG so H5/fresh comparisons use the same sampling stream.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    with amp_context(device, args.precision):
        generated = model.inference(
            text_ids.to(device),
            text_lens.to(device),
            prompt_codes,
            enroll_x_lens=enroll_lens.to(device),
            top_k=args.top_k,
            temperature=args.temperature,
            sample_on_cpu=args.sample_on_cpu,
        )
    if isinstance(generated, tuple):
        generated = generated[0]
    if generated.ndim != 3 or generated.shape[2] != 8:
        raise RuntimeError(f"Unexpected VALL-E output shape: {tuple(generated.shape)}")

    generated_frames = int(generated.shape[1])
    ratio = generated_frames / max(1, reference_frames)
    length_ok = bool(
        generated_frames > 0
        and args.min_frame_ratio <= ratio <= args.max_frame_ratio
    )
    token_path = output_dir / f"generated_{name}_prompt.npy"
    np.save(token_path, generated[0].detach().cpu().numpy().astype(np.int16))
    wav_path = output_dir / f"generated_{name}_prompt.wav"
    if generated_frames:
        common.save_wav(
            wav_path,
            common.decode_codes(st_model, generated),
            st_model.sample_rate,
        )

    return {
        "prompt_source": name,
        "prompt_frames": int(prompt_codes.shape[1]),
        "generated_frames": generated_frames,
        "reference_target_frames": int(reference_frames),
        "generated_to_reference_ratio": float(ratio),
        "length_check_passed": length_ok,
        "tokens": str(token_path),
        "wav": str(wav_path) if generated_frames else None,
    }, generated


def make_verdict(token_comparison: Dict, teacher: Dict, h5_run: Dict, fresh_run: Dict) -> Dict:
    token_issue = not token_comparison["exactly_equal"]
    h5_failure = not h5_run["length_check_passed"]
    fresh_failure = not fresh_run["length_check_passed"]
    teacher_random_like = teacher["learning_state"] == "RANDOM_LIKE_OR_WRONG_CHECKPOINT"

    findings = []
    if token_issue:
        findings.append("TOKEN_PREP_INCONSISTENCY")
    if teacher_random_like:
        findings.append("VALLE_AR_RANDOM_LIKE_OR_WRONG_CHECKPOINT")
    if h5_failure:
        findings.append("VALLE_FREE_RUN_FAILURE_WITH_H5_PROMPT")
    if not h5_failure and fresh_failure:
        findings.append("FRESH_PROMPT_TOKEN_MISMATCH_BREAKS_INFERENCE")
    if h5_failure and not fresh_failure:
        findings.append("H5_PROMPT_TOKEN_MISMATCH_BREAKS_INFERENCE")
    if (
        h5_failure
        and fresh_failure
        and teacher["learning_state"] == "AR_HAS_LEARNED_STORED_TOKENS"
    ):
        findings.append("AR_LEARNED_BUT_FREE_RUNNING_OR_EOS_FAILS")
    if not findings:
        findings.append("NO_FAILURE_DETECTED_BY_MINIMAL_TEST")

    if token_issue and h5_failure != fresh_failure and not teacher_random_like:
        primary = "TOKEN_PIPELINE_IS_PRIMARY"
    elif token_issue and (teacher_random_like or (h5_failure and fresh_failure)):
        primary = "BOTH_TOKEN_AND_VALLE_ISSUES_DETECTED"
    elif teacher_random_like or h5_failure:
        primary = "VALLE_CHECKPOINT_OR_FREE_INFERENCE_IS_PRIMARY"
    elif token_issue:
        primary = "TOKEN_INCONSISTENCY_DETECTED_BUT_SYNTHESIS_LENGTH_PASSED"
    else:
        primary = "NO_CLEAR_FAILURE"

    return {
        "primary": primary,
        "findings": findings,
        "token_problem_confirmed": token_issue,
        "valle_h5_prompt_free_run_failed": h5_failure,
        "valle_fresh_prompt_free_run_failed": fresh_failure,
        "teacher_forcing_state": teacher["learning_state"],
        "note": (
            "A length pass does not prove intelligibility; listen to both generated WAVs. "
            "A failure with both prompt sources cannot be explained only by fresh-vs-H5 mismatch."
        ),
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")

    checkpoint_path = common.resolve_file(args.valle_checkpoint, [SCRIPT_DIR, common.PROJECT_DIR])
    manifest_path = common.resolve_file(args.manifest, [SCRIPT_DIR, common.PROJECT_DIR])
    st_config_path = common.resolve_file(args.st_config, [common.NEUMARK_ROOT, SCRIPT_DIR])
    st_checkpoint_path = common.resolve_file(
        args.st_checkpoint, [common.NEUMARK_ROOT, SCRIPT_DIR]
    )
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    logging.info("Hashing and loading VALL-E checkpoint: %s", checkpoint_path)
    checkpoint_hash = streamed_sha256(checkpoint_path)
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False, mmap=True
    )
    model_args = common.AttributeDict(checkpoint)
    checkpoint_vocab = getattr(model_args, "text_tokens", None)
    if not checkpoint_vocab and not args.text_tokens:
        raise ValueError("Checkpoint has no text_tokens; pass --text-tokens")
    text_tokens_path = common.resolve_file(
        args.text_tokens or checkpoint_vocab, [SCRIPT_DIR, common.PROJECT_DIR]
    )
    text_collater = common.get_text_token_collater(str(text_tokens_path))

    model = common.get_model(model_args)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    del checkpoint

    logging.info("Loading SpeechTokenizer: %s", st_checkpoint_path)
    st_model = common.SpeechTokenizer.load_from_checkpoint(
        str(st_config_path), str(st_checkpoint_path)
    ).to(device)
    st_model.eval()

    cuts = list(common.load_manifest_lazy(manifest_path))
    by_id = {cut.id: cut for cut in cuts}
    if args.prompt_cut_id not in by_id or args.target_cut_id not in by_id:
        raise KeyError("Prompt or target cut ID is absent from the manifest")
    prompt_cut = by_id[args.prompt_cut_id]
    target_cut = by_id[args.target_cut_id]
    prompt_speaker = prompt_cut.supervisions[0].speaker
    target_speaker = target_cut.supervisions[0].speaker
    if prompt_speaker != target_speaker:
        raise ValueError(f"Speaker mismatch: {prompt_speaker} != {target_speaker}")

    prompt_phonemes = common.supervision_phonemes(prompt_cut)
    target_phonemes = common.supervision_phonemes(target_cut)
    unknown = sorted(
        (set(prompt_phonemes) | set(target_phonemes) | {"_"})
        - set(text_collater.token2idx)
    )
    if unknown:
        raise ValueError(f"Phonemes absent from checkpoint vocabulary: {unknown}")

    h5_prompt = torch.from_numpy(prompt_cut.load_features()).long().unsqueeze(0).to(device)
    h5_target = torch.from_numpy(target_cut.load_features()).long().unsqueeze(0).to(device)
    prompt_waveform = load_cut_waveform(prompt_cut, st_model.sample_rate, device)
    with torch.inference_mode():
        fresh_prompt = normalize_st_codes(st_model.encode(prompt_waveform))
    comparison = compare_codes(h5_prompt, fresh_prompt)
    logging.info(
        "Token comparison: exact=%s, overall=%.4f, q0=%.4f, all8-frame=%.4f",
        comparison["exactly_equal"],
        comparison["overall_token_match"],
        comparison["q0_match"],
        comparison["all_8_layers_frame_match"],
    )

    common.save_wav(
        output_dir / "prompt_h5_tokens.wav",
        common.decode_codes(st_model, h5_prompt),
        st_model.sample_rate,
    )
    common.save_wav(
        output_dir / "prompt_fresh_tokens.wav",
        common.decode_codes(st_model, fresh_prompt),
        st_model.sample_rate,
    )
    common.save_wav(
        output_dir / "target_h5_reference.wav",
        common.decode_codes(st_model, h5_target),
        st_model.sample_rate,
    )

    target_text_ids, target_text_lens = text_collater([target_phonemes])
    with torch.inference_mode():
        teacher = teacher_forcing_diagnostic(
            model,
            target_text_ids,
            target_text_lens,
            h5_target,
            device,
            args.precision,
        )
    logging.info(
        "Teacher forcing: state=%s, NLL=%.4f (random=%.4f), top1=%.4f, "
        "top10=%.4f, EOS rank=%d, EOS p=%.6f",
        teacher["learning_state"],
        teacher["nll"],
        teacher["uniform_random_nll"],
        teacher["top1_accuracy"],
        teacher["top10_accuracy"],
        teacher["true_eos_rank"],
        teacher["true_eos_probability"],
    )

    full_text_ids, full_text_lens = text_collater(
        [prompt_phonemes + ["_"] + target_phonemes]
    )
    _, enroll_lens = text_collater([prompt_phonemes])
    with torch.inference_mode():
        h5_run, _ = free_synthesis(
            "h5",
            model,
            st_model,
            full_text_ids,
            full_text_lens,
            enroll_lens,
            h5_prompt,
            h5_target.shape[1],
            args,
            device,
            output_dir,
        )
        fresh_run, _ = free_synthesis(
            "fresh",
            model,
            st_model,
            full_text_ids,
            full_text_lens,
            enroll_lens,
            fresh_prompt,
            h5_target.shape[1],
            args,
            device,
            output_dir,
        )

    verdict = make_verdict(comparison, teacher, h5_run, fresh_run)
    report = {
        "verdict": verdict,
        "artifacts": {
            "valle_checkpoint": str(checkpoint_path),
            "valle_checkpoint_sha256": checkpoint_hash,
            "speech_tokenizer_checkpoint": str(st_checkpoint_path),
            "speech_tokenizer_checkpoint_sha256": streamed_sha256(st_checkpoint_path),
            "speech_tokenizer_config": str(st_config_path),
            "speech_tokenizer_config_sha256": streamed_sha256(st_config_path),
            "text_tokens": str(text_tokens_path),
            "text_tokens_sha256": streamed_sha256(text_tokens_path),
            "manifest": str(manifest_path),
        },
        "model": {
            "checkpoint_train_stage": getattr(model_args, "train_stage", None),
            "prefix_mode": getattr(model_args, "prefix_mode", None),
            "top_k": args.top_k,
            "temperature": args.temperature,
            "precision": args.precision,
            "sample_on_cpu": args.sample_on_cpu,
            "seed": args.seed,
        },
        "sample": {
            "prompt_cut_id": prompt_cut.id,
            "prompt_text": prompt_cut.supervisions[0].text,
            "prompt_duration": prompt_cut.duration,
            "target_cut_id": target_cut.id,
            "target_text": target_cut.supervisions[0].text,
            "target_duration": target_cut.duration,
            "speaker": prompt_speaker,
        },
        "token_comparison": comparison,
        "teacher_forcing": teacher,
        "free_synthesis": {"h5_prompt": h5_run, "fresh_prompt": fresh_run},
    }
    report_path = output_dir / "diagnosis.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print("\n" + "=" * 80)
    print("VALL-E token vs checkpoint diagnosis")
    print("=" * 80)
    print(f"PRIMARY : {verdict['primary']}")
    for finding in verdict["findings"]:
        print(f"FINDING : {finding}")
    print(
        f"TOKEN   : exact={comparison['exactly_equal']} "
        f"q0_match={comparison['q0_match']:.4f} "
        f"all_match={comparison['overall_token_match']:.4f}"
    )
    print(
        f"AR-TF   : {teacher['learning_state']} nll={teacher['nll']:.4f} "
        f"top10={teacher['top10_accuracy']:.4f} eos_rank={teacher['true_eos_rank']}"
    )
    print(
        f"FREE-H5 : frames={h5_run['generated_frames']} "
        f"ratio={h5_run['generated_to_reference_ratio']:.3f} "
        f"pass={h5_run['length_check_passed']}"
    )
    print(
        f"FREE-NEW: frames={fresh_run['generated_frames']} "
        f"ratio={fresh_run['generated_to_reference_ratio']:.3f} "
        f"pass={fresh_run['length_check_passed']}"
    )
    print(f"REPORT  : {report_path}")
    print("Listen to generated_h5_prompt.wav and generated_fresh_prompt.wav if present.")
    print("=" * 80)

    # A diagnostic finding should not make the batch job look like an execution
    # failure; non-zero is reserved for exceptions or invalid inputs.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

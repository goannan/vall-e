#!/usr/bin/env python3
"""English Seed-TTS WER with the official Whisper-large-v3 protocol."""

import argparse
import json
import string
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from tqdm import tqdm
from zhon.hanzi import punctuation as chinese_punctuation


def normalize_english(text: str) -> list[str]:
    # Match the released Seed-TTS evaluator: lowercase and remove punctuation,
    # except apostrophes (e.g. don't remains one word).
    punctuation = set(chinese_punctuation + string.punctuation)
    punctuation.discard("'")
    normalized = "".join("" if char in punctuation else char for char in text)
    normalized = normalized.replace("  ", " ")
    return normalized.lower().split(" ")


def edit_counts(reference: list[str], hypothesis: list[str]):
    # Each cell stores (total errors, substitutions, deletions, insertions).
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, ref_word in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, hyp_word in enumerate(hypothesis, 1):
            if ref_word == hyp_word:
                current.append(previous[j - 1])
                continue
            sub = previous[j - 1]
            delete = previous[j]
            insert = current[j - 1]
            candidates = (
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            )
            current.append(min(candidates))
        previous = current
    return previous[-1]


def read_pairs(path: Path):
    pairs = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        fields = raw_line.split("|")
        if len(fields) < 3:
            raise ValueError(f"{path}:{line_number}: expected generated|prompt|text")
        generated, _, target = fields[0], fields[1], "|".join(fields[2:])
        generated_path = Path(generated)
        if not generated_path.is_file():
            raise FileNotFoundError(generated_path)
        pairs.append((generated_path, target))
    return pairs


def load_audio(path: Path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sample_rate != 16000:
        divisor = np.gcd(sample_rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, sample_rate // divisor)
    return audio.astype(np.float32, copy=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="openai/whisper-large-v3")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Whisper-large-v3 evaluation requires a CUDA GPU.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    pairs = read_pairs(args.pairs.expanduser().resolve())
    if not pairs:
        raise RuntimeError(f"No evaluation pairs in {args.pairs}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    ).to("cuda").eval()
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="english", task="transcribe"
    )

    details = []
    total_ref_words = 0
    total_errors = 0
    for start in tqdm(range(0, len(pairs), args.batch_size), desc="Whisper WER"):
        batch = pairs[start : start + args.batch_size]
        audio = [load_audio(path) for path, _ in batch]
        features = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            # Whisper's encoder requires exactly 30 seconds / 3000 mel
            # frames. ``padding=True`` only pads to the longest item in the
            # current batch and produces a shorter, invalid tensor here.
            padding="max_length",
            max_length=processor.feature_extractor.n_samples,
            truncation=True,
        ).input_features.to("cuda", dtype=torch.float16)
        with torch.inference_mode():
            predicted_ids = model.generate(
                features,
                forced_decoder_ids=forced_decoder_ids,
            )
        hypotheses = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        for (wav_path, target), hypothesis in zip(batch, hypotheses):
            reference_words = normalize_english(target)
            hypothesis_words = normalize_english(hypothesis)
            errors, substitutions, deletions, insertions = edit_counts(
                reference_words, hypothesis_words
            )
            reference_count = len(reference_words)
            wer = errors / reference_count if reference_count else 0.0
            total_ref_words += reference_count
            total_errors += errors
            details.append(
                {
                    "wav": str(wav_path),
                    "reference": target,
                    "hypothesis": hypothesis,
                    "reference_words": reference_count,
                    "errors": errors,
                    "substitutions": substitutions,
                    "deletions": deletions,
                    "insertions": insertions,
                    "wer": wer,
                }
            )

    macro_wer = sum(item["wer"] for item in details) / len(details)
    summary = {
        "count": len(details),
        "model": args.model,
        "protocol": "Seed-TTS English WER (Whisper-large-v3; utterance-macro average)",
        "macro_wer": macro_wer,
        "macro_wer_percent": macro_wer * 100,
        "corpus_wer": total_errors / total_ref_words if total_ref_words else 0.0,
        "corpus_wer_percent": (
            total_errors / total_ref_words * 100 if total_ref_words else 0.0
        ),
        "total_reference_words": total_ref_words,
        "total_errors": total_errors,
    }
    (args.output_dir / "wer_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "wer_details.jsonl").open("w", encoding="utf-8") as stream:
        for item in details:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run one VALL-E synthesis with the preserved lab-era runtime and source tree.

This intentionally avoids autocast and uses fresh SpeechTokenizer prompt codes.
The launcher puts ``src/valle`` first on PYTHONPATH and activates the historical
Python 3.10 / torch 1.13 environment.
"""

import argparse
import json
import random
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
import torchaudio


def install_optional_import_stubs():
    """The preserved checkout lost TraceableSpeech .py files; inference does not use them."""
    names = [
        "traceableSpeech",
        "traceableSpeech.env",
        "traceableSpeech.meldataset",
        "traceableSpeech.models",
        "traceableSpeech.watermark",
    ]
    for name in names:
        module = types.ModuleType(name)
        if name == "traceableSpeech":
            module.__path__ = []
        sys.modules[name] = module

    class Unused:
        pass

    sys.modules["traceableSpeech.env"].AttrDict = Unused
    sys.modules["traceableSpeech.meldataset"].mel_spectrogram = Unused
    for name in ("Generator", "Encoder", "Quantizer"):
        setattr(sys.modules["traceableSpeech.models"], name, Unused)
    for name in ("Watermark_Encoder", "Watermark_Decoder", "Random_watermark"):
        setattr(sys.modules["traceableSpeech.watermark"], name, Unused)


install_optional_import_stubs()

# The modern ``voicemark`` environment used for the cross test does not have
# phonemizer installed.  This script consumes prepared phoneme tokens and never
# constructs TextTokenizer, so lightweight import stubs are sufficient there.
try:
    import phonemizer  # noqa: F401
except ImportError:
    import importlib.machinery

    for name in [
        "phonemizer",
        "phonemizer.backend",
        "phonemizer.backend.espeak",
        "phonemizer.backend.espeak.language_switch",
        "phonemizer.backend.espeak.words_mismatch",
        "phonemizer.punctuation",
        "phonemizer.separator",
    ]:
        module = MagicMock()
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
        sys.modules[name] = module

for name in [
    "k2",
    "k2.version",
    "kaldialign",
    "pypinyin",
    "pypinyin.contrib",
    "pypinyin.contrib.tone_convert",
]:
    if name not in sys.modules:
        import importlib.machinery

        module = MagicMock()
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
        sys.modules[name] = module

from lhotse import load_manifest_lazy  # noqa: E402
from STmodels.model import SpeechTokenizer  # noqa: E402
from valle.data.collation import get_text_token_collater  # noqa: E402
from valle.models import get_model  # noqa: E402


class AttributeDict(dict):
    """Minimal checkpoint-argument view; avoids importing unused icefall/k2."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--text-tokens", required=True)
    parser.add_argument("--prompt-cut-id", required=True)
    parser.add_argument("--target-cut-id", required=True)
    parser.add_argument("--st-config", required=True)
    parser.add_argument("--st-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--top-k", type=int, default=-100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample-on-cpu", action="store_true")
    return parser.parse_args()


def resolve_audio(cut):
    source = Path(cut.recording.sources[0].source)
    if source.is_file():
        return source
    marker = "LibriTTS/"
    if marker in str(source):
        relative = Path(str(source).split(marker, 1)[1])
        root = Path("/home/pj25001109/ku60000344/dataset/libriTTS/LibriTTS")
        candidate = root / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Cannot relocate audio for {}: {}".format(cut.id, source))


def load_audio(cut, sample_rate, device):
    waveform, source_rate = torchaudio.load(str(resolve_audio(cut)))
    start = round(cut.start * source_rate)
    count = round(cut.duration * source_rate)
    waveform = waveform[:, start : start + count].float()
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    return waveform.unsqueeze(0).to(device)


def normalize_codes(codes):
    while isinstance(codes, (list, tuple)):
        codes = codes[0]
    if codes.shape[0] == 8:
        codes = codes.permute(1, 2, 0)
    elif codes.shape[1] == 8:
        codes = codes.permute(0, 2, 1)
    if codes.ndim != 3 or codes.shape[0] != 1 or codes.shape[2] != 8:
        raise ValueError("Unexpected SpeechTokenizer code shape: {}".format(tuple(codes.shape)))
    return codes.long().contiguous()


def save_wav(path, waveform, sample_rate):
    audio = waveform.detach().float().cpu()
    if audio.ndim == 3:
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(str(path), audio, sample_rate)


def decode(st_model, codes):
    return st_model.decode(codes.permute(2, 0, 1).long().contiguous())


def main():
    args = get_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    try:
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )
    except TypeError:  # torch 1.13 has no weights_only argument.
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = AttributeDict(checkpoint)
    collater = get_text_token_collater(args.text_tokens)
    model = get_model(model_args)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    del checkpoint

    st_model = SpeechTokenizer.load_from_checkpoint(
        args.st_config, args.st_checkpoint
    ).to(device)
    st_model.eval()

    wanted = {args.prompt_cut_id, args.target_cut_id}
    cuts = {cut.id: cut for cut in load_manifest_lazy(args.manifest) if cut.id in wanted}
    prompt_cut = cuts[args.prompt_cut_id]
    target_cut = cuts[args.target_cut_id]
    prompt_phones = list(prompt_cut.supervisions[0].custom["tokens"]["text"])
    target_phones = list(target_cut.supervisions[0].custom["tokens"]["text"])
    text_ids, text_lens = collater([prompt_phones + ["_"] + target_phones])
    _, enroll_lens = collater([prompt_phones])

    with torch.no_grad():
        prompt_audio = load_audio(prompt_cut, st_model.sample_rate, device)
        prompt_codes = normalize_codes(st_model.encode(prompt_audio))
        inference_kwargs = {
            "enroll_x_lens": enroll_lens.to(device),
            "top_k": args.top_k,
            "temperature": args.temperature,
        }
        if args.sample_on_cpu:
            inference_kwargs["sample_on_cpu"] = True
        generated = model.inference(
            text_ids.to(device),
            text_lens.to(device),
            prompt_codes,
            **inference_kwargs
        )
        if isinstance(generated, tuple):
            generated = generated[0]

        save_wav(output_dir / "prompt_fresh_codec.wav", decode(st_model, prompt_codes), st_model.sample_rate)
        if generated.shape[1] > 0:
            save_wav(output_dir / "generated_legacy_fp32.wav", decode(st_model, generated), st_model.sample_rate)

    np.save(
        str(output_dir / "generated_legacy_fp32.npy"),
        generated[0].detach().cpu().numpy().astype(np.int16),
    )
    report = {
        "python": sys.version,
        "torch": torch.__version__,
        "valle_module": sys.modules["valle"].__file__,
        "precision": "fp32",
        "sample_on_cpu": args.sample_on_cpu,
        "prompt_text": prompt_cut.supervisions[0].text,
        "target_text": target_cut.supervisions[0].text,
        "prompt_frames": int(prompt_codes.shape[1]),
        "generated_frames": int(generated.shape[1]),
        "seed": args.seed,
        "top_k": args.top_k,
        "temperature": args.temperature,
    }
    with (output_dir / "legacy_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

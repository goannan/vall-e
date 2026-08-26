#!/usr/bin/env python3
"""Re-encode one LibriTTS cut manifest with SpeechTokenizer on GENKAI.

Text, supervision, and recording metadata are retained from an existing
manifest.  Previous acoustic features are never read.  Every cut waveform is
loaded, downmixed/resampled, and encoded independently so batch padding cannot
change its codec tokens.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import torchaudio
from lhotse import CutSet, load_manifest_lazy
from lhotse.features import Features
from lhotse.utils import compute_num_frames, fastcopy
from STmodels.model import SpeechTokenizer
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[3]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--text-tokens", required=True)
    parser.add_argument("--st-config", required=True)
    parser.add_argument("--st-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preview-dir", default=None)
    parser.add_argument("--max-previews", type=int, default=0)
    parser.add_argument("--flush-interval", type=int, default=100)
    parser.add_argument("--max-cuts", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--min-duration", type=float, default=0.0)
    parser.add_argument("--max-duration", type=float, default=float("inf"))
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_audio(cut):
    source = Path(cut.recording.sources[0].source).expanduser()
    if source.is_file():
        return source.resolve()

    marker = "LibriTTS/"
    relative = Path(str(source).split(marker, 1)[1]) if marker in str(source) else None
    roots = [
        WORKSPACE / "dataset" / "libriTTS" / "LibriTTS",
        WORKSPACE / "dataset" / "LibriTTS",
        SCRIPT_DIR / "download" / "LibriTTS",
    ]
    if relative is not None:
        for root in roots:
            candidate = root / relative
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError("Cannot relocate audio for {}: {}".format(cut.id, source))


def load_waveform(cut, sample_rate, device):
    path = resolve_audio(cut)
    waveform, source_rate = torchaudio.load(str(path))
    start = round(cut.start * source_rate)
    count = round(cut.duration * source_rate)
    waveform = waveform[:, start : start + count].float()
    if waveform.shape[-1] == 0:
        raise ValueError("Empty waveform for {}".format(cut.id))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    return waveform.unsqueeze(0).to(device), path


def normalize_codes(codes):
    while isinstance(codes, (list, tuple)):
        codes = codes[0]
    if codes.ndim != 3:
        raise ValueError("Unexpected SpeechTokenizer output shape: {}".format(codes.shape))
    if codes.shape[0] == 8:
        codes = codes.permute(1, 2, 0)
    elif codes.shape[1] == 8:
        codes = codes.permute(0, 2, 1)
    if codes.shape[0] != 1 or codes.shape[2] != 8:
        raise ValueError("Expected [1, frames, 8], got {}".format(tuple(codes.shape)))
    return codes.long().contiguous()


def read_vocabulary(path):
    symbols = set()
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            line = line.rstrip("\n")
            if not line:
                continue
            symbol, _index = line.rsplit(maxsplit=1)
            symbols.add(symbol)
    return symbols


def save_preview(preview_dir, index, cut, waveform, codes, st_model):
    item_dir = preview_dir / "sample_{:02d}".format(index)
    item_dir.mkdir(parents=True, exist_ok=True)
    reference = waveform.detach().float().cpu()[0]
    reconstructed = st_model.decode(codes.permute(2, 0, 1)).detach().float().cpu()[0]
    torchaudio.save(str(item_dir / "00_reference.wav"), reference, st_model.sample_rate)
    torchaudio.save(
        str(item_dir / "01_speechtokenizer_reconstruction.wav"),
        reconstructed,
        st_model.sample_rate,
    )
    metadata = {
        "cut_id": cut.id,
        "text": cut.supervisions[0].text if cut.supervisions else "",
        "duration": cut.duration,
        "frames": int(codes.shape[1]),
    }
    (item_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main():
    args = get_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("Require 0 <= rank < world_size")

    input_manifest = Path(args.input_manifest).expanduser().resolve()
    output_manifest = Path(args.output_manifest).expanduser().resolve()
    output_h5 = Path(args.output_h5).expanduser().resolve()
    text_tokens = Path(args.text_tokens).expanduser().resolve()
    st_config = Path(args.st_config).expanduser().resolve()
    st_checkpoint = Path(args.st_checkpoint).expanduser().resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    vocabulary = read_vocabulary(text_tokens)
    st_model = SpeechTokenizer.load_from_checkpoint(
        str(st_config), str(st_checkpoint)
    ).to(device).eval()

    expected_attrs = {
        "generation_version": 1,
        "extractor": "SpeechTokenizer",
        "encoding_mode": "single_utterance_no_padding",
        "resampler": "torchaudio.functional.resample",
        "source_manifest_sha256": sha256(input_manifest),
        "text_tokens_sha256": sha256(text_tokens),
        "st_config_sha256": sha256(st_config),
        "st_checkpoint_sha256": sha256(st_checkpoint),
        "sample_rate": int(st_model.sample_rate),
        "frame_shift": 0.02,
        "num_quantizers": 8,
        "shard_rank": args.rank,
        "shard_world_size": args.world_size,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
    }

    preview_dir = Path(args.preview_dir).expanduser().resolve() if args.preview_dir else None
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    output_cuts = []
    generated = 0
    resumed = 0
    with h5py.File(str(output_h5), "a") as store, torch.inference_mode():
        if len(store) > 0:
            mismatches = {
                key: (store.attrs.get(key), value)
                for key, value in expected_attrs.items()
                if store.attrs.get(key) != value
            }
            if mismatches:
                raise RuntimeError("Existing output H5 has incompatible provenance: {}".format(mismatches))
        else:
            for key, value in expected_attrs.items():
                store.attrs[key] = value
            store.flush()

        cuts = load_manifest_lazy(input_manifest)
        for index, cut in enumerate(tqdm(cuts, desc=input_manifest.stem)):
            if args.max_cuts > 0 and index >= args.max_cuts:
                break
            if not args.min_duration <= cut.duration <= args.max_duration:
                continue
            if index % args.world_size != args.rank:
                continue
            if not cut.has_recording or not cut.supervisions:
                raise ValueError("Cut {} lacks recording or supervision".format(cut.id))
            phones = cut.supervisions[0].custom["tokens"]["text"]
            unknown = sorted(set(phones) - vocabulary)
            if unknown:
                raise ValueError("Cut {} has tokens absent from vocabulary: {}".format(cut.id, unknown))

            waveform = None
            if cut.id in store:
                array = store[cut.id][:]
                resumed += 1
            else:
                waveform, audio_path = load_waveform(cut, st_model.sample_rate, device)
                codes = normalize_codes(st_model.encode(waveform))
                expected_frames = compute_num_frames(
                    duration=cut.duration,
                    frame_shift=0.02,
                    sampling_rate=st_model.sample_rate,
                )
                if codes.shape[1] < expected_frames:
                    raise ValueError(
                        "Cut {} produced {} frames; expected at least {}".format(
                            cut.id, codes.shape[1], expected_frames
                        )
                    )
                codes = codes[:, :expected_frames]
                array = codes[0].cpu().numpy().astype(np.int16)
                dataset = store.create_dataset(cut.id, data=array)
                dataset.attrs["audio_path"] = str(audio_path)
                dataset.attrs["duration"] = float(cut.duration)
                generated += 1
                if generated % args.flush_interval == 0:
                    store.flush()

            if array.ndim != 2 or array.shape[1] != 8:
                raise ValueError("Bad stored shape for {}: {}".format(cut.id, array.shape))
            features = Features(
                type="speechtokenizer",
                num_frames=int(array.shape[0]),
                num_features=8,
                frame_shift=0.02,
                sampling_rate=int(st_model.sample_rate),
                start=0.0,
                duration=float(array.shape[0] * 0.02),
                storage_type="numpy_hdf5",
                storage_path=str(output_h5),
                storage_key=cut.id,
            )
            output_cuts.append(fastcopy(cut, features=features))

            if preview_dir is not None and index < args.max_previews:
                if waveform is None:
                    waveform, _ = load_waveform(cut, st_model.sample_rate, device)
                preview_codes = torch.from_numpy(array).long().to(device).unsqueeze(0)
                save_preview(preview_dir, index, cut, waveform, preview_codes, st_model)

        store.flush()

    CutSet.from_cuts(output_cuts).to_file(output_manifest)
    report = {
        "input_manifest": str(input_manifest),
        "output_manifest": str(output_manifest),
        "output_h5": str(output_h5),
        "num_cuts": len(output_cuts),
        "generated": generated,
        "resumed": resumed,
        "rank": args.rank,
        "world_size": args.world_size,
        "provenance": expected_attrs,
    }
    report_path = output_manifest.with_suffix(output_manifest.suffix + ".report.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logging.info("Prepared %d cuts (%d generated, %d resumed)", len(output_cuts), generated, resumed)
    logging.info("Report: %s", report_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Decode pre-generated VALL-E native tokens directly into audio (WAV) files
using the Neural Audio Codec (SpeechTokenizer/AudioTokenizer).
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import argparse
import logging
import sys
from pathlib import Path
import torch
import torchaudio

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent.parent
NEUMARK_ROOT = (SCRIPT_DIR.parent.parent.parent / "NeuMark").resolve()
if not NEUMARK_ROOT.exists():
    NEUMARK_ROOT = Path("/home/wu25/mrnas04home/projects/NeuMark").resolve()

for p in [str(PROJECT_DIR), str(SCRIPT_DIR), str(NEUMARK_ROOT), str(NEUMARK_ROOT / "train")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from lhotse import CutSet
from valle.data.tokenizer import AudioTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Decode acoustic tokens to WAV audio.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/tokenized_voicemark/cuts_dev_valle_native.jsonl.gz",
        help="Path to manifest containing VALL-E native cuts",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp/valle_native_samples",
        help="Directory to save synthesized WAV files",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to decode and save",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run audio codec on",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading AudioTokenizer (SpeechTokenizer) on {device}...")
    audio_tokenizer = AudioTokenizer(
        device=device,
        watermark_backend="neumark",
        voicemark_root=str(NEUMARK_ROOT),
        voicemark_config=str(NEUMARK_ROOT / "STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json"),
        voicemark_st_checkpoint=str(NEUMARK_ROOT / "STmodels/pretrained_model/SpeechTokenizer.pt"),
    )

    logging.info(f"Loading cuts from {args.manifest}...")
    cuts = CutSet.from_file(args.manifest)
    total_cuts = len(cuts)
    logging.info(f"Total available cuts in manifest: {total_cuts}")

    num_to_process = min(args.num_samples, total_cuts)
    logging.info(f"Decoding first {num_to_process} samples...")

    print("=" * 80)
    print(f"{'IDX':<4} | {'SPEAKER':<8} | {'DUR(s)':<6} | {'TEXT'}")
    print("=" * 80)

    for i in range(num_to_process):
        cut = cuts[i]
        spk = cut.custom.get("speaker", "unknown")
        text = cut.supervisions[0].text if cut.supervisions else "N/A"
        dur = cut.duration
        print(f"{i+1:<4} | {spk:<8} | {dur:<6.2f} | {text}")

        # Load 8-layer tokens [T, 8]
        feat_np = cut.load_features() # shape: [T, 8]
        
        # AudioTokenizer decode expects frames: [(codes, scale)] with codes [B, 8, T]
        tokens_tensor = torch.from_numpy(feat_np).long().to(device)  # [T, 8]
        tokens_tensor = tokens_tensor.unsqueeze(0).transpose(1, 2)   # [1, 8, T]
        frames = [(tokens_tensor, None)]

        with torch.no_grad():
            clean_wav = audio_tokenizer.decode(frames, watermark_sign=None).squeeze(0).cpu()
            if clean_wav.dim() == 1:
                clean_wav = clean_wav.unsqueeze(0)

        # Save synthesized wav (sample rate is 16kHz for SpeechTokenizer)
        synth_path = out_dir / f"sample_{i+1:02d}_spk{spk}_valle_synthesized.wav"
        torchaudio.save(str(synth_path), clean_wav, 16000)

        # Also save GT audio if available for reference
        gt_path = out_dir / f"sample_{i+1:02d}_spk{spk}_ground_truth.wav"
        if cut.has_recording:
            try:
                # Load full raw recording safely
                gt_audio = torch.from_numpy(cut.recording.load_audio()).float()
                if cut.recording.sampling_rate != 16000:
                    gt_audio = torchaudio.functional.resample(gt_audio, cut.recording.sampling_rate, 16000)
                torchaudio.save(str(gt_path), gt_audio, 16000)
            except Exception as e:
                logging.warning(f"Could not save GT recording for {cut.id}: {e}")

    print("=" * 80)
    logging.info(f"Done! Successfully synthesized {num_to_process} audio samples.")
    logging.info(f"Output directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

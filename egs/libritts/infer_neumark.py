#!/usr/bin/env python3
"""
NeuMark Interactive TTS Synthesis and Listening Demonstration Script.

Generates:
1. clean.wav        - Clean speech reconstructed from VALL-E / SpeechTokenizer codes.
2. watermarked.wav  - NeuMark watermarked speech embedded with a 16-bit payload.
3. diff_x10.wav     - Residual difference amplified by 10x for acoustic transparency inspection.

And verifies:
- Watermark extraction accuracy (Bit Acc & Detected probability) using the trained WMDetector.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import torchaudio

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

# NeuMark dynamic path search & import
def find_neumark_root() -> Path:
    candidates = [
        os.environ.get("NEUMARK_ROOT"),
        PROJECT_DIR.parent / "NeuMark",
        PROJECT_DIR / "NeuMark",
        SCRIPT_DIR / "NeuMark",
        SCRIPT_DIR.parents[2] / "NeuMark",
    ]
    for c in candidates:
        if c and Path(c).is_dir():
            return Path(c).resolve()
    return (PROJECT_DIR.parent / "NeuMark").resolve()

NEUMARK_ROOT = find_neumark_root()
for p in [str(NEUMARK_ROOT), str(NEUMARK_ROOT / "train"), str(SCRIPT_DIR)]:
    if p in sys.path:
        sys.path.remove(p)
    if os.path.exists(p):
        sys.path.insert(0, p)

from models import WMEmbedder, WMDetector
from STmodels.model import SpeechTokenizer


def load_neumark_models(
    neumark_ckpt_path: str,
    st_config_path: str,
    st_ckpt_path: str,
    device: torch.device,
):
    print(f"[1/4] Loading SpeechTokenizer from {st_ckpt_path}...")
    with open(st_config_path) as f:
        st_cfg = json.load(f)
    st_model = SpeechTokenizer(st_cfg)
    st_state = torch.load(st_ckpt_path, map_location="cpu")
    st_model.load_state_dict(st_state)
    st_model.eval().to(device)

    print(f"[2/4] Loading NeuMark Embedder & Detector from {neumark_ckpt_path}...")
    msg_processor = WMEmbedder(
        nchunks=2,
        nchunk_size=8,
        dim=1024,
        depth=3,
        heads=8,
        dim_head=64,
        ff_mult=2,
    ).to(device)

    detector = WMDetector(
        nchunks=2,
        nchunk_size=8,
        dim=1024,
        depth=3,
        heads=8,
        dim_head=64,
        ff_mult=2,
    ).to(device)

    ckpt = torch.load(neumark_ckpt_path, map_location="cpu")
    if "msg_processor" in ckpt:
        msg_processor.load_state_dict(ckpt["msg_processor"])
        detector.load_state_dict(ckpt["detector"])
        print(f"      Successfully loaded checkpoint saved at step {ckpt.get('steps', 'N/A')}, epoch {ckpt.get('epoch', 'N/A')}.")
    else:
        msg_processor.load_state_dict(ckpt)

    msg_processor.eval()
    detector.eval()
    return st_model, msg_processor, detector


def synthesize_sample(
    codes: torch.Tensor,
    message_bits: torch.Tensor,
    st_model: SpeechTokenizer,
    msg_processor: WMEmbedder,
    detector: WMDetector,
    device: torch.device,
):
    """
    Codes: [1, 8, T] or [8, T]
    Message: [1, 16] (binary 0/1)
    """
    if codes.ndim == 2:
        codes = codes.unsqueeze(0)
    codes = codes.to(device)
    message_bits = message_bits.to(device)

    codes_qbt = codes.permute(1, 0, 2).contiguous() if codes.shape[1] == 8 else codes
    quantized_layers = [st_model.quantizer.decode(codes_qbt[k : k + 1], st=k) for k in range(8)]

    # 1. Clean synthesis
    z_clean = sum(quantized_layers)
    clean_audio = st_model.decoder(z_clean)  # [1, 1, T_samples]

    # 2. Watermarked synthesis
    watermarked_layers = [msg_processor(q, message_bits) for q in quantized_layers]
    z_wm = sum(watermarked_layers)
    wm_audio = st_model.decoder(z_wm)  # [1, 1, T_samples]

    # 3. Watermark Extraction & Detection
    embedding = st_model.forward_feature(wm_audio)
    logits, chunk_logits = detector(embedding)
    detect_prob, pred_bits, detected = detector.detect_watermark(embedding)

    bit_matches = (pred_bits.long() == message_bits.long()).sum().item()
    bit_acc = bit_matches / message_bits.numel()
    detection_acc = (logits > 0.0).float().mean().item()

    return clean_audio, wm_audio, bit_acc, detection_acc, pred_bits


def main():
    parser = argparse.ArgumentParser(description="NeuMark Speech Synthesis & Listening Demo")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to NeuMark .pt checkpoint")
    parser.add_argument("--manifest", type=str, default="data/tokenized_voicemark/cuts_dev.jsonl.gz", help="Lhotse cut manifest for testing")
    parser.add_argument("--sample_index", type=int, default=0, help="Index of sample cut from manifest to synthesize")
    parser.add_argument("--message", type=str, default="1011001110001101", help="16-bit binary watermark payload (e.g. 1011001110001101)")
    parser.add_argument("--output_dir", type=str, default="exp/neumark_listening_demo", help="Directory to save wavs")
    parser.add_argument("--st_config", type=str, default=str(NEUMARK_ROOT / "STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json"))
    parser.add_argument("--st_checkpoint", type=str, default=str(NEUMARK_ROOT / "STmodels/pretrained_model/SpeechTokenizer.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== NeuMark Listening Demo on Device: {device} ===")

    # 1. Parse message bits
    assert len(args.message) == 16 and all(c in "01" for c in args.message), "Message must be exactly 16 binary bits (0 or 1)!"
    message_tensor = torch.tensor([[int(c) for c in args.message]], dtype=torch.int64, device=device)

    # 2. Load models
    st_model, msg_processor, detector = load_neumark_models(
        neumark_ckpt_path=args.checkpoint,
        st_config_path=args.st_config,
        st_ckpt_path=args.st_checkpoint,
        device=device,
    )

    # 3. Load sample cut
    print(f"[3/4] Loading sample #{args.sample_index} from {args.manifest}...")
    from lhotse import load_manifest_lazy
    cuts = load_manifest_lazy(args.manifest)
    cut = None
    for idx, c in enumerate(cuts):
        if idx == args.sample_index:
            cut = c
            break

    if cut is None:
        raise ValueError(f"Could not find sample at index {args.sample_index} in {args.manifest}")

    codes_np = cut.load_features()
    codes = torch.from_numpy(codes_np).long().transpose(0, 1)
    text = cut.supervisions[0].text if cut.supervisions else ""
    duration = cut.duration

    print(f"      Utterance ID: {cut.id}")
    print(f"      Duration:     {duration:.2f} seconds")
    print(f"      Transcript:   \"{text}\"")

    # 4. Synthesize Clean and Watermarked Audio
    print("[4/4] Performing NeuMark synthesis & watermark extraction...")
    with torch.no_grad():
        clean_audio, wm_audio, bit_acc, det_acc, pred_bits = synthesize_sample(
            codes=codes,
            message_bits=message_tensor,
            st_model=st_model,
            msg_processor=msg_processor,
            detector=detector,
            device=device,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_path = out_dir / "clean_tts.wav"
    wm_path = out_dir / "neumark_watermarked.wav"
    diff_path = out_dir / "residual_difference_x10.wav"

    # Save audio files (16kHz)
    torchaudio.save(str(clean_path), clean_audio.squeeze(0).cpu(), 16000)
    torchaudio.save(str(wm_path), wm_audio.squeeze(0).cpu(), 16000)

    # Amplified difference for listening
    min_len = min(clean_audio.shape[-1], wm_audio.shape[-1])
    diff_wav = torch.clamp((wm_audio[..., :min_len] - clean_audio[..., :min_len]) * 10.0, -1.0, 1.0)
    torchaudio.save(str(diff_path), diff_wav.squeeze(0).cpu(), 16000)

    pred_msg_str = "".join(str(b.item()) for b in pred_bits[0])

    print("\n" + "=" * 70)
    print("  NeuMark TTS Demonstration Results")
    print("=" * 70)
    print(f"  Target Payload (16-bit):   {args.message}")
    print(f"  Extracted Payload:         {pred_msg_str}")
    print(f"  Bit Accuracy (BER):        {bit_acc * 100:.2f}% (BER: {(1-bit_acc)*100:.2f}%)")
    print(f"  Presence Detection Acc:    {det_acc * 100:.2f}%")
    print("-" * 70)
    print("  Output Audio Files Saved:")
    print(f"  [1] Clean Speech:          {clean_path}")
    print(f"  [2] Watermarked Speech:    {wm_path}")
    print(f"  [3] Residual (10x Gain):   {diff_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

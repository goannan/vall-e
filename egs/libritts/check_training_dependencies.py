#!/usr/bin/env python3
# Copyright (c) 2026
# Comprehensive Dependency & Environment Check for NeuMark / TTS-Native Training

import importlib
import os
import sys
from pathlib import Path

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
    if p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)

def check_item(name, fn):
    try:
        res = fn()
        status = "OK" if res is None or res is True else f"OK ({res})"
        print(f"  [+] {name:<35} : {status}")
        return True
    except Exception as e:
        print(f"  [-] {name:<35} : FAILED ({e})")
        return False

print("=" * 80)
print(f" Dependency & Environment Check for Watermark Training")
print(f" Python Executable : {sys.executable}")
print(f" Python Version    : {sys.version.split()[0]}")
print(f" NeuMark Root Path : {NEUMARK_ROOT}")
print("=" * 80)

print("\n--- 1. Core Deep Learning & Hardware Acceleration ---")
check_item("PyTorch", lambda: importlib.import_module("torch").__version__)
check_item("CUDA Available", lambda: importlib.import_module("torch").cuda.is_available())
check_item("GPU Count", lambda: importlib.import_module("torch").cuda.device_count())
check_item("BF16 Supported", lambda: importlib.import_module("torch").cuda.is_bf16_supported())
check_item("HuggingFace Accelerate", lambda: importlib.import_module("accelerate").__version__)
check_item("TensorBoard", lambda: importlib.import_module("torch.utils.tensorboard").__name__)

print("\n--- 2. Audio Processing & Data IO ---")
check_item("TorchAudio", lambda: importlib.import_module("torchaudio").__version__)
check_item("SoundFile", lambda: importlib.import_module("soundfile").__version__)
check_item("Lhotse", lambda: importlib.import_module("lhotse").__version__)
check_item("h5py", lambda: importlib.import_module("h5py").__version__)
check_item("Julius (DSP filtering)", lambda: importlib.import_module("julius").__version__)

print("\n--- 3. Codec & Attack Evaluation Libraries ---")
check_item("EnCodec (Facebook)", lambda: importlib.import_module("encodec").__version__)
check_item("DAC (Descript Audio Codec)", lambda: importlib.import_module("dac").__version__)
check_item("SNAC (Multi-scale Codec)", lambda: importlib.import_module("snac").__name__)

print("\n--- 4. Audio Quality Metrics ---")
check_item("PESQ (Perceptual Audio)", lambda: importlib.import_module("pesq").__name__)
check_item("PySTOI (Intelligibility)", lambda: importlib.import_module("pystoi").__name__)
check_item("TorchMetrics", lambda: importlib.import_module("torchmetrics").__version__)

print("\n--- 5. NeuMark / VoiceMark Model Architecture ---")
check_item("SpeechTokenizer Model", lambda: importlib.import_module("STmodels.model").SpeechTokenizer.__name__)
check_item("MultiScale Discriminator", lambda: importlib.import_module("STmodels.discriminators").MultiScaleDiscriminator.__name__)
check_item("MultiPeriod Discriminator", lambda: importlib.import_module("STmodels.discriminators").MultiPeriodDiscriminator.__name__)
check_item("MultiScale STFT Discriminator", lambda: importlib.import_module("STmodels.discriminators").MultiScaleSTFTDiscriminator.__name__)
check_item("WMEmbedder & WMDetector", lambda: (importlib.import_module("models").WMEmbedder.__name__, importlib.import_module("models").WMDetector.__name__))

print("\n--- 6. Training Pipeline Submodules ---")
check_item("TTS-Native Dataset Loader", lambda: importlib.import_module("tts_native_dataset").get_tts_native_dataloader.__name__)
check_item("TTS-Native Loss Modules", lambda: importlib.import_module("tts_native_loss").__name__)
check_item("TTS-Native Attacks Suite", lambda: importlib.import_module("tts_native_attacks").get_validation_attack_suite.__name__)

print("\n--- 7. Pretrained Weights Check ---")
st_ckpt = NEUMARK_ROOT / "STmodels/pretrained_model/SpeechTokenizer.pt"
st_cfg = NEUMARK_ROOT / "STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json"
check_item("SpeechTokenizer Checkpoint", lambda: f"Found ({st_ckpt.stat().st_size / 1e6:.1f} MB)" if st_ckpt.exists() else False)
check_item("SpeechTokenizer Config", lambda: f"Found" if st_cfg.exists() else False)

print("=" * 80)

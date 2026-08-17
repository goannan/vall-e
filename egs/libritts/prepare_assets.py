#!/usr/bin/env python3
"""
NeuMark Environment & Assets Readiness Diagnostic Tool.

Checks:
1. Python environment & required packages (torch, torchaudio, accelerate, lhotse, etc.)
2. SpeechTokenizer model checkpoint & configuration
3. WavLM speaker similarity checkpoint (wavlm_large_finetune.pth)
4. Lhotse data manifests (cuts_train.jsonl.gz, cuts_dev.jsonl.gz)
5. CUDA availability & GPU VRAM
"""

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]

# Dynamic NEUMARK_ROOT resolution
def get_neumark_root():
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


def check_python_packages():
    packages = [
        ("torch", "PyTorch"),
        ("torchaudio", "TorchAudio"),
        ("accelerate", "HuggingFace Accelerate"),
        ("lhotse", "Lhotse Speech Toolkit"),
        ("julius", "Julius DSP"),
        ("encodec", "Meta EnCodec"),
        ("dac", "Descript Audio Codec"),
        ("snac", "SNAC Multi-scale Codec"),
        ("tensorboard", "TensorBoard"),
    ]
    results = []
    for pkg, name in packages:
        try:
            __import__(pkg)
            results.append((name, True, "Installed ✅"))
        except ImportError:
            results.append((name, False, f"Missing! (pip install {pkg}) ❌"))
    return results


def check_assets():
    neumark_root = get_neumark_root()
    assets = [
        (
            "SpeechTokenizer Configuration",
            neumark_root / "STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json",
            "Required for VALL-E acoustic token decoding",
        ),
        (
            "SpeechTokenizer Checkpoint (.pt)",
            neumark_root / "STmodels/pretrained_model/SpeechTokenizer.pt",
            "SpeechTokenizer pretrained neural audio codec weights (459 MB)",
        ),
        (
            "WavLM SV Checkpoint (.pth)",
            SCRIPT_DIR / "models/wavlm_large_finetune.pth",
            "Speaker verification model for SpeakerSimLoss (1.2 GB)",
        ),
        (
            "Training Data Manifest (cuts_train)",
            SCRIPT_DIR / "data/tokenized_voicemark/cuts_train.jsonl.gz",
            "Lhotse training dataset with 8-layer SpeechTokenizer codes (38 MB)",
        ),
        (
            "Validation Data Manifest (cuts_dev)",
            SCRIPT_DIR / "data/tokenized_voicemark/cuts_dev.jsonl.gz",
            "Lhotse validation dataset (0.7 MB)",
        ),
    ]

    results = []
    for name, path, desc in assets:
        exists = Path(path).exists()
        size_str = ""
        if exists:
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            size_str = f" ({size_mb:.1f} MB)"
        results.append((name, exists, str(path) + size_str, desc))
    return results, neumark_root


def main():
    print("=" * 75)
    print("  NeuMark Cross-Device Environment & Assets Diagnostic Tool")
    print("=" * 75)

    # 1. Check Python Packages
    print("\n[1/3] Checking Python Dependencies:")
    print("-" * 75)
    pkg_results = check_python_packages()
    all_pkgs_ok = True
    for name, ok, msg in pkg_results:
        print(f"  - {name:<30} : {msg}")
        if not ok:
            all_pkgs_ok = False

    # 2. Check CUDA & GPU
    print("\n[2/3] Checking CUDA & GPU Acceleration:")
    print("-" * 75)
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"  - CUDA Available               : True ✅ ({device_name}, {vram_gb:.1f} GB VRAM)")
        else:
            print("  - CUDA Available               : False ⚠️ (Running on CPU mode)")
    except Exception as e:
        print(f"  - CUDA Check Error             : {e}")

    # 3. Check Large File Assets & Paths
    print("\n[3/3] Checking Model Weights & Data Manifests:")
    print("-" * 75)
    asset_results, neumark_root = check_assets()
    print(f"  Resolved NeuMark Root Path     : {neumark_root}\n")

    all_assets_ok = True
    for name, exists, path, desc in asset_results:
        status_icon = "✅ Ready" if exists else "❌ NOT FOUND"
        print(f"  - {name:<32} : {status_icon}")
        print(f"    Path : {path}")
        print(f"    Note : {desc}\n")
        if not exists:
            all_assets_ok = False

    # Overall Summary
    print("=" * 75)
    if all_pkgs_ok and all_assets_ok:
        print("  🎉 ALL CHECKS PASSED! The environment is 100% ready for training & inference.")
    else:
        print("  ⚠️ SOME ASSETS OR PACKAGES ARE MISSING. Please review the items above.")
        print("  👉 See README_PORTABILITY.md for step-by-step asset preparation guide.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()

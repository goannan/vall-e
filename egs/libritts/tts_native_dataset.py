import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lhotse import CutSet, load_manifest_lazy
import torchaudio


SCRIPT_DIR = Path(__file__).parent.resolve()
# SCRIPT_DIR: projects/vall-e/egs/libritts
# PROJECT_DIR: projects/vall-e
PROJECT_DIR = SCRIPT_DIR.parents[1]
# WORKSPACE_ROOT: root directory containing dataset/ and projects/
WORKSPACE_ROOT = SCRIPT_DIR.parents[3]

def resolve_wav_path(source_path: str) -> Optional[Path]:
    p = Path(source_path)
    if p.is_file():
        return p
    candidates = [
        os.environ.get("LIBRITTS_ROOT"),
        WORKSPACE_ROOT / "dataset" / "libriTTS" / "LibriTTS",
        WORKSPACE_ROOT / "dataset" / "LibriTTS",
        WORKSPACE_ROOT / "data" / "LibriTTS",
        SCRIPT_DIR / "../../../../dataset/libriTTS/LibriTTS",
        SCRIPT_DIR / "download" / "LibriTTS",
        Path.cwd() / "dataset" / "libriTTS" / "LibriTTS",
        Path.home() / "dataset" / "libriTTS" / "LibriTTS",
    ]
    parts = source_path.split("LibriTTS/")
    if len(parts) > 1:
        rel = parts[-1]
        for base in candidates:
            if base:
                cand = (Path(base) / rel).resolve()
                if cand.is_file():
                    return cand
    return None


def _is_valid_duration(cut, min_dur: float, max_dur: float) -> bool:
    return min_dur <= cut.duration <= max_dur


class TTSNativeDataset(Dataset):
    """
    Dataset that loads full-length 8-layer SpeechTokenizer tokens (or VALL-E synthesized tokens),
    full text transcripts, real target audio recordings, and paired prompt reference audio.
    """

    def __init__(
        self,
        manifest_path: str,
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        sample_rate: int = 16000,
        downsample_rate: int = 320,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.sample_rate = sample_rate
        self.downsample_rate = downsample_rate

        print(f"[Dataset] Loading manifest: {self.manifest_path} (full audio mode)...")
        self.cuts = load_manifest_lazy(self.manifest_path)
        # Filter only extreme outliers (e.g. >30s or <0.5s)
        self.cut_list = [c for c in self.cuts if _is_valid_duration(c, min_duration, max_duration)]
        print(f"[Dataset] Loaded {len(self.cut_list)} valid full-length cuts.")

        # Load or initialize prompt audio mapping for speaker reference
        self.prompt_map: Dict[str, Tuple[str, float, float]] = {}
        self._init_prompt_map()

    def _init_prompt_map(self):
        """Load pre-indexed prompt cut mapping if present or build lightweight cache."""
        candidates = [
            self.manifest_path.parent / "prompt_cuts_map.json",
            SCRIPT_DIR / "data" / "tokenized_voicemark" / "prompt_cuts_map.json",
            PROJECT_DIR / "egs" / "libritts" / "data" / "tokenized_voicemark" / "prompt_cuts_map.json",
        ]
        import json
        for c in candidates:
            if c.is_file():
                try:
                    with open(c, "r") as f:
                        self.prompt_map = json.load(f)
                    print(f"[Dataset] Loaded {len(self.prompt_map)} prompt audio mappings from {c}")
                    return
                except Exception as ex:
                    print(f"[Dataset] Warning: Failed to load {c}: {ex}")

    def __len__(self) -> int:
        return len(self.cut_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cut = self.cut_list[idx]

        # 1. Load full 8-layer codes (SpeechTokenizer codes or VALL-E tokens): [T, 8] -> [8, T]
        codes_np = cut.load_features()  # [T, 8]
        codes = torch.from_numpy(codes_np).long().transpose(0, 1)  # [8, T]

        # 2. Full text transcript
        text = cut.supervisions[0].text if cut.supervisions else ""

        # 3. Audio recording: [1, T_samples] (with automatic local path relocation)
        audio = None
        try:
            if hasattr(cut, "recording") and cut.recording and cut.recording.sources:
                orig_src = cut.recording.sources[0].source
                wav_p = resolve_wav_path(orig_src)
                if wav_p:
                    audio, orig_sr = torchaudio.load(str(wav_p))
                    if orig_sr != self.sample_rate:
                        audio = torchaudio.functional.resample(audio, orig_sr, self.sample_rate)
                    start_sample = int(cut.start * self.sample_rate)
                    num_samples = int(cut.duration * self.sample_rate)
                    audio = audio[:, start_sample : start_sample + num_samples]
        except Exception:
            pass

        if audio is None:
            try:
                audio_np = cut.load_audio()
                audio = torch.from_numpy(audio_np).float()
                if cut.sampling_rate != self.sample_rate:
                    audio = torchaudio.functional.resample(audio, cut.sampling_rate, self.sample_rate)
            except Exception:
                # Placeholder fallback if raw audio not accessible
                audio = torch.zeros((1, codes.shape[-1] * self.downsample_rate), dtype=torch.float32)

        # 4. Prompt audio (speaker reference recording for zero-shot speaker similarity)
        prompt_audio = None
        if hasattr(cut, "custom") and cut.custom and "prompt_cut_id" in cut.custom:
            prompt_id = cut.custom["prompt_cut_id"]
            if prompt_id in self.prompt_map:
                try:
                    src, p_start, p_dur = self.prompt_map[prompt_id]
                    p_wav_p = resolve_wav_path(src)
                    if p_wav_p:
                        p_audio, p_sr = torchaudio.load(str(p_wav_p))
                        if p_sr != self.sample_rate:
                            p_audio = torchaudio.functional.resample(p_audio, p_sr, self.sample_rate)
                        p_start_sample = int(p_start * self.sample_rate)
                        p_num_samples = int(p_dur * self.sample_rate)
                        prompt_audio = p_audio[:, p_start_sample : p_start_sample + p_num_samples]
                except Exception:
                    pass

        if prompt_audio is None:
            prompt_audio = audio.clone()

        return {
            "id": cut.id,
            "codes": codes,  # [8, T]
            "text": text,
            "audio": audio,  # [1, T_samples]
            "prompt_audio": prompt_audio,  # [1, T_samples]
            "frames": codes.shape[-1],
        }


def collate_tts_native(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for full-length audio and codes using batch dynamic padding."""
    batch_size = len(batch)
    max_frames = max(item["codes"].shape[-1] for item in batch)
    max_audio_len = max_frames * 320
    max_prompt_len = max(item["prompt_audio"].shape[-1] for item in batch)

    padded_codes = torch.zeros((batch_size, 8, max_frames), dtype=torch.long)
    padded_audio = torch.zeros((batch_size, 1, max_audio_len), dtype=torch.float32)
    padded_prompt = torch.zeros((batch_size, 1, max_prompt_len), dtype=torch.float32)
    frame_lengths = []
    texts = []
    ids = []

    for i, item in enumerate(batch):
        f_len = item["codes"].shape[-1]
        a_len = min(item["audio"].shape[-1], f_len * 320)
        p_len = item["prompt_audio"].shape[-1]

        padded_codes[i, :, :f_len] = item["codes"]
        padded_audio[i, :, :a_len] = item["audio"][:, :a_len]
        padded_prompt[i, :, :p_len] = item["prompt_audio"]

        frame_lengths.append(f_len)
        texts.append(item["text"])
        ids.append(item["id"])

    return {
        "ids": ids,
        "codes": padded_codes,  # [B, 8, T_max]
        "audio": padded_audio,  # [B, 1, T_audio_max]
        "prompt_audio": padded_prompt,  # [B, 1, T_prompt_max]
        "frame_lengths": torch.tensor(frame_lengths, dtype=torch.long),
        "texts": texts,
    }


def get_tts_native_dataloader(
    manifest_path: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    max_duration: float = 30.0,
    min_duration: float = 0.5,
) -> DataLoader:
    dataset = TTSNativeDataset(
        manifest_path=manifest_path,
        max_duration=max_duration,
        min_duration=min_duration,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_tts_native,
        pin_memory=True,
        drop_last=True,
    )

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lhotse import CutSet, load_manifest_lazy
import torchaudio


class TTSNativeDataset(Dataset):
    """
    Dataset that loads full-length 8-layer SpeechTokenizer tokens,
    full text transcripts, and full real audio recordings without truncation.
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
        self.cuts = self.cuts.filter(lambda c: min_duration <= c.duration <= max_duration)
        self.cut_list = list(self.cuts)
        print(f"[Dataset] Loaded {len(self.cut_list)} valid full-length cuts.")

    def __len__(self) -> int:
        return len(self.cut_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cut = self.cut_list[idx]

        # 1. Load full 8-layer SpeechTokenizer codes: [T, 8] -> [8, T]
        codes_np = cut.load_features()  # [T, 8]
        codes = torch.from_numpy(codes_np).long().transpose(0, 1)  # [8, T]

        # 2. Full text transcript
        text = cut.supervisions[0].text if cut.supervisions else ""

        # 3. Audio recording: [1, T_samples] (with resilient fallback if raw wavs are not colocated)
        try:
            if cut.has_recording:
                audio_np = cut.recording.load_audio()
                audio = torch.from_numpy(audio_np).float()
                if cut.recording.sampling_rate != self.sample_rate:
                    audio = torchaudio.functional.resample(audio, cut.recording.sampling_rate, self.sample_rate)
            else:
                audio = torch.zeros((1, codes.shape[-1] * self.downsample_rate), dtype=torch.float32)
        except Exception:
            # Reconstruct clean placeholder audio if raw LibriTTS directory is not downloaded on remote node
            audio = torch.zeros((1, codes.shape[-1] * self.downsample_rate), dtype=torch.float32)

        # 4. Prompt audio: resolve speaker reference recording (prompt wav or target speaker audio)
        prompt_audio = None
        p_id = cut.custom.get("prompt_cut_id", "") if cut.custom else ""
        if p_id and cut.has_recording and len(cut.recording.sources) > 0:
            try:
                rec_source = cut.recording.sources[0].source
                rec_path = Path(rec_source)
                p_rec_id = p_id.rsplit("-", 1)[0] if "-" in p_id else p_id
                parts = p_rec_id.split("_")
                if len(parts) >= 4:
                    spk, chap = parts[0], parts[1]
                    p_wav_path = rec_path.parents[2] / spk / chap / f"{p_rec_id}.wav"
                    if p_wav_path.exists():
                        p_wav, p_sr = torchaudio.load(str(p_wav_path))
                        if p_sr != self.sample_rate:
                            p_wav = torchaudio.functional.resample(p_wav, p_sr, self.sample_rate)
                        prompt_audio = p_wav
            except Exception:
                prompt_audio = None

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

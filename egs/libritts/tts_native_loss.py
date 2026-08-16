import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
NEUMARK_ROOT = PROJECT_DIR.parent / "NeuMark"

# Add VoiceMark and UniSpeech to sys.path
for p in [str(NEUMARK_ROOT), str(NEUMARK_ROOT / "train"), str(SCRIPT_DIR / "tools/seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification")]:
    if p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)

from loss import (
    adversarial_loss_d,
    adversarial_loss_g,
    bits_to_chunks,
    cos_loss,
    decoding_loss,
    feature_loss,
    vad_based_loss,
)


# ==========================================
# 1. UTMOS Loss (Subjective Naturalness)
# ==========================================
class UTMOSLoss(nn.Module):
    """Predicts MOS naturalness score using SpeechMOS UTMOS22 model and maximizes it."""

    def __init__(self, repository: str = "tarepan/SpeechMOS:v1.2.0", model_name: str = "utmos22_strong", device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        try:
            self.model = torch.hub.load(repository, model_name, trust_repo=True, verbose=False)
            self.model.eval()
            self.model.requires_grad_(False)
            self.model.to(self.device)
            self.available = True
        except Exception as e:
            print(f"[Warning] Failed to load UTMOS model: {e}. UTMOS loss will be disabled.")
            self.available = False

    def forward(self, audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Input: audio waveform [B, 1, T] or [B, T]
        Output: scalar loss (- mean MOS score)
        """
        if not self.available:
            return torch.zeros((), device=audio.device, requires_grad=True)

        if audio.ndim == 3:
            audio = audio.squeeze(1)  # [B, T]

        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)

        # Minimum receptive field check (400 samples)
        if audio.shape[-1] < 400:
            return torch.zeros((), device=audio.device, requires_grad=True)

        # UTMOS forward
        scores = self.model(audio, 16000)  # [B, 1] or [B]
        # Loss is negative score (higher MOS -> lower loss)
        loss = -scores.mean()
        return loss


# ==========================================
# 2. Speaker Similarity Loss (WavLM-Large SV)
# ==========================================
class SpeakerSimLoss(nn.Module):
    """Calculates cosine similarity loss against target speaker prompt embedding."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        if checkpoint_path is None:
            checkpoint_path = str(SCRIPT_DIR / "models/wavlm_large_finetune.pth")

        self.available = False
        if os.path.exists(checkpoint_path):
            try:
                from verification import init_model
                self.model = init_model("wavlm_large", checkpoint=checkpoint_path)
                self.model.eval()
                self.model.requires_grad_(False)
                self.model.to(self.device)
                self.available = True
            except Exception as e:
                print(f"[Warning] Failed to initialize WavLM SV model: {e}. SpeakerSimLoss will fall back to Cosine.")
        else:
            print(f"[Warning] WavLM checkpoint not found at {checkpoint_path}. SpeakerSimLoss will fall back.")

    def forward(self, wm_audio: torch.Tensor, prompt_audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Inputs: wm_audio [B, 1, T], prompt_audio [B, 1, T_p]
        Output: scalar loss (1.0 - cosine_sim)
        """
        if wm_audio.ndim == 3:
            wm_audio = wm_audio.squeeze(1)
        if prompt_audio.ndim == 3:
            prompt_audio = prompt_audio.squeeze(1)

        if sample_rate != 16000:
            wm_audio = torchaudio.functional.resample(wm_audio, sample_rate, 16000)
            prompt_audio = torchaudio.functional.resample(prompt_audio, sample_rate, 16000)

        if not self.available:
            # Fallback: simple spectrogram or feature cosine similarity
            return cos_loss(wm_audio, prompt_audio[..., :wm_audio.shape[-1]])

        # Extract embeddings
        emb_wm = self.model(wm_audio)  # [B, D]
        with torch.no_grad():
            emb_prompt = self.model(prompt_audio)  # [B, D]

        cos_sim = F.cosine_similarity(emb_wm, emb_prompt, dim=-1)
        loss = (1.0 - cos_sim).mean()
        return loss


# ==========================================
# 3. ASR CTC Loss (Wav2Vec2 Pronunciation)
# ==========================================
class ASRLoss(nn.Module):
    """Calculates CTC Loss against ground truth text tokens using Wav2Vec2 ASR."""

    def __init__(self, bundle_name: str = "WAV2VEC2_ASR_BASE_960H", device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.bundle = getattr(torchaudio.pipelines, bundle_name)
        self.model = self.bundle.get_model().to(self.device).eval()
        self.model.requires_grad_(False)
        self.labels = self.bundle.get_labels()
        self.char2id = {c: i for i, c in enumerate(self.labels)}
        self.blank_id = 0
        self.ctc_loss_fn = nn.CTCLoss(blank=self.blank_id, zero_infinity=True)

    def text_to_ids(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        targets = []
        lengths = []
        for t in texts:
            # Standardize text for Wav2Vec2 (uppercase, only supported chars)
            clean_t = t.upper().replace("\n", " ").strip()
            ids = [self.char2id[c] for c in clean_t if c in self.char2id]
            if not ids:
                ids = [self.char2id.get("|", 4)]
            targets.extend(ids)
            lengths.append(len(ids))
        return torch.tensor(targets, dtype=torch.long, device=self.device), torch.tensor(lengths, dtype=torch.long, device=self.device)

    def forward(self, audio: torch.Tensor, texts: List[str], sample_rate: int = 16000) -> torch.Tensor:
        """
        Inputs: audio [B, 1, T] or [B, T], texts list of length B
        Output: scalar CTC loss
        """
        if audio.ndim == 3:
            audio = audio.squeeze(1)

        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)

        emissions, _ = self.model(audio)  # [B, frames, num_classes]
        log_probs = F.log_softmax(emissions, dim=-1).transpose(0, 1)  # [frames, B, classes]

        targets, target_lengths = self.text_to_ids(texts)
        input_lengths = torch.full((audio.size(0),), log_probs.size(0), dtype=torch.long, device=self.device)

        loss = self.ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
        return loss


# ==========================================
# 4. Latent Space Cosine Loss (Anchor)
# ==========================================
def latent_cosine_loss(z_wm: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
    """Cosine similarity loss on continuous latent representation to anchor perturbations."""
    min_len = min(z_wm.shape[-1], z_q.shape[-1])
    z_wm = z_wm[..., :min_len]
    z_q = z_q[..., :min_len]
    return cos_loss(z_wm, z_q)

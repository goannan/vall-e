import sys
import os
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
NEUMARK_ROOT = PROJECT_DIR.parent / "NeuMark"

# Insert NeuMark root at index 0 to ensure priority
for p in [str(NEUMARK_ROOT / "train"), str(NEUMARK_ROOT)]:
    if p in sys.path:
        sys.path.remove(p)
    if os.path.exists(p):
        sys.path.insert(0, p)

try:
    from losses import (
        adversarial_loss_d,
        adversarial_loss_g,
        bits_to_chunks,
        cos_loss,
        decoding_loss,
        feature_loss,
        vad_based_loss,
    )
except ImportError:
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

        scores = self.model(audio, 16000)  # [B, 1] or [B]
        loss = -scores.mean()
        return loss


# ==========================================
# 2. Speaker Similarity Loss (WavLM-Large SV)
# ==========================================
class SpeakerSimLoss(nn.Module):
    def get_similarity(self, wm_audio: torch.Tensor, prompt_audio: torch.Tensor, sample_rate: int = 16000) -> float:
        if not self.available or self.model is None:
            return 0.0
        if wm_audio.ndim == 3:
            wm_audio = wm_audio.squeeze(1)
        if prompt_audio.ndim == 3:
            prompt_audio = prompt_audio.squeeze(1)
        if sample_rate != 16000:
            wm_audio = torchaudio.functional.resample(wm_audio, sample_rate, 16000)
            prompt_audio = torchaudio.functional.resample(prompt_audio, sample_rate, 16000)
        with torch.no_grad():
            emb_wm = self.model(wm_audio)
            emb_prompt = self.model(prompt_audio)
            sim = F.cosine_similarity(emb_wm, emb_prompt, dim=-1).mean().item()
        return sim

    """Calculates cosine similarity loss against target speaker prompt embedding."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        if checkpoint_path is None or not os.path.isabs(checkpoint_path):
            checkpoint_path = str(SCRIPT_DIR / (checkpoint_path or "models/wavlm_large_finetune.pth"))

        self.available = False
        if os.path.exists(checkpoint_path):
            try:
                import importlib.util
                ecapa_file = SCRIPT_DIR / "tools/seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification/models/ecapa_tdnn.py"
                spec = importlib.util.spec_from_file_location("ecapa_tdnn_standalone", str(ecapa_file))
                ecapa_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ecapa_mod)
                ECAPA_TDNN_SMALL = ecapa_mod.ECAPA_TDNN_SMALL

                self.model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large")
                state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
                self.model.load_state_dict(state_dict["model"], strict=False)
                self.model.eval()
                self.model.requires_grad_(False)
                self.model.to(self.device)
                self.available = True
            except Exception as e:
                print(f"[Warning] Failed to initialize WavLM SV model: {e}. SpeakerSimLoss will fall back.")
                self.available = False
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
            return cos_loss(wm_audio, prompt_audio[..., :wm_audio.shape[-1]])

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
    def decode_greedy(self, audio: torch.Tensor, sample_rate: int = 16000) -> List[str]:
        if audio.ndim == 3:
            audio = audio.squeeze(1)
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)
        with torch.no_grad():
            emissions, _ = self.model(audio)
            indices = torch.argmax(emissions, dim=-1)
        
        transcripts = []
        for seq in indices:
            collapsed = []
            prev = None
            for idx in seq.tolist():
                if idx != prev:
                    if idx != self.blank_id:
                        collapsed.append(self.labels[idx])
                prev = idx
            text = "".join(collapsed).replace("|", " ").strip()
            transcripts.append(text)
        return transcripts

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
            clean_t = t.upper().replace("\n", " ").strip()
            ids = [self.char2id[c] for c in clean_t if c in self.char2id]
            if not ids:
                ids = [self.char2id.get("|", 4)]
            targets.extend(ids)
            lengths.append(len(ids))
        return torch.tensor(targets, dtype=torch.long, device=self.device), torch.tensor(lengths, dtype=torch.long, device=self.device)

    def forward(self, audio: torch.Tensor, texts: List[str], sample_rate: int = 16000) -> torch.Tensor:
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


# ==========================================
# 5. Multi-scale Mel Spectrogram Loss (NeuMark)
# ==========================================
_mel_basis_cache: Dict[str, torch.Tensor] = {}
_hann_window_cache: Dict[str, torch.Tensor] = {}


def dynamic_range_compression_torch(x: torch.Tensor, C: float = 1.0, clip_val: float = 1e-5) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val) * C)


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int = 1024,
    num_mels: int = 80,
    sample_rate: int = 16000,
    hop_size: int = 240,
    win_size: int = 1024,
    fmin: float = 0.0,
    fmax: Optional[float] = 8000.0,
    center: bool = False,
) -> torch.Tensor:
    """Extract Log-Mel spectrogram matching NeuMark pipeline."""
    if y.ndim == 3:
        y = y.squeeze(1)

    n_fft = int(n_fft)
    num_mels = int(num_mels)
    sample_rate = int(sample_rate)
    hop_size = int(hop_size)
    win_size = int(win_size)
    fmin = float(fmin)
    fmax = float(fmax) if fmax is not None else float(sample_rate // 2)

    global _mel_basis_cache, _hann_window_cache
    dev = y.device
    basis_key = f"{fmax}_{win_size}_{n_fft}_{num_mels}_{dev}"
    if basis_key not in _mel_basis_cache:
        mel_transform = torchaudio.transforms.MelScale(
            n_mels=num_mels,
            sample_rate=sample_rate,
            n_stft=n_fft // 2 + 1,
            f_min=fmin,
            f_max=fmax,
            norm="slaney",
            mel_scale="htk",
        )
        _mel_basis_cache[basis_key] = mel_transform.fb.float().T.to(dev)

    win_key = f"{win_size}_{dev}"
    if win_key not in _hann_window_cache:
        _hann_window_cache[win_key] = torch.hann_window(win_size, device=dev)

    # Pad waveform
    pad_len = int((n_fft - hop_size) / 2)
    y_padded = F.pad(y.unsqueeze(1), (pad_len, pad_len), mode="reflect").squeeze(1)

    spec_complex = torch.stft(
        y_padded,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=_hann_window_cache[win_key],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    spec = torch.view_as_real(spec_complex)
    spec = torch.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2 + 1e-9)
    spec = torch.matmul(_mel_basis_cache[basis_key], spec)
    spec = dynamic_range_compression_torch(spec)
    return spec


def single_mel_loss(x: torch.Tensor, x_hat: torch.Tensor, **kwargs) -> torch.Tensor:
    """L1 Mel Spectrogram loss between reference x and watermarked x_hat."""
    x_mel = mel_spectrogram(x, **kwargs)
    x_hat_mel = mel_spectrogram(x_hat, **kwargs)
    min_len = min(x_mel.size(-1), x_hat_mel.size(-1))
    return F.l1_loss(x_mel[..., :min_len], x_hat_mel[..., :min_len])


class MultiScaleMelLoss(nn.Module):
    """Multi-scale Mel Spectrogram L1 Loss matching NeuMark default configuration."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_size: int = 240,
        win_size: int = 1024,
        num_mels: int = 80,
        fmin: float = 0.0,
        fmax: Optional[float] = 8000.0,
        lambdas: Optional[List[float]] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.lambdas = lambdas or [5.0, 1.0, 1.0, 1.0]
        self.kwargs_list = []
        mult = 1
        for _ in range(len(self.lambdas)):
            self.kwargs_list.append(
                {
                    "n_fft": n_fft // mult,
                    "num_mels": num_mels,
                    "sample_rate": sample_rate,
                    "hop_size": hop_size // mult,
                    "win_size": win_size // mult,
                    "fmin": fmin,
                    "fmax": fmax,
                }
            )
            mult *= 2

    def forward(self, ref_audio: torch.Tensor, wm_audio: torch.Tensor) -> torch.Tensor:
        """
        ref_audio: [B, 1, T] or [B, T] (Reconstructed audio without watermark)
        wm_audio:  [B, 1, T] or [B, T] (Watermarked audio)
        """
        total_mel = torch.zeros((), device=ref_audio.device, dtype=torch.float32)
        for w, kw in zip(self.lambdas, self.kwargs_list):
            l_scale = single_mel_loss(ref_audio, wm_audio, **kw)
            total_mel = total_mel + (w * l_scale)
        return total_mel


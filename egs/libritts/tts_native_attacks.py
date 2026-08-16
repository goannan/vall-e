import random
from typing import Callable, Dict, List, Optional, Tuple

import julius
import torch
import torch.nn.functional as F

# Import attack implementations
try:
    from encodec import EncodecModel
except ImportError:
    EncodecModel = None

try:
    import dac
except ImportError:
    dac = None

try:
    from snac import SNAC
except ImportError:
    SNAC = None


def _match_audio_length(audio: torch.Tensor, target_len: int) -> torch.Tensor:
    if audio.shape[-1] > target_len:
        return audio[..., :target_len]
    elif audio.shape[-1] < target_len:
        return F.pad(audio, (0, target_len - audio.shape[-1]))
    return audio


# -----------------------------------------------------------------------------
# 1. Encodec Attack Wrapper (24kHz standard model)
# -----------------------------------------------------------------------------
class EncodecAttack:
    _model = None

    @classmethod
    def get_model(cls, device):
        if cls._model is None:
            model = EncodecModel.encodec_model_24khz()
            model.set_target_bandwidth(6.0)
            model.eval().to(device)
            for p in model.parameters():
                p.requires_grad = False
            cls._model = model
        return cls._model

    @classmethod
    def compress(cls, wav: torch.Tensor, bandwidth: float, sample_rate: int = 16000) -> torch.Tensor:
        orig_len = wav.shape[-1]
        device = wav.device
        model = cls.get_model(device)
        model.set_target_bandwidth(bandwidth)

        with torch.no_grad():
            wav_24k = julius.resample_frac(wav, sample_rate, 24000) if sample_rate != 24000 else wav
            if wav_24k.ndim == 2:
                wav_24k = wav_24k.unsqueeze(1)
            frames = model.encode(wav_24k)
            decoded = model.decode(frames)
            out = julius.resample_frac(decoded, 24000, sample_rate) if sample_rate != 24000 else decoded
        return _match_audio_length(out, orig_len)

    @classmethod
    def release(cls):
        cls._model = None


# -----------------------------------------------------------------------------
# 2. DAC Attack Wrapper (16kHz, 24kHz, 44.1kHz)
# -----------------------------------------------------------------------------
class DACAttack:
    _models = {}

    @classmethod
    def get_model(cls, model_type: str, device):
        if model_type not in cls._models:
            model_path = dac.utils.download(model_type=model_type)
            model = dac.DAC.load(model_path)
            model.eval().to(device)
            for p in model.parameters():
                p.requires_grad = False
            cls._models[model_type] = model
        return cls._models[model_type]

    @classmethod
    def compress(cls, wav: torch.Tensor, model_type: str, sample_rate: int = 16000) -> torch.Tensor:
        orig_len = wav.shape[-1]
        device = wav.device
        target_sr = 16000 if model_type == "16khz" else (24000 if model_type == "24khz" else 44100)
        model = cls.get_model(model_type, device)

        with torch.no_grad():
            wav_target = julius.resample_frac(wav, sample_rate, target_sr) if sample_rate != target_sr else wav
            if wav_target.ndim == 2:
                wav_target = wav_target.unsqueeze(1)
            x = model.preprocess(wav_target, target_sr)
            z, codes, latents, _, _ = model.encode(x)
            decoded = model.decode(z)
            out = julius.resample_frac(decoded, target_sr, sample_rate) if sample_rate != target_sr else decoded
        return _match_audio_length(out, orig_len)

    @classmethod
    def release(cls):
        cls._models.clear()


# -----------------------------------------------------------------------------
# 3. SNAC Attack Wrapper (24kHz 0.98kbps, 32kHz 1.9kbps, 44.1kHz 2.6kbps)
# -----------------------------------------------------------------------------
class SNACAttack:
    _models = {}

    @classmethod
    def get_model(cls, model_type: str, device):
        if model_type not in cls._models:
            model_map = {
                "24khz": "hubertsiuzdak/snac_24khz",
                "32khz": "hubertsiuzdak/snac_32khz",
                "44khz": "hubertsiuzdak/snac_44khz",
            }
            model = SNAC.from_pretrained(model_map[model_type]).eval().to(device)
            for p in model.parameters():
                p.requires_grad = False
            cls._models[model_type] = model
        return cls._models[model_type]

    @classmethod
    def compress(cls, wav: torch.Tensor, model_type: str, sample_rate: int = 16000) -> torch.Tensor:
        orig_len = wav.shape[-1]
        device = wav.device
        target_sr = 24000 if model_type == "24khz" else (32000 if model_type == "32khz" else 44100)
        model = cls.get_model(model_type, device)

        with torch.no_grad():
            wav_target = julius.resample_frac(wav, sample_rate, target_sr) if sample_rate != target_sr else wav
            if wav_target.ndim == 2:
                wav_target = wav_target.unsqueeze(1)
            codes = model.encode(wav_target)
            decoded = model.decode(codes)
            out = julius.resample_frac(decoded, target_sr, sample_rate) if sample_rate != target_sr else decoded
        return _match_audio_length(out, orig_len)

    @classmethod
    def release(cls):
        cls._models.clear()


def release_codec_models():
    """Release cached attack models to free VRAM."""
    EncodecAttack.release()
    DACAttack.release()
    SNACAttack.release()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# 4. Voice Conversion / Masking Attack for VAD Supervision
# -----------------------------------------------------------------------------
def apply_masking(audio: torch.Tensor, orig_audio: Optional[torch.Tensor], mask_prob: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    B, C, T = audio.shape
    downsampling_ratios = 320
    n_frames = T // downsampling_ratios

    vad_labels = torch.ones(B, n_frames, device=audio.device)
    mask = torch.rand(B, n_frames, device=audio.device) < mask_prob
    vad_labels[mask] = 0.0

    mask_expanded = mask.view(B, 1, n_frames, 1).expand(-1, -1, -1, downsampling_ratios).reshape(B, 1, n_frames * downsampling_ratios)
    if mask_expanded.shape[-1] < T:
        mask_pad = torch.zeros(B, 1, T - mask_expanded.shape[-1], dtype=torch.bool, device=mask.device)
        mask_expanded = torch.cat([mask_expanded, mask_pad], dim=-1)

    augmented = audio.clone()
    fill_source = orig_audio if orig_audio is not None else torch.zeros_like(audio)
    if fill_source.shape[-1] < T:
        fill_source = F.pad(fill_source, (0, T - fill_source.shape[-1]))
    else:
        fill_source = fill_source[..., :T]

    augmented[mask_expanded] = fill_source[mask_expanded]
    return augmented, vad_labels


# -----------------------------------------------------------------------------
# 5. Training Attack Function (Only Identity, Masking, EnCodec 3/6/12 kbps)
# -----------------------------------------------------------------------------
def apply_train_augmentation(
    audio: torch.Tensor, sample_rate: int = 16000, orig_audio: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    B, C, T = audio.shape
    n_frames = T // 320

    # User specified: Only Identity, VC Masking, and Encodec 3, 6, 12 kbps
    attacks = [
        "identity",
        "vc_masking",
        "encodec_3kbps",
        "encodec_6kbps",
        "encodec_12kbps",
    ]
    # Balanced weights: 30% clean, 20% masking, 50% Encodec
    weights = [3, 2, 2, 2, 2]
    attack_name = random.choices(attacks, weights=weights, k=1)[0]

    if attack_name == "identity":
        augmented = audio
        vad_labels = torch.ones(B, n_frames, device=audio.device)
    elif attack_name == "vc_masking":
        # Randomly compress with Encodec first, then mask
        bw = random.choice([3.0, 6.0, 12.0])
        comp = EncodecAttack.compress(audio, bw, sample_rate)
        augmented, vad_labels = apply_masking(comp, orig_audio)
    elif attack_name == "encodec_3kbps":
        augmented = EncodecAttack.compress(audio, 3.0, sample_rate)
        vad_labels = torch.ones(B, n_frames, device=audio.device)
    elif attack_name == "encodec_6kbps":
        augmented = EncodecAttack.compress(audio, 6.0, sample_rate)
        vad_labels = torch.ones(B, n_frames, device=audio.device)
    elif attack_name == "encodec_12kbps":
        augmented = EncodecAttack.compress(audio, 12.0, sample_rate)
        vad_labels = torch.ones(B, n_frames, device=audio.device)

    augmented = _match_audio_length(augmented, T)
    return torch.clamp(augmented, -1.0, 1.0), vad_labels, attack_name


# -----------------------------------------------------------------------------
# 6. Validation Suite (Clean, Encodec 3/6/12k, DAC 6/8/24k, SNAC 0.98/1.9/2.6k)
# -----------------------------------------------------------------------------
def get_validation_attack_suite(sample_rate: int = 16000) -> List[Tuple[str, str, Callable[[torch.Tensor], torch.Tensor]]]:
    """Returns the ordered validation attacks list: (Family, Bitrate, fn)"""
    return [
        ("Clean", "Identity", lambda wav: wav),
        ("Encodec", "3 kbps", lambda wav: EncodecAttack.compress(wav, 3.0, sample_rate)),
        ("Encodec", "6 kbps", lambda wav: EncodecAttack.compress(wav, 6.0, sample_rate)),
        ("Encodec", "12 kbps", lambda wav: EncodecAttack.compress(wav, 12.0, sample_rate)),
        ("DAC", "6 kbps", lambda wav: DACAttack.compress(wav, "16khz", sample_rate)),
        ("DAC", "8 kbps", lambda wav: DACAttack.compress(wav, "44khz", sample_rate)),
        ("DAC", "24 kbps", lambda wav: DACAttack.compress(wav, "24khz", sample_rate)),
        ("SNAC", "0.98 kbps", lambda wav: SNACAttack.compress(wav, "24khz", sample_rate)),
        ("SNAC", "1.9 kbps", lambda wav: SNACAttack.compress(wav, "32khz", sample_rate)),
        ("SNAC", "2.6 kbps", lambda wav: SNACAttack.compress(wav, "44khz", sample_rate)),
    ]


def format_codec_eval_table(step: int, results: Dict[str, Dict[str, float]]) -> str:
    """Renders a beautiful ASCII table matching VALL-E eval style."""
    header = f"| {'Codec Family':<14} | {'Bitrate':<12} | {'Bit Acc (%)':<13} | {'BER (%)':<11} | {'Detect ACC (%)':<14} |"
    sep = f"|{'-'*16}|{'-'*14}|{'-'*15}|{'-'*13}|{'-'*16}|"
    lines = [
        "=" * len(header),
        f"  TTS-Native Watermark Validation Report (Step: {step:07d})",
        "=" * len(header),
        header,
        sep,
    ]
    for key, stats in results.items():
        family, bitrate = key.split("::")
        bit_acc = stats["bit_acc"]
        ber = stats["ber"]
        det_acc = stats["detect_acc"]
        lines.append(f"| {family:<14} | {bitrate:<12} | {bit_acc:>10.2f} %  | {ber:>8.2f} %  | {det_acc:>11.2f} %   |")
    lines.append("=" * len(header))
    return "\n".join(lines)

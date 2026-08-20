import random
from typing import Callable, Dict, List, Optional, Tuple

import julius
import torch
import torch.nn.functional as F
import torchaudio

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


def edit_distance(seq1, seq2) -> int:
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def compute_wer_cer(ref: str, hyp: str) -> Tuple[float, float]:
    ref_clean = ref.strip().upper()
    hyp_clean = hyp.strip().upper()

    # Word Error Rate
    r_words = ref_clean.split()
    h_words = hyp_clean.split()
    wer = (edit_distance(r_words, h_words) / max(1, len(r_words))) if r_words else 0.0

    # Character Error Rate
    r_chars = list(ref_clean.replace(" ", "|"))
    h_chars = list(hyp_clean.replace(" ", "|"))
    cer = (edit_distance(r_chars, h_chars) / max(1, len(r_chars))) if r_chars else 0.0

    return min(1.0, wer), min(1.0, cer)


def _match_audio_length(audio: torch.Tensor, target_len: int) -> torch.Tensor:
    if audio.shape[-1] > target_len:
        return audio[..., :target_len]
    elif audio.shape[-1] < target_len:
        return F.pad(audio, (0, target_len - audio.shape[-1]))
    return audio


# -----------------------------------------------------------------------------
# 1. DSP Audio Effects
# -----------------------------------------------------------------------------
class AudioEffects:
    @staticmethod
    def identity(wav: torch.Tensor) -> torch.Tensor:
        return wav

    @staticmethod
    def random_noise(wav: torch.Tensor, noise_std: float = 0.001) -> torch.Tensor:
        return wav + torch.randn_like(wav) * noise_std

    @staticmethod
    def pink_noise(wav: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
        length = wav.shape[-1]
        num_rows = 16
        array = torch.randn(num_rows, length // num_rows + 1, device=wav.device)
        reshaped = torch.cumsum(array, dim=1).reshape(-1)[:length]
        pink = reshaped / torch.max(torch.abs(reshaped) + 1e-8)
        return wav + pink * noise_std

    @staticmethod
    def lowpass_filter(wav: torch.Tensor, cutoff_freq: float = 5000, sample_rate: int = 16000) -> torch.Tensor:
        return torchaudio.functional.lowpass_biquad(wav, sample_rate=sample_rate, cutoff_freq=cutoff_freq)

    @staticmethod
    def highpass_filter(wav: torch.Tensor, cutoff_freq: float = 500, sample_rate: int = 16000) -> torch.Tensor:
        return torchaudio.functional.highpass_biquad(wav, sample_rate=sample_rate, cutoff_freq=cutoff_freq)

    @staticmethod
    def bandpass_filter(wav: torch.Tensor, cutoff_freq_low: float = 300, cutoff_freq_high: float = 8000, sample_rate: int = 16000) -> torch.Tensor:
        mid = (cutoff_freq_low + cutoff_freq_high) / 2
        bw = cutoff_freq_high - cutoff_freq_low
        q = mid / max(1.0, bw)
        return torchaudio.functional.bandpass_biquad(wav, sample_rate=sample_rate, central_freq=mid, Q=q)

    @staticmethod
    def echo(wav: torch.Tensor, volume: float = 0.3, duration: float = 0.2, sample_rate: int = 16000) -> torch.Tensor:
        delay = int(sample_rate * duration)
        if delay >= wav.shape[-1]:
            return wav
        echo_sig = torch.zeros_like(wav)
        echo_sig[..., delay:] = wav[..., :-delay] * volume
        return wav + echo_sig

    @staticmethod
    def smooth(wav: torch.Tensor, window_size: int = 5) -> torch.Tensor:
        kernel = torch.ones(1, 1, window_size, device=wav.device) / window_size
        padded = F.pad(wav, (window_size // 2, window_size // 2), mode="reflect")
        return F.conv1d(padded, kernel)

    @staticmethod
    def boost_audio(wav: torch.Tensor, amount: float = 10) -> torch.Tensor:
        gain = 10 ** (amount / 20)
        return wav * gain

    @staticmethod
    def duck_audio(wav: torch.Tensor, amount: float = 10) -> torch.Tensor:
        gain = 10 ** (-amount / 20)
        return wav * gain

    @staticmethod
    def updownresample(wav: torch.Tensor, sample_rate: int = 16000, intermediate_freq: int = 32000) -> torch.Tensor:
        resampled = julius.resample_frac(wav, sample_rate, intermediate_freq)
        return julius.resample_frac(resampled, intermediate_freq, sample_rate)

    @staticmethod
    def speed(wav: torch.Tensor, speed_factor: float = 1.1, sample_rate: int = 16000) -> torch.Tensor:
        orig_len = wav.shape[-1]
        stretched = julius.resample_frac(wav, int(sample_rate * speed_factor), sample_rate)
        return _match_audio_length(stretched, orig_len)


# -----------------------------------------------------------------------------
# 2. Neural Codec Attacks (Encodec, DAC, SNAC)
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

        with torch.inference_mode():
            wav_detached = wav.detach()
            wav_24k = julius.resample_frac(wav_detached, sample_rate, 24000) if sample_rate != 24000 else wav_detached
            if wav_24k.ndim == 2:
                wav_24k = wav_24k.unsqueeze(1)
            frames = model.encode(wav_24k)
            decoded = model.decode(frames)
            out = julius.resample_frac(decoded, 24000, sample_rate) if sample_rate != 24000 else decoded
        return _match_audio_length(out, orig_len)

    @classmethod
    def release(cls):
        cls._model = None


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
    EncodecAttack.release()
    DACAttack.release()
    SNACAttack.release()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# 3. Voice Conversion Masking for VAD Supervision
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
# 4. Training Augmentation (Full DSP + Neural Codec + VC Masking)
# -----------------------------------------------------------------------------
def apply_train_augmentation(
    audio: torch.Tensor, sample_rate: int = 16000, orig_audio: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    B, C, T = audio.shape
    n_frames = T // 320
    vad_labels = torch.ones(B, n_frames, device=audio.device)

    # Complete list of Training Attacks: Clean + DSP + Codecs + VC Masking
    attacks = [
        "identity",
        "vc_masking",
        "gaussian_noise",
        "pink_noise",
        "lowpass",
        "highpass",
        "bandpass",
        "echo",
        "smooth",
        "resample",
        "volume_boost",
        "volume_duck",
        "speed",
        "encodec_3kbps",
        "encodec_6kbps",
        "encodec_12kbps",
    ]
    # Balanced weights across clean, VC, DSP and Codec categories
    weights = [
        4,  # identity
        3,  # vc_masking
        1,  # gaussian_noise
        1,  # pink_noise
        1,  # lowpass
        1,  # highpass
        1,  # bandpass
        1,  # echo
        1,  # smooth
        1,  # resample
        1,  # volume_boost
        1,  # volume_duck
        1,  # speed
        2,  # encodec_3kbps
        2,  # encodec_6kbps
        2,  # encodec_12kbps
    ]
    attack_name = random.choices(attacks, weights=weights, k=1)[0]

    if attack_name == "identity":
        augmented = audio
    elif attack_name == "vc_masking":
        bw = random.choice([3.0, 6.0, 12.0])
        comp = EncodecAttack.compress(audio, bw, sample_rate)
        augmented, vad_labels = apply_masking(comp, orig_audio)
    elif attack_name == "gaussian_noise":
        augmented = AudioEffects.random_noise(audio, 0.001)
    elif attack_name == "pink_noise":
        augmented = AudioEffects.pink_noise(audio, 0.01)
    elif attack_name == "lowpass":
        augmented = AudioEffects.lowpass_filter(audio, 5000, sample_rate)
    elif attack_name == "highpass":
        augmented = AudioEffects.highpass_filter(audio, 500, sample_rate)
    elif attack_name == "bandpass":
        augmented = AudioEffects.bandpass_filter(audio, 300, 8000, sample_rate)
    elif attack_name == "echo":
        augmented = AudioEffects.echo(audio, 0.3, 0.2, sample_rate)
    elif attack_name == "smooth":
        augmented = AudioEffects.smooth(audio, 5)
    elif attack_name == "resample":
        augmented = AudioEffects.updownresample(audio, sample_rate, 32000)
    elif attack_name == "volume_boost":
        augmented = AudioEffects.boost_audio(audio, 10)
    elif attack_name == "volume_duck":
        augmented = AudioEffects.duck_audio(audio, 10)
    elif attack_name == "speed":
        factor = random.choice([0.9, 1.1])
        augmented = AudioEffects.speed(audio, factor, sample_rate)
    elif attack_name == "encodec_3kbps":
        augmented = EncodecAttack.compress(audio, 3.0, sample_rate)
    elif attack_name == "encodec_6kbps":
        augmented = EncodecAttack.compress(audio, 6.0, sample_rate)
    elif attack_name == "encodec_12kbps":
        augmented = EncodecAttack.compress(audio, 12.0, sample_rate)
    else:
        augmented = audio

    augmented = _match_audio_length(augmented, T)
    return torch.clamp(augmented, -1.0, 1.0), vad_labels, attack_name


# -----------------------------------------------------------------------------
# 5. Full Validation Attack Suite (DSP + Neural Codecs)
# -----------------------------------------------------------------------------
def get_validation_attack_suite(sample_rate: int = 16000) -> List[Tuple[str, str, str, Callable[[torch.Tensor], torch.Tensor]]]:
    return [
        # DSP Attacks
        ("DSP", "Clean (Identity)", "", lambda wav: AudioEffects.identity(wav)),
        ("DSP", "Gaussian Noise", "", lambda wav: AudioEffects.random_noise(wav, 0.001)),
        ("DSP", "Pink Noise", "", lambda wav: AudioEffects.pink_noise(wav, 0.01)),
        ("DSP", "Lowpass Filter (5k)", "", lambda wav: AudioEffects.lowpass_filter(wav, 5000, sample_rate)),
        ("DSP", "Highpass Filter (500)", "", lambda wav: AudioEffects.highpass_filter(wav, 500, sample_rate)),
        ("DSP", "Bandpass Filter", "", lambda wav: AudioEffects.bandpass_filter(wav, 300, 8000, sample_rate)),
        ("DSP", "Echo/Reverb", "", lambda wav: AudioEffects.echo(wav, 0.3, 0.2, sample_rate)),
        ("DSP", "Smooth (Moving Avg)", "", lambda wav: AudioEffects.smooth(wav, 5)),
        ("DSP", "Resampling (32k-16k)", "", lambda wav: AudioEffects.updownresample(wav, sample_rate, 32000)),
        ("DSP", "Volume Boost (+10%)", "", lambda wav: AudioEffects.boost_audio(wav, 10)),
        ("DSP", "Volume Duck (-10%)", "", lambda wav: AudioEffects.duck_audio(wav, 10)),
        ("DSP", "Speed (0.8x-1.2x)", "", lambda wav: AudioEffects.speed(wav, 1.1, sample_rate)),

        # Codec Attacks
        ("Codec", "Encodec", "3 kbps", lambda wav: EncodecAttack.compress(wav, 3.0, sample_rate)),
        ("Codec", "Encodec", "6 kbps", lambda wav: EncodecAttack.compress(wav, 6.0, sample_rate)),
        ("Codec", "Encodec", "12 kbps", lambda wav: EncodecAttack.compress(wav, 12.0, sample_rate)),
        ("Codec", "DAC", "6 kbps", lambda wav: DACAttack.compress(wav, "16khz", sample_rate)),
        ("Codec", "DAC", "8 kbps", lambda wav: DACAttack.compress(wav, "44khz", sample_rate)),
        ("Codec", "DAC", "24 kbps", lambda wav: DACAttack.compress(wav, "24khz", sample_rate)),
        ("Codec", "SNAC", "0.98 kbps", lambda wav: SNACAttack.compress(wav, "24khz", sample_rate)),
        ("Codec", "SNAC", "1.9 kbps", lambda wav: SNACAttack.compress(wav, "32khz", sample_rate)),
        ("Codec", "SNAC", "2.6 kbps", lambda wav: SNACAttack.compress(wav, "44khz", sample_rate)),
    ]


def format_full_validation_table(step: int, results: Dict[str, Dict[str, float]], quality_metrics: Optional[Dict[str, float]] = None) -> str:
    lines = [
        "=" * 90,
        f"  NeuMark Validation Report (Step: {step:07d})",
        "=" * 90,
        f"{'Attack Type':<30} | {'Pos ACC(TP)':<12} | {'Neg ACC(TN)':<12} | {'Detect ACC':<12} | {'WM Bit Acc':<12}",
        "-" * 90,
    ]

    dsp_pos, dsp_neg, dsp_accs, dsp_bits = [], [], [], []
    codec_pos, codec_neg, codec_accs, codec_bits = [], [], [], []

    # 1. Print DSP Section
    for name, stats in results.items():
        if stats["category"] == "DSP":
            pos_acc = stats.get("pos_acc", 0.0)
            neg_acc = stats.get("neg_acc", 0.0)
            det_acc = stats["detect_acc"]
            bit_acc = stats["bit_acc"]
            dsp_pos.append(pos_acc)
            dsp_neg.append(neg_acc)
            dsp_accs.append(det_acc)
            dsp_bits.append(bit_acc)
            lines.append(f"{name:<30} | {pos_acc:<12.4f} | {neg_acc:<12.4f} | {det_acc:<12.4f} | {bit_acc:<12.4f}")

    if dsp_accs:
        avg_dsp_pos = sum(dsp_pos) / len(dsp_pos)
        avg_dsp_neg = sum(dsp_neg) / len(dsp_neg)
        avg_dsp_det = sum(dsp_accs) / len(dsp_accs)
        avg_dsp_bit = sum(dsp_bits) / len(dsp_bits)
        lines.append(f"{'DSP Avg.':<30} | {avg_dsp_pos:<12.4f} | {avg_dsp_neg:<12.4f} | {avg_dsp_det:<12.4f} | {avg_dsp_bit:<12.4f}")
        lines.append("-" * 90)

    # 2. Print Codec Section (Grouped by Family)
    codec_families = ["Encodec", "DAC", "SNAC"]
    for family in codec_families:
        first = True
        for name, stats in results.items():
            if stats["category"] == "Codec" and stats["family"] == family:
                pos_acc = stats.get("pos_acc", 0.0)
                neg_acc = stats.get("neg_acc", 0.0)
                det_acc = stats["detect_acc"]
                bit_acc = stats["bit_acc"]
                codec_pos.append(pos_acc)
                codec_neg.append(neg_acc)
                codec_accs.append(det_acc)
                codec_bits.append(bit_acc)
                bitrate = stats["bitrate"]
                if first:
                    label = f"{family:<14} {bitrate}"
                    first = False
                else:
                    label = f"{'':<14} {bitrate}"
                lines.append(f"{label:<30} | {pos_acc:<12.4f} | {neg_acc:<12.4f} | {det_acc:<12.4f} | {bit_acc:<12.4f}")
        lines.append("")

    if lines[-1] == "":
        lines.pop()

    if codec_accs:
        avg_codec_pos = sum(codec_pos) / len(codec_pos)
        avg_codec_neg = sum(codec_neg) / len(codec_neg)
        avg_codec_det = sum(codec_accs) / len(codec_accs)
        avg_codec_bit = sum(codec_bits) / len(codec_bits)
        lines.append(f"{'Codec Avg.':<30} | {avg_codec_pos:<12.4f} | {avg_codec_neg:<12.4f} | {avg_codec_det:<12.4f} | {avg_codec_bit:<12.4f}")
        lines.append("-" * 90)

    all_pos = dsp_pos + codec_pos
    all_neg = dsp_neg + codec_neg
    all_accs = dsp_accs + codec_accs
    all_bits = dsp_bits + codec_bits
    if all_accs:
        total_avg_pos = sum(all_pos) / len(all_pos)
        total_avg_neg = sum(all_neg) / len(all_neg)
        total_avg_det = sum(all_accs) / len(all_accs)
        total_avg_bit = sum(all_bits) / len(all_bits)
        lines.append(f"{'Avg.':<30} | {total_avg_pos:<12.4f} | {total_avg_neg:<12.4f} | {total_avg_det:<12.4f} | {total_avg_bit:<12.4f}")

    if quality_metrics:
        lines.append("=" * 80)
        lines.append("  Speech Quality & Fidelity Degradation (Clean TTS vs. NeuMark Watermarked):")
        lines.append("-" * 80)
        lines.append(f"{'Metric':<25} | {'Clean TTS':<12} | {'Watermarked':<12} | {'Delta (WM - Clean)':<18}")
        lines.append("-" * 80)

        # UTMOS
        c_ut = quality_metrics.get("clean_utmos", 0.0)
        w_ut = quality_metrics.get("wm_utmos", 0.0)
        d_ut = w_ut - c_ut
        lines.append(f"{'UTMOS (MOS 1.0 - 5.0)':<25} | {c_ut:<12.4f} | {w_ut:<12.4f} | {d_ut:+12.4f}")

        # SIM
        c_sim = quality_metrics.get("clean_sim", 0.0)
        w_sim = quality_metrics.get("wm_sim", 0.0)
        d_sim = w_sim - c_sim
        lines.append(f"{'SIM (Speaker Cosine Sim)':<25} | {c_sim:<12.4f} | {w_sim:<12.4f} | {d_sim:+12.4f}")

        # WER
        c_wer = quality_metrics.get("clean_wer", 0.0)
        w_wer = quality_metrics.get("wm_wer", 0.0)
        d_wer = w_wer - c_wer
        lines.append(f"{'ASR WER (Word Error Rate)':<25} | {c_wer:<12.4f} | {w_wer:<12.4f} | {d_wer:+12.4f}")

        # CER
        c_cer = quality_metrics.get("clean_cer", 0.0)
        w_cer = quality_metrics.get("wm_cer", 0.0)
        d_cer = w_cer - c_cer
        lines.append(f"{'ASR CER (Char Error Rate)':<25} | {c_cer:<12.4f} | {w_cer:<12.4f} | {d_cer:+12.4f}")

    lines.append("=" * 80)
    return "\n".join(lines)

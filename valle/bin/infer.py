#!/usr/bin/env python3
# Copyright    2023                            (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Phonemize Text and EnCodec Audio.

Usage example:
    python3 bin/infer.py \
        --decoder-dim 128 --nhead 4 --num-decoder-layers 4 --model-name valle \
        --text-prompts "Go to her." \
        --audio-prompts ./prompts/61_70970_000007_000001.wav \
        --output-dir infer/demo_valle_epoch20 \
        --checkpoint exp/valle_nano_v2/epoch-20.pt

"""
import argparse
import logging
import os
from pathlib import Path

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import torchaudio
import torchaudio.functional as AF
import numpy as np
from icefall.utils import AttributeDict, str2bool

try:
    from torchmetrics.functional.audio import pesq as tm_pesq
    from torchmetrics.functional.audio import stoi as tm_stoi
except Exception:
    tm_pesq = None
    tm_stoi = None

# torchmetrics versions differ; resolve callable fallbacks if imports yielded modules
if tm_pesq is not None and not callable(tm_pesq):
    try:
        from torchmetrics.functional.audio.pesq import pesq as tm_pesq_fn  # type: ignore

        tm_pesq = tm_pesq_fn
    except Exception:
        pass
if tm_stoi is not None and not callable(tm_stoi):
    try:
        from torchmetrics.functional.audio.stoi import stoi as tm_stoi_fn  # type: ignore

        tm_stoi = tm_stoi_fn
    except Exception:
        pass

try:
    from pesq import pesq as pesq_pkg
except Exception:
    pesq_pkg = None

try:
    from pystoi import stoi as stoi_pkg
except Exception:
    stoi_pkg = None

try:
    from .attacks import AudioEffects
except ImportError:
    # Fallback when running as a standalone script (no package context)
    from attacks import AudioEffects

from valle.data import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
)
from valle.data.collation import get_text_token_collater
from valle.models import get_model


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text-prompts",
        type=str,
        default="",
        help="Text prompts which are separated by |.",
    )

    parser.add_argument(
        "--audio-prompts",
        type=str,
        default="",
        help="Audio prompts which are separated by | and should be aligned with --text-prompts.",
    )

    parser.add_argument(
        "--text",
        type=str,
        default="To get up and running quickly just follow the steps below.",
        help="Text to be synthesized.",
    )

    parser.add_argument(
        "--text-file",
        type=str,
        default="",
        help="Path to a file with one text per line (overrides --text).",
    )

    # model
    # add_model_arguments(parser)
    # parser.add_argument(
    #     "--text-tokens",
    #     type=str,
    #     default="data/tokenized/unique_text_tokens.k2symbols",
    #     help="Path to the unique text tokens file.",
    # )

    parser.add_argument(
        "--text-extractor",
        type=str,
        default="espeak",
        help="espeak or pypinyin or pypinyin_initials_finals",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="exp/vallf_nano_full/checkpoint-100000.pt",
        help="Path to the saved checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("infer/demo"),
        help="Path to the tokenized files.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=-100,
        help="Whether AR Decoder do top_k(if > 0) sampling.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature of AR Decoder top_k sampling.",
    )

    parser.add_argument(
        "--continual",
        type=str2bool,
        default=False,
        help="Do continual task.",
    )

    parser.add_argument(
        "--ts-enable",
        type=str2bool,
        default=False,
        help="Legacy switch for --watermark-backend traceablespeech.",
    )

    parser.add_argument(
        "--ts-checkpoint-file",
        type=str,
        default="traceableSpeech/g_00150000",
        help="The checkpoint file of TraceableSpeech model.",
    )

    parser.add_argument(
        "--watermark-backend",
        type=str,
        default="encodec",
        choices=["encodec", "traceablespeech", "voicemark"],
        help="Codec/watermark backend used by AudioTokenizer.",
    )

    parser.add_argument(
        "--voicemark-root",
        type=str,
        default="/home/wu25/mrnas04home/projects/VoiceMark",
        help="Path to the VoiceMark project root.",
    )

    parser.add_argument(
        "--voicemark-config",
        type=str,
        default="STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json",
        help="VoiceMark SpeechTokenizer config, absolute or relative to --voicemark-root.",
    )

    parser.add_argument(
        "--voicemark-st-checkpoint",
        type=str,
        default="STmodels/pretrained_model/SpeechTokenizer.pt",
        help="VoiceMark SpeechTokenizer checkpoint, absolute or relative to --voicemark-root.",
    )

    parser.add_argument(
        "--voicemark-checkpoint",
        type=str,
        default="train/Log/spt_base/20260601-123358/WatermarkTrainer_final_00150000.pt",
        help="VoiceMark watermark checkpoint, absolute or relative to --voicemark-root.",
    )

    parser.add_argument(
        "--voicemark-embed-vq1",
        type=str2bool,
        default=True,
        help="Whether VoiceMark embeds/detects watermark in VQ1-8 instead of VQ2-8.",
    )

    return parser.parse_args()


def load_model(checkpoint, device):
    if not checkpoint:
        return None

    checkpoint = torch.load(checkpoint, map_location=device)

    args = AttributeDict(checkpoint)
    model = get_model(args)

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model"], strict=True
    )
    assert not missing_keys
    model.to(device)
    model.eval()

    text_tokens = args.text_tokens

    return model, text_tokens


def _ensure_mono_and_resample(waveform: torch.Tensor, orig_sr: int, target_sr: int):
    if waveform.dim() == 3:
        waveform = waveform.squeeze(1)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if orig_sr != target_sr:
        waveform = AF.resample(waveform, orig_sr, target_sr)
    return waveform


def compute_pesq_stoi(ref_waveform: torch.Tensor, test_waveform: torch.Tensor, orig_sr: int):
    target_sr = 16000
    ref_rs = _ensure_mono_and_resample(ref_waveform.cpu(), orig_sr, target_sr)
    test_rs = _ensure_mono_and_resample(test_waveform.cpu(), orig_sr, target_sr)
    # convert to numpy float32 and length-align by padding shorter one
    ref_np = ref_rs.squeeze().detach().cpu().numpy().astype("float32")
    test_np = test_rs.squeeze().detach().cpu().numpy().astype("float32")
    # Trim leading/trailing all-zero regions, then length-align by padding
    def _trim_silence(x: np.ndarray):
        if not x.any():
            return x
        nz = np.flatnonzero(x)
        return x[nz[0] : nz[-1] + 1]

    ref_np = _trim_silence(ref_np)
    test_np = _trim_silence(test_np)

    max_len = max(len(ref_np), len(test_np))
    if len(ref_np) < max_len:
        ref_np = torch.nn.functional.pad(
            torch.from_numpy(ref_np), (0, max_len - len(ref_np))
        ).numpy()
    if len(test_np) < max_len:
        test_np = torch.nn.functional.pad(
            torch.from_numpy(test_np), (0, max_len - len(test_np))
        ).numpy()

    min_required = int(1.125 * target_sr)
    if max_len < min_required:
        logging.warning("Audio too short for PESQ/STOI; skip metrics.")
        return None, None

    # Avoid all-silence inputs that trigger PESQ 'No utterances detected'
    if np.max(np.abs(ref_np)) < 1e-6 or np.max(np.abs(test_np)) < 1e-6:
        logging.warning("Audio near-silent; skip PESQ/STOI.")
        return None, None

    pesq_score = None
    stoi_score = None

    # Try torchmetrics first
    pesq_fn = tm_pesq if callable(tm_pesq) else None
    if pesq_fn is None and hasattr(tm_pesq, "pesq"):
        pesq_fn = getattr(tm_pesq, "pesq")
    stoi_fn = tm_stoi if callable(tm_stoi) else None
    if stoi_fn is None and hasattr(tm_stoi, "stoi"):
        stoi_fn = getattr(tm_stoi, "stoi")

    if callable(pesq_fn):
        try:
            pesq_score = pesq_fn(ref_rs, test_rs, target_sr, "wb")
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"torchmetrics pesq failed: {exc}")
    if callable(stoi_fn):
        try:
            stoi_score = stoi_fn(ref_rs, test_rs, target_sr, extended=False)
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"torchmetrics stoi failed: {exc}")

    # Fallback to pesq/pystoi packages if still missing
    if pesq_score is None and pesq_pkg is not None:
        try:
            pesq_score = pesq_pkg(target_sr, ref_np, test_np, "wb")
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"pesq package failed: {exc}")
    if stoi_score is None and stoi_pkg is not None:
        try:
            stoi_score = stoi_pkg(ref_np, test_np, target_sr, extended=False)
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"pystoi package failed: {exc}")

    if pesq_score is None or stoi_score is None:
        logging.warning("PESQ/STOI unavailable; skip metrics.")
        return None, None

    return float(pesq_score), float(stoi_score)


@torch.no_grad()
def main():
    args = get_args()
    text_tokenizer = TextTokenizer(backend=args.text_extractor)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    model, text_tokens = load_model(args.checkpoint, device)
    text_collater = get_text_token_collater(text_tokens)

    watermark_backend = args.watermark_backend
    if args.ts_enable and watermark_backend == "encodec":
        watermark_backend = "traceablespeech"

    ts_config = Path(args.ts_checkpoint_file).parent / "config.json"
    audio_tokenizer = AudioTokenizer(
        enable_ts=args.ts_enable,
        ts_checkpoint=args.ts_checkpoint_file,
        ts_config=str(ts_config),
        watermark_backend=watermark_backend,
        voicemark_root=args.voicemark_root,
        voicemark_config=args.voicemark_config,
        voicemark_st_checkpoint=args.voicemark_st_checkpoint,
        voicemark_checkpoint=args.voicemark_checkpoint,
        voicemark_embed_vq1=args.voicemark_embed_vq1,
    )
    audio_sample_rate = audio_tokenizer.sample_rate

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    text_prompts = " ".join(args.text_prompts.split("|"))

    audio_prompts = []
    if args.audio_prompts:
        for n, audio_file in enumerate(args.audio_prompts.split("|")):
            encoded_frames = tokenize_audio(audio_tokenizer, audio_file)
            if False:
                samples = audio_tokenizer.decode(encoded_frames)
                torchaudio.save(
                    f"{args.output_dir}/p{n}.wav", samples[0], audio_sample_rate
                )

            audio_prompts.append(encoded_frames[0][0])

        assert len(args.text_prompts.split("|")) == len(audio_prompts)
        # import pdb; pdb.set_trace()
        audio_prompts = torch.concat(audio_prompts, dim=-1).transpose(2, 1)
        audio_prompts = audio_prompts.to(device)

    texts = None
    # accumulators for batch metrics
    pesq_list = []
    stoi_list = []
    wm_bits_correct = 0
    wm_bits_total = 0
    # Old TraceableSpeech attack settings (kept for reference)
    # attack_ops = [
    #     ("CLP", 0.13),
    #     ("RSP-90", 0.15),
    #     ("Noise-W35", 0.14),
    #     ("SS-01", 0.15),
    #     ("AS-90", 0.15),
    #     ("EA-0301", 0.14),
    #     ("LP5000", 0.14),
    # ]
    # attack_stats = {
    #     "N": {op: {"count": 0, "bits_correct": 0, "bits_total": 0} for op, _ in attack_ops},
    #     "Y": {op: {"count": 0, "bits_correct": 0, "bits_total": 0} for op, _ in attack_ops},
    # }

    # New attacks powered by AudioEffects. Speed fixed to 1.2.
    attack_fns = [
        ("speed", lambda wav: AudioEffects.speed(wav, speed_range=(1.2, 1.2), sample_rate=audio_sample_rate)),
        ("updownresample", lambda wav: AudioEffects.updownresample(wav, sample_rate=audio_sample_rate)),
        ("echo", lambda wav: AudioEffects.echo(wav, sample_rate=audio_sample_rate)),
        ("random_noise", lambda wav: AudioEffects.random_noise(wav)),
        ("pink_noise", lambda wav: AudioEffects.pink_noise(wav)),
        ("lowpass_filter", lambda wav: AudioEffects.lowpass_filter(wav, sample_rate=audio_sample_rate)),
        ("highpass_filter", lambda wav: AudioEffects.highpass_filter(wav, sample_rate=audio_sample_rate)),
        ("bandpass_filter", lambda wav: AudioEffects.bandpass_filter(wav, sample_rate=audio_sample_rate)),
        ("smooth", lambda wav: AudioEffects.smooth(wav)),
        ("boost_audio", lambda wav: AudioEffects.boost_audio(wav)),
        ("duck_audio", lambda wav: AudioEffects.duck_audio(wav)),
        ("identity", lambda wav: AudioEffects.identity(wav)),
        ("shush", lambda wav: AudioEffects.shush(wav)),
        ("encodec", lambda wav: AudioEffects.encodec(wav, sample_rate=audio_sample_rate)),
    ]
    attack_stats = {name: {"count": 0, "bits_correct": 0, "bits_total": 0} for name, _ in attack_fns}

    if args.text_file:
        with open(args.text_file) as f:
            texts = [ln.strip() for ln in f if ln.strip()]
    elif os.path.isfile(args.text):  # legacy demo file with tab-separated fields
        # https://github.com/lifeiteng/lifeiteng.github.com/blob/main/valle/prepare.py
        with open(args.text) as f:
            for line in f:
                fields = line.strip().split("\t")
                assert len(fields) == 4
                prompt_text, prompt_audio, text, audio_path = fields
                logging.info(f"synthesize text: {text}")
                text_tokens, text_tokens_lens = text_collater(
                    [
                        tokenize_text(
                            text_tokenizer, text=f"{prompt_text} {text}".strip()
                        )
                    ]
                )
                _, enroll_x_lens = text_collater(
                    [
                        tokenize_text(
                            text_tokenizer, text=f"{prompt_text}".strip()
                        )
                    ]
                )

                audio_prompts = tokenize_audio(audio_tokenizer, prompt_audio)
                audio_prompts = audio_prompts[0][0].transpose(2, 1).to(device)

                # synthesis
                encoded_frames = model.inference(
                    text_tokens.to(device),
                    text_tokens_lens.to(device),
                    audio_prompts,
                    enroll_x_lens=enroll_x_lens,
                    top_k=args.top_k,
                    temperature=args.temperature,
                )

                samples = audio_tokenizer.decode(
                    [(encoded_frames.transpose(2, 1), None)]
                )
                # store
                torchaudio.save(audio_path, samples[0].cpu(), audio_sample_rate)
        return
    else:
        texts = args.text.split("|")

    for n, text in enumerate(texts):
        logging.info(f"synthesize text: {text}")
        text_tokens, text_tokens_lens = text_collater(
            [
                tokenize_text(
                    text_tokenizer, text=f"{text_prompts} {text}".strip()
                )
            ]
        )

        # synthesis
        if args.continual:
            assert text == ""
            encoded_frames = model.continual(
                text_tokens.to(device),
                text_tokens_lens.to(device),
                audio_prompts,
            )
        else:
            enroll_x_lens = None
            if text_prompts:
                _, enroll_x_lens = text_collater(
                    [
                        tokenize_text(
                            text_tokenizer, text=f"{text_prompts}".strip()
                        )
                    ]
                )
            encoded_frames = model.inference(
                text_tokens.to(device),
                text_tokens_lens.to(device),
                audio_prompts, # audio_prompts: [B, 8, T/320]
                enroll_x_lens=enroll_x_lens,
                top_k=args.top_k,
                temperature=args.temperature,
            )
        # import pdb; pdb.set_trace()
        if audio_prompts != []:
            # Guard against empty AR output (early EOS). The time dimension is dim=1 for encoded_frames [B, T, n_q].
            time_len = encoded_frames.shape[1] if encoded_frames.dim() >= 2 else 0
            if time_len == 0 or encoded_frames.numel() == 0:
                logging.warning(f"Skip utt {n}: empty encoded frames (early EOS).")
                continue

            encoded_for_decode = [(encoded_frames.transpose(2, 1), None)]

            # Extra guard: after transpose we expect shape [B, n_q, T]; check T.
            seq_len = encoded_for_decode[0][0].shape[2] if encoded_for_decode[0][0].dim() >= 3 else 0
            if seq_len == 0:
                logging.warning(f"Skip utt {n}: zero-length sequence after transpose (early EOS).")
                continue

            watermark_sign = audio_tokenizer.random_watermark(encoded_frames.size(0))

            # decode both reference (zero watermark) and watermarked in one call
            decoded_clean, decoded_wm = audio_tokenizer.decode_pair(
                encoded_for_decode, watermark_sign=watermark_sign
            )

            clean_path = Path(args.output_dir) / f"{n}_clean.wav"
            torchaudio.save(str(clean_path), decoded_clean[0].cpu(), audio_sample_rate)

            if decoded_wm is not None:
                wm_path = Path(args.output_dir) / f"{n}_wm.wav"
                torchaudio.save(str(wm_path), decoded_wm[0].cpu(), audio_sample_rate)

                pesq_score, stoi_score = compute_pesq_stoi(
                    decoded_clean, decoded_wm, audio_sample_rate
                )
                if pesq_score is not None and stoi_score is not None:
                    logging.info(
                        f"PESQ(clean vs watermark)={pesq_score:.4f}, STOI={stoi_score:.4f}"
                    )
                    pesq_list.append(pesq_score)
                    stoi_list.append(stoi_score)

                detection = audio_tokenizer.detect_watermark(decoded_wm)
                if detection is not None and watermark_sign is not None:
                    detect_prob, sign_pred, _ = detection
                    bits_total = watermark_sign.numel()
                    bits_match = (
                        sign_pred == watermark_sign.to(sign_pred.device)
                    ).sum().item()
                    logging.info(
                        f"Watermark prob={detect_prob.mean().item():.4f}, bits {bits_match}/{bits_total} (utt {n})"
                    )
                    wm_bits_correct += bits_match
                    wm_bits_total += bits_total

                    # Attack-based watermark robustness check using AudioEffects
                    for attack_name, attack_fn in attack_fns:
                        attacked = attack_fn(decoded_wm.clone())
                        if not isinstance(attacked, torch.Tensor):
                            # AudioEffects can return (tensor, mask); only tensor is needed here.
                            attacked = attacked[0]

                        attacked_detection = audio_tokenizer.detect_watermark(attacked)
                        if attacked_detection is None:
                            continue
                        _, sign_pred_attacked, _ = attacked_detection
                        bits_match_attacked = (
                            sign_pred_attacked
                            == watermark_sign.to(sign_pred_attacked.device)
                        ).sum().item()
                        attack_stats[attack_name]["count"] += 1
                        attack_stats[attack_name]["bits_correct"] += bits_match_attacked
                        attack_stats[attack_name]["bits_total"] += bits_total
                        logging.info(
                            f"Watermark attack {attack_name}: bits {bits_match_attacked}/{bits_total} (utt {n})"
                        )
            else:
                logging.warning(
                    "Watermark backend unavailable; metrics skipped."
                )
        else:  # Transformer
            pass

    # batch summaries
    if pesq_list and stoi_list:
        logging.info(
            f"PESQ avg={np.mean(pesq_list):.4f}, min={np.min(pesq_list):.4f}, max={np.max(pesq_list):.4f}"
        )
        logging.info(
            f"STOI avg={np.mean(stoi_list):.4f}, min={np.min(stoi_list):.4f}, max={np.max(stoi_list):.4f}"
        )
    if wm_bits_total > 0:
        wm_acc = wm_bits_correct / wm_bits_total
        logging.info(
            f"Watermark bit accuracy avg={wm_acc:.4f} ({wm_bits_correct}/{wm_bits_total})"
        )
    total_attack_bits = 0
    total_attack_correct = 0
    for attack_name, stats in attack_stats.items():
        if stats["bits_total"] > 0:
            acc = stats["bits_correct"] / stats["bits_total"]
            total_attack_bits += stats["bits_total"]
            total_attack_correct += stats["bits_correct"]
            logging.info(
                f"Attack {attack_name}: acc={acc:.4f} ({stats['bits_correct']}/{stats['bits_total']})"
            )
    if total_attack_bits > 0:
        overall_attack_acc = total_attack_correct / total_attack_bits
        logging.info(
            f"Attack overall: acc={overall_attack_acc:.4f} ({total_attack_correct}/{total_attack_bits})"
        )


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)
if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()

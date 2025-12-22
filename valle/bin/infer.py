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
import json
import logging
import os
import sys
from pathlib import Path

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import torchaudio
from icefall.utils import AttributeDict, str2bool

from valle.data import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
)
from valle.data.collation import get_text_token_collater
from valle.models import get_model

# TraceableSpeech watermark imports are loaded lazily because the project lives
# outside this repo. Keep the symbols local to avoid import errors when the
# user does not request watermarking.
TS_DEPENDENCIES = None


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

    # model
    # add_model_arguments(parser)
    # parser.add_argument(
    #     "--text-tokens",
    #     type=str,
    #     default="data/tokenized/unique_text_tokens.k2symbols",
    #     help="Path to the unique text tokens file.",
    # )

    # TraceableSpeech watermarking
    parser.add_argument(
        "--ts-enable",
        type=str2bool,
        default=False,
        help="Enable TraceableSpeech watermark embed + detect on generated audio.",
    )
    parser.add_argument(
        "--ts-checkpoint-file",
        type=str,
        default="/home/wu25/mrnas04home/projects/TraceableSpeech/save_model320/g_00150000",
        help="Path to TraceableSpeech checkpoint (g_*.pt).",
    )
    parser.add_argument(
        "--ts-sample-num",
        type=int,
        default=5,
        help="Number of random watermarks to embed for accuracy stats per utterance.",
    )
    parser.add_argument(
        "--ts-bit-num",
        type=int,
        default=4,
        help="Watermark bit length (used for accuracy calculation).",
    )


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


def load_traceablespeech(ts_checkpoint_file: str, device: torch.device):
    """Load TraceableSpeech models + config lazily.

    Returns a dict with models and helpers or None if path missing.
    """

    ckpt_path = Path(ts_checkpoint_file).expanduser().resolve()
    if not ckpt_path.is_file():
        logging.warning(f"TraceableSpeech checkpoint not found: {ckpt_path}")
        return None

    ts_root = ckpt_path.parent
    project_root = ts_root.parent  # TraceableSpeech project dir that holds env.py

    # Make sure both the checkpoint dir (config) and project dir (env.py, models.py) are importable
    for p in (ts_root, project_root):
        if str(p) not in sys.path:
            sys.path.append(str(p))

    if not (project_root / "env.py").is_file():
        logging.error(
            f"Cannot find env.py under TraceableSpeech project root: {project_root}. "
            "Please set --ts-checkpoint-file to a path inside the TraceableSpeech project or adjust the project root."
        )
        return None

    try:
        from env import AttrDict
        from meldataset import MAX_WAV_VALUE, mel_spectrogram
        from models import Encoder, Generator, Quantizer
        from watermark import (
            Random_watermark,
            Watermark_Decoder,
            Watermark_Encoder,
            attack,
            clip,
            sign_loss,
        )
    except ModuleNotFoundError as e:
        logging.error(
            "Failed to import TraceableSpeech modules. Ensure TraceableSpeech project root "
            f"({project_root}) is accessible and contains env.py/models.py/etc. Original error: {e}"
        )
        return None

    config_file = ts_root / "config.json"
    if not config_file.is_file():
        logging.warning(f"TraceableSpeech config not found: {config_file}")
        return None

    with open(config_file) as f:
        h = AttrDict(json.loads(f.read()))

    checkpoint_dict = torch.load(ckpt_path, map_location=device)

    generator = Generator(h).to(device)
    encoder = Encoder(h).to(device)
    quantizer_audio = Quantizer(h, "Audio").to(device)
    watermark_encoder = Watermark_Encoder(h).to(device)
    watermark_decoder = Watermark_Decoder(h).to(device)

    generator.load_state_dict(checkpoint_dict["generator"])
    encoder.load_state_dict(checkpoint_dict["encoder"])
    quantizer_audio.load_state_dict(checkpoint_dict["quantizer_Audio"])
    watermark_encoder.load_state_dict(checkpoint_dict["watermark_encoder"])
    watermark_decoder.load_state_dict(checkpoint_dict["watermark_decoder"])

    generator.eval(); generator.remove_weight_norm()
    encoder.eval(); encoder.remove_weight_norm()
    quantizer_audio.eval()
    watermark_encoder.eval()
    watermark_decoder.eval()

    attack_plan = [
        ("CLP", 0.13),
        ("RSP-90", 0.15),
        ("Noise-W35", 0.14),
        ("SS-01", 0.15),
        ("AS-90", 0.15),
        ("EA-0301", 0.14),
        ("LP5000", 0.14),
    ]

    return {
        "h": h,
        "generator": generator,
        "encoder": encoder,
        "quantizer": quantizer_audio,
        "watermark_encoder": watermark_encoder,
        "watermark_decoder": watermark_decoder,
        "Random_watermark": Random_watermark,
        "attack": attack,
        "clip": clip,
        "sign_loss": sign_loss,
        "mel_spectrogram": mel_spectrogram,
        "MAX_WAV_VALUE": MAX_WAV_VALUE,
        "attack_plan": attack_plan,
    }


def count_common_elements(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> int:
    cnt = 0
    for i in range(tensor_a.size(1)):
        if tensor_a[0][i] == tensor_b[0][i]:
            cnt += 1
    return cnt


def apply_traceablespeech_watermark(
    wav: torch.Tensor,
    ts_state: dict,
    sample_num: int,
    bit_num: int,
    device: torch.device,
):
    """Embed watermark on a single wav tensor [1, 1, T] and collect stats."""

    h = ts_state["h"]
    encoder = ts_state["encoder"]
    generator = ts_state["generator"]
    quantizer = ts_state["quantizer"]
    watermark_encoder = ts_state["watermark_encoder"]
    watermark_decoder = ts_state["watermark_decoder"]
    Random_watermark = ts_state["Random_watermark"]
    attack = ts_state["attack"]
    clip = ts_state["clip"]
    sign_loss = ts_state["sign_loss"]
    mel_spectrogram = ts_state["mel_spectrogram"]

    wav = wav.to(device)
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)
    if wav.shape[1] != 1:
        raise ValueError("TraceableSpeech expects mono audio [B, 1, T]")

    en_y = encoder(wav)
    q, _, c = quantizer(en_y)
    q = torch.stack([code.reshape(q.size(0), -1) for code in c], -1)
    q = quantizer.embed(q, h.Audio["infer_need_layer"])

    first_audio = None
    stats = []
    total_correct = 0
    total_bits = 0

    for idx in range(sample_num):
        sign = Random_watermark(1).to(device)
        sign_trait = watermark_encoder(sign)
        y_g_hat = generator(q, sign_trait)

        if first_audio is None:
            first_audio = y_g_hat.detach().cpu()

        y_g_hat, clip_flag = clip(y_g_hat)
        y_g_hat, opera = attack(y_g_hat, ts_state["attack_plan"])

        # STFT on GPU sometimes triggers cuFFT_INTERNAL_ERROR; run mel on CPU, then move back to device for the decoder.
        y_g_hat_mel = mel_spectrogram(
            y_g_hat.squeeze(1).cpu(),
            h.n_fft,
            h.num_mels,
            h.sampling_rate,
            h.hop_size,
            h.win_size,
            h.fmin,
            h.fmax_for_loss,
        ).to(device)

        sign_score, sign_pred = watermark_decoder(y_g_hat_mel)
        loss = sign_loss(sign_score, sign)

        correct_bits = count_common_elements(sign, sign_pred)
        total_correct += correct_bits
        total_bits += bit_num

        stats.append(
            {
                "clip": clip_flag,
                "opera": opera,
                "correct_bits": correct_bits,
                "total_bits": bit_num,
                "loss": float(loss.item()),
            }
        )

    summary = {
        "accuracy": total_correct / total_bits if total_bits else 0.0,
        "samples": stats,
    }

    return first_audio, summary


def save_audio_with_watermark(
    samples: torch.Tensor,
    base_sr: int,
    output_path: Path,
    args,
    ts_state,
    device: torch.device,
):
    """Save plain VALL-E audio, plus watermarked version if enabled.

    Args:
      samples: Tensor shaped [B, C, T] from AudioTokenizer.decode.
    """

    # Save plain audio
    torchaudio.save(str(output_path), samples[0].cpu(), base_sr)

    if not args.ts_enable or ts_state is None:
        return None

    wm_audio, wm_summary = apply_traceablespeech_watermark(
        samples, ts_state, args.ts_sample_num, args.ts_bit_num, device
    )

    if wm_audio is None:
        return None

    wm_output = output_path.with_suffix(".wm.wav")
    audio = wm_audio.squeeze()
    audio = audio * ts_state["MAX_WAV_VALUE"]
    audio = audio.cpu().numpy().astype("int16")
    torchaudio.save(str(wm_output), torch.from_numpy(audio).unsqueeze(0), ts_state["h"].sampling_rate)

    logging.info(
        f"Watermark saved to {wm_output}, accuracy={wm_summary['accuracy']:.4f} over {args.ts_sample_num} samples"
    )
    for idx, item in enumerate(wm_summary["samples"]):
        logging.info(
            f"WM sample {idx}: opera={item['opera']} clip={item['clip']} correct={item['correct_bits']}/{item['total_bits']} loss={item['loss']:.4f}"
        )

    return wm_summary


@torch.no_grad()
def main():
    args = get_args()
    text_tokenizer = TextTokenizer(backend=args.text_extractor)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    model, text_tokens = load_model(args.checkpoint, device)
    text_collater = get_text_token_collater(text_tokens)

    audio_tokenizer = AudioTokenizer()

    ts_state = None
    if args.ts_enable:
        ts_state = load_traceablespeech(args.ts_checkpoint_file, device)
        if ts_state is None:
            logging.warning("TraceableSpeech watermark is enabled but failed to load; continuing without watermarking.")
            args.ts_enable = False

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    text_prompts = " ".join(args.text_prompts.split("|"))

    audio_prompts = []
    if args.audio_prompts:
        for n, audio_file in enumerate(args.audio_prompts.split("|")):
            encoded_frames = tokenize_audio(audio_tokenizer, audio_file)
            if False:
                samples = audio_tokenizer.decode(encoded_frames)
                torchaudio.save(
                    f"{args.output_dir}/p{n}.wav", samples[0], 24000
                )

            audio_prompts.append(encoded_frames[0][0]) #encoded_frames[0][0]: (batch_size = 1, num_q = 8, T/320)

        assert len(args.text_prompts.split("|")) == len(audio_prompts)
        audio_prompts = torch.concat(audio_prompts, dim=-1).transpose(2, 1)
        audio_prompts = audio_prompts.to(device) #(1, T/320, num_q = 8)

    if os.path.isfile(args.text):  # for demos
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
                save_audio_with_watermark(
                    samples=samples,
                    base_sr=24000,
                    output_path=Path(audio_path),
                    args=args,
                    ts_state=ts_state,
                    device=device,
                )
        return

    for n, text in enumerate(args.text.split("|")):
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
                audio_prompts,
                enroll_x_lens=enroll_x_lens,
                top_k=args.top_k,
                temperature=args.temperature,
            )

        if audio_prompts != []:
            samples = audio_tokenizer.decode(
                [(encoded_frames.transpose(2, 1), None)]
            )
            save_audio_with_watermark(
                samples=samples,
                base_sr=24000,
                output_path=Path(f"{args.output_dir}/{n}.wav"),
                args=args,
                ts_state=ts_state,
                device=device,
            )
        else:  # Transformer
            pass


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

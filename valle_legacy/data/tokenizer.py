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

from email import parser
from pyexpat.errors import codes
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Pattern, Union


import numpy as np
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from lhotse.features import FeatureExtractor
from lhotse.utils import Seconds, compute_num_frames
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator

import os
import argparse
import json
# TraceableSpeech project lives at vall-e/traceableSpeech; imports are lowercase on disk.
from traceableSpeech.env import AttrDict
from traceableSpeech.models import Generator, Encoder, Quantizer
from traceableSpeech.watermark import Watermark_Encoder, Watermark_Decoder, Random_watermark


try:
    from pypinyin import Style, pinyin
    from pypinyin.style._utils import get_finals, get_initials
except Exception:
    pass


class PypinyinBackend:
    """PypinyinBackend for Chinese. Most codes is referenced from espnet.
    There are two types pinyin or initials_finals, one is
    just like "ni1 hao3", the other is like "n i1 h ao3".
    """

    def __init__(
        self,
        backend="initials_finals",
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
    ) -> None:
        self.backend = backend
        self.punctuation_marks = punctuation_marks

    def phonemize(
        self, text: List[str], separator: Separator, strip=True, njobs=1
    ) -> List[str]:
        assert isinstance(text, List)
        phonemized = []
        for _text in text:
            _text = re.sub(" +", " ", _text.strip())
            _text = _text.replace(" ", separator.word)
            phones = []
            if self.backend == "pypinyin":
                for n, py in enumerate(
                    pinyin(
                        _text, style=Style.TONE3, neutral_tone_with_five=True
                    )
                ):
                    if all([c in self.punctuation_marks for c in py[0]]):
                        if len(phones):
                            assert phones[-1] == separator.syllable
                            phones.pop(-1)

                        phones.extend(list(py[0]))
                    else:
                        phones.extend([py[0], separator.syllable])
            elif self.backend == "pypinyin_initials_finals":
                for n, py in enumerate(
                    pinyin(
                        _text, style=Style.TONE3, neutral_tone_with_five=True
                    )
                ):
                    if all([c in self.punctuation_marks for c in py[0]]):
                        if len(phones):
                            assert phones[-1] == separator.syllable
                            phones.pop(-1)
                        phones.extend(list(py[0]))
                    else:
                        if py[0][-1].isalnum():
                            initial = get_initials(py[0], strict=False)
                            if py[0][-1].isdigit():
                                final = (
                                    get_finals(py[0][:-1], strict=False)
                                    + py[0][-1]
                                )
                            else:
                                final = get_finals(py[0], strict=False)
                            phones.extend(
                                [
                                    initial,
                                    separator.phone,
                                    final,
                                    separator.syllable,
                                ]
                            )
                        else:
                            assert ValueError
            else:
                raise NotImplementedError
            phonemized.append(
                "".join(phones).rstrip(f"{separator.word}{separator.syllable}")
            )
        return phonemized


class TextTokenizer:
    """Phonemize Text."""

    def __init__(
        self,
        language="en-us",
        backend="espeak",
        separator=Separator(word="_", syllable="-", phone="|"),
        preserve_punctuation=True,
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
        with_stress: bool = False,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = "keep-flags",
        words_mismatch: WordMismatch = "ignore",
    ) -> None:
        if backend == "espeak":
            phonemizer = EspeakBackend(
                language,
                punctuation_marks=punctuation_marks,
                preserve_punctuation=preserve_punctuation,
                with_stress=with_stress,
                tie=tie,
                language_switch=language_switch,
                words_mismatch=words_mismatch,
            )
        elif backend in ["pypinyin", "pypinyin_initials_finals"]:
            phonemizer = PypinyinBackend(
                backend=backend,
                punctuation_marks=punctuation_marks + separator.word,
            )
        else:
            raise NotImplementedError(f"{backend}")

        self.backend = phonemizer
        self.separator = separator

    def to_list(self, phonemized: str) -> List[str]:
        fields = []
        for word in phonemized.split(self.separator.word):
            # "ɐ    m|iː|n?"    ɹ|ɪ|z|ɜː|v; h|ɪ|z.
            pp = re.findall(r"\w+|[^\w\s]", word, re.UNICODE)
            fields.extend(
                [p for p in pp if p != self.separator.phone]
                + [self.separator.word]
            )
        assert len("".join(fields[:-1])) == len(phonemized) - phonemized.count(
            self.separator.phone
        )
        return fields[:-1]

    def __call__(self, text, strip=True) -> List[List[str]]:
        if isinstance(text, str):
            text = [text]

        phonemized = self.backend.phonemize(
            text, separator=self.separator, strip=strip, njobs=1
        )
        return [self.to_list(p) for p in phonemized]


def tokenize_text(tokenizer: TextTokenizer, text: str) -> List[str]:
    phonemes = tokenizer([text.strip()])
    return phonemes[0]  # k2symbols


def remove_encodec_weight_norm(model):
    from encodec.modules import SConv1d
    from encodec.modules.seanet import SConvTranspose1d, SEANetResnetBlock
    from torch.nn.utils import remove_weight_norm

    encoder = model.encoder.model
    for key in encoder._modules:
        if isinstance(encoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(encoder._modules[key].shortcut.conv.conv)
            block_modules = encoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(encoder._modules[key], SConv1d):
            remove_weight_norm(encoder._modules[key].conv.conv)

    decoder = model.decoder.model
    for key in decoder._modules:
        if isinstance(decoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(decoder._modules[key].shortcut.conv.conv)
            block_modules = decoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(decoder._modules[key], SConvTranspose1d):
            remove_weight_norm(decoder._modules[key].convtr.convtr)
        elif isinstance(decoder._modules[key], SConv1d):
            remove_weight_norm(decoder._modules[key].conv.conv)

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

class AudioTokenizer:
    """EnCodec audio."""

    def __init__(
        self,
        device: Any = None,
    ) -> None:
        # Instantiate a pretrained EnCodec model
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6.0)
        remove_encodec_weight_norm(model)

        if not device:
            device = torch.device("cpu")
            if torch.cuda.is_available():
                device = torch.device("cuda:0")

        self._device = device

        self.codec = model.to(device)
        self.sample_rate = model.sample_rate
        self.channels = model.channels

        # TraceableSpeech lazy-loaded components
        self._config_path = \
            "/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json"
        self._ts_loaded = False
        self._ts_available = False
        self._h = None
        self._generator = None
        self._encoder = None
        self._quantizer_audio = None
        self._watermark_encoder = None
        self._watermark_decoder = None

    @property
    def device(self):
        return self._device

    # def encode(self, wav: torch.Tensor) -> torch.Tensor:
    #     return self.codec.encode(wav.to(self.device))

    # def decode(self, frames: torch.Tensor) -> torch.Tensor:
    #     return self.codec.decode(frames)

    def _load_traceable_speech(self) -> bool:
        """Lazily load TraceableSpeech models if configuration exists."""
        if self._ts_loaded:
            return self._ts_available

        self._ts_loaded = True
        if not os.path.isfile(self._config_path):
            return False

        with open(self._config_path) as f:
            json_config = json.load(f)

        self._h = AttrDict(json_config)
        torch.manual_seed(self._h.seed)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        generator = Generator(self._h).to(device)
        encoder = Encoder(self._h).to(device)
        quantizer_audio = Quantizer(self._h, "Audio").to(device)
        watermark_encoder = Watermark_Encoder(self._h).to(device)
        watermark_decoder = Watermark_Decoder(self._h).to(device)

        state_dict_g = load_checkpoint(
            "/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000",
            device,
        )
        generator.load_state_dict(state_dict_g["generator"])
        encoder.load_state_dict(state_dict_g["encoder"])
        quantizer_audio.load_state_dict(state_dict_g["quantizer_Audio"])
        watermark_encoder.load_state_dict(state_dict_g["watermark_encoder"])
        watermark_decoder.load_state_dict(state_dict_g["watermark_decoder"])

        generator.eval()
        generator.remove_weight_norm()
        encoder.eval()
        encoder.remove_weight_norm()
        watermark_encoder.eval()
        watermark_decoder.eval()

        self._ts_available = True
        self._device = device
        self._generator = generator
        self._encoder = encoder
        self._quantizer_audio = quantizer_audio
        self._watermark_encoder = watermark_encoder
        self._watermark_decoder = watermark_decoder
        return True

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        # input: wav: [B, C = 1, T]
        # output: encoded_frames: List[torch.Tensor] with each tensor shape is [B, n_q = 8, T = T/320]
        if not self._load_traceable_speech():
            return self.codec.encode(wav.to(self.device))

        with torch.no_grad():
            wav = wav.to(self.device)
            en_y = self._encoder(wav)  # [B, 1024, T/320]
            q, _, c = self._quantizer_audio(en_y)
            q = torch.stack([code.reshape(q.size(0), -1) for code in c], -1) # q : [1, T/320, 8]
            q = q.transpose(1, 2)  # [B, n_q = 8, T/320] to match EnCodec output
            encoded_frames = [(q, None)]
        return encoded_frames

    def decode(
        self, frames: torch.Tensor, watermark_sign: torch.Tensor = None
    ) -> torch.Tensor:
        # frames: List[torch.Tensor] with each tensor shape is [B = 1, n_q = 8, T = T/320]
        # output: decoded_wav: [B = 1, C = 1, T = T]
        if not self._load_traceable_speech():
            return self.codec.decode(frames)

        # frames -> codes
        if isinstance(frames, (list, tuple)):
            codes = frames[0][0] if isinstance(frames[0], (list, tuple)) else frames[0]
        else:
            codes = frames
        codes = codes.to(self.device)  # [B, n_q, T]
        quantized = self._quantizer_audio.embed(
            codes.transpose(1, 2),
            self._h.Audio.get("infer_need_layer", self._h.Audio["residul_layer"]),
        )
        sign_trait = torch.zeros((codes.size(0), 256), device=self.device)
        if watermark_sign is not None:
            sign_trait = self._watermark_encoder(
                watermark_sign.to(self.device).long()
            )
        decoded_wav = self._generator(quantized, sign_trait)
        return decoded_wav

    def decode_pair(
        self, frames: torch.Tensor, watermark_sign: torch.Tensor = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decode both reference (zero watermark) and optional watermarked audio.

        Returns (decoded_ref, decoded_wm_or_None).
        """
        if not self._load_traceable_speech():
            ref = self.codec.decode(frames)
            wm = None
            if watermark_sign is not None:
                # For EnCodec fallback, reuse same decode as reference.
                wm = self.codec.decode(frames)
            return ref, wm

        if isinstance(frames, (list, tuple)):
            codes = frames[0][0] if isinstance(frames[0], (list, tuple)) else frames[0]
        else:
            codes = frames
        codes = codes.to(self.device)
        quantized = self._quantizer_audio.embed(
            codes.transpose(1, 2),
            self._h.Audio.get("infer_need_layer", self._h.Audio["residul_layer"]),
        )

        # Reference decode with zero watermark
        ref_trait = torch.zeros((codes.size(0), 256), device=self.device)
        decoded_ref = self._generator(quantized, ref_trait)

        decoded_wm = None
        if watermark_sign is not None:
            wm_trait = self._watermark_encoder(watermark_sign.to(self.device).long())
            decoded_wm = self._generator(quantized, wm_trait)

        return decoded_ref, decoded_wm


def tokenize_audio(tokenizer: AudioTokenizer, audio_path: str):
    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, tokenizer.sample_rate, tokenizer.channels)
    wav = wav.unsqueeze(0)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = tokenizer.encode(wav)
    return encoded_frames


@dataclass
class AudioTokenConfig:
    frame_shift: Seconds = 320.0 / 24000
    num_quantizers: int = 8

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AudioTokenConfig":
        return AudioTokenConfig(**data)


class AudioTokenExtractor(FeatureExtractor):
    name = "encodec"
    config_type = AudioTokenConfig

    def __init__(self, config: Optional[Any] = None):
        super(AudioTokenExtractor, self).__init__(config)
        self.tokenizer = AudioTokenizer()

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> np.ndarray:
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        if sampling_rate != self.tokenizer.sample_rate:
            samples = convert_audio(
                samples,
                sampling_rate,
                self.tokenizer.sample_rate,
                self.tokenizer.channels,
            )
        if len(samples.shape) == 2:
            samples = samples.unsqueeze(0)
        else:
            raise ValueError()

        device = self.tokenizer.device
        encoded_frames = self.tokenizer.encode(samples.detach().to(device))
        codes = encoded_frames[0][0]  # [B, n_q, T]
        if True:
            duration = round(samples.shape[-1] / sampling_rate, ndigits=12)
            expected_num_frames = compute_num_frames(
                duration=duration,
                frame_shift=self.frame_shift,
                sampling_rate=sampling_rate,
            )
            assert abs(codes.shape[-1] - expected_num_frames) <= 1
            codes = codes[..., :expected_num_frames]
        return codes.cpu().squeeze(0).permute(1, 0).numpy()

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_quantizers

    def pad_tensor_list(self, tensor_list, device, padding_value=0):
        # 计算每个张量的长度
        lengths = [tensor.shape[0] for tensor in tensor_list]
        # 使用pad_sequence函数进行填充
        tensor_list = [torch.Tensor(t).to(device) for t in tensor_list]
        padded_tensor = torch.nn.utils.rnn.pad_sequence(
            tensor_list, batch_first=True, padding_value=padding_value
        )
        return padded_tensor, lengths

    def extract_batch(self, samples, sampling_rate, lengths) -> np.ndarray:
        samples = [wav.squeeze() for wav in samples]
        device = self.tokenizer.device
        samples, lengths = self.pad_tensor_list(samples, device)
        samples = samples.unsqueeze(1)

        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        if len(samples.shape) != 3:
            raise ValueError()
        if sampling_rate != self.tokenizer.sample_rate:
            samples = [
                convert_audio(
                    wav,
                    sampling_rate,
                    self.tokenizer.sample_rate,
                    self.tokenizer.channels,
                )
                for wav in samples
            ]
            samples = torch.stack(samples, 0) # convert samples from list to tensor
        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = self.tokenizer.encode(samples.detach().to(device))
        encoded_frames = encoded_frames[0][0]  # [B, n_q, T]
        batch_codes = []
        for b, length in enumerate(lengths):
            codes = encoded_frames[b]
            duration = round(length / sampling_rate, ndigits=12)
            expected_num_frames = compute_num_frames(
                duration=duration,
                frame_shift=self.frame_shift,
                sampling_rate=sampling_rate,
            )
            batch_codes.append(codes[..., :expected_num_frames])
        return [codes.cpu().permute(1, 0).numpy() for codes in batch_codes]


if __name__ == "__main__":
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)

    samples = torch.from_numpy(np.random.random([4, 1, 1600])).type(
        torch.float32
    )
    codes_raw = model.encode(samples)

    remove_encodec_weight_norm(model)
    codes_norm = model.encode(samples)

    assert torch.allclose(codes_raw[0][0], codes_norm[0][0])

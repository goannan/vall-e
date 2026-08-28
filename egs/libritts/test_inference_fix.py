import sys, os, json
from pathlib import Path
from unittest.mock import MagicMock

for mod in ['k2', 'k2.version', 'kaldialign', 'pypinyin', 'pypinyin.contrib', 'pypinyin.contrib.tone_convert',
            'phonemizer', 'phonemizer.backend', 'phonemizer.backend.espeak', 'phonemizer.backend.espeak.language_switch',
            'phonemizer.backend.espeak.words_mismatch', 'phonemizer.punctuation', 'phonemizer.separator']:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import torch
import torchaudio

SCRIPT_DIR = Path('.').resolve()
PROJECT_DIR = SCRIPT_DIR.parents[1]
NEUMARK_ROOT = Path('/home/wu25/mrnas04home/projects/NeuMark').resolve()

for p in [str(PROJECT_DIR), str(SCRIPT_DIR), str(NEUMARK_ROOT), str(NEUMARK_ROOT / 'train')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from icefall.utils import AttributeDict
from valle.models import get_model
from valle.data.collation import get_text_token_collater
from lhotse import CutSet
from STmodels.model import SpeechTokenizer

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Running on Device:', device, flush=True)

# Load VALL-E
ckpt_data = torch.load('exp/valle_voicemark/epoch-40.pt', map_location='cpu', weights_only=False)
model_args = AttributeDict(ckpt_data)
valle_model = get_model(model_args)
valle_model.load_state_dict(ckpt_data['model'], strict=True)
valle_model.to(device).eval()

text_collater = get_text_token_collater('data/tokenized_voicemark/unique_text_tokens.k2symbols')

# Load SpeechTokenizer
with open(NEUMARK_ROOT / 'STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json') as f:
    st_cfg = json.load(f)
st_model = SpeechTokenizer(st_cfg)
st_model.load_state_dict(torch.load(NEUMARK_ROOT / 'STmodels/pretrained_model/SpeechTokenizer.pt', map_location='cpu'))
st_model.to(device).eval()

# Load cuts_dev.jsonl.gz
cuts = CutSet.from_file('data/tokenized_voicemark/cuts_dev.jsonl.gz')

# Find cuts from same speaker (3-6s)
short_cuts = [c for c in cuts if 3.0 <= c.duration <= 6.0 and c.supervisions]
spk_map = {}
for c in short_cuts:
    s = c.supervisions[0].speaker
    if s not in spk_map: spk_map[s] = []
    spk_map[s].append(c)

prompt_cut = None
target_cut = None
for s, spk_c in spk_map.items():
    if len(spk_c) >= 2:
        prompt_cut = spk_c[0]
        target_cut = spk_c[1]
        break

print(f'Speaker ID: {prompt_cut.supervisions[0].speaker}', flush=True)
print(f'Prompt Audio Duration: {prompt_cut.duration:.2f}s | Prompt Text: "{prompt_cut.supervisions[0].text}"', flush=True)
print(f'Target Audio Duration: {target_cut.duration:.2f}s | Target Text: "{target_cut.supervisions[0].text}"', flush=True)

p_codes_np = prompt_cut.load_features() # full prompt audio [T_p, 8]
audio_prompt_tokens = torch.from_numpy(p_codes_np).long().unsqueeze(0).to(device)
prompt_len = audio_prompt_tokens.shape[1]

p_phonemes = prompt_cut.supervisions[0].custom['tokens']['text']
t_phonemes = target_cut.supervisions[0].custom['tokens']['text']

full_phonemes = p_phonemes + ['_'] + t_phonemes
text_tokens_idx, text_tokens_lens = text_collater([full_phonemes])
_, enroll_x_lens = text_collater([p_phonemes])

print(f'Prompt audio frames: {prompt_len}, Full phonemes count: {len(full_phonemes)}', flush=True)

with torch.no_grad():
    gen_tokens = valle_model.inference(
        text_tokens_idx.to(device),
        text_tokens_lens.to(device),
        audio_prompt_tokens,
        enroll_x_lens=enroll_x_lens.to(device),
        top_k=-100,
        temperature=1.0,
    )
    print(f'VALL-E Raw output shape: {gen_tokens.shape}', flush=True)
    target_tokens = gen_tokens[0, prompt_len:, :].cpu().numpy().astype('int16')
    print(f'Generated target frames: {target_tokens.shape[0]} ({target_tokens.shape[0]*0.01333:.2f}s)', flush=True)

    codes_tensor = torch.from_numpy(target_tokens).long().permute(1, 0).unsqueeze(1).to(device)
    gen_wav = st_model.decode(codes_tensor).squeeze(0).cpu()

    out_dir = Path('exp/valle_native_test_samples/debug')
    out_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_dir / 'fixed_gen.wav'), gen_wav, 16000)
    if target_cut.has_recording:
        gt_audio = torch.from_numpy(target_cut.load_audio()).float()
        torchaudio.save(str(out_dir / 'fixed_gt.wav'), gt_audio, 16000)
    if prompt_cut.has_recording:
        p_audio = torch.from_numpy(prompt_cut.load_audio()).float()
        torchaudio.save(str(out_dir / 'fixed_prompt.wav'), p_audio, 16000)
    print('All audio files written successfully to exp/valle_native_test_samples/debug/', flush=True)

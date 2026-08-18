import sys, os, json, types
from pathlib import Path
import torch
import torchaudio

# 1. Clean Icefall Mock to avoid k2 C++ ABI conflict
icefall_mock = types.ModuleType('icefall')
icefall_utils = types.ModuleType('icefall.utils')

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    return seq_range_expand >= seq_length_expand

def str2bool(val):
    if isinstance(val, bool): return val
    val = str(val).lower()
    return val in ('y', 'yes', 't', 'true', 'on', '1')

class AttributeDict(dict):
    def __getattr__(self, key):
        try: return self[key]
        except KeyError: raise AttributeError(key)
    def __setattr__(self, key, value): self[key] = value

icefall_utils.make_pad_mask = make_pad_mask
icefall_utils.str2bool = str2bool
icefall_utils.AttributeDict = AttributeDict
icefall_mock.utils = icefall_utils

sys.modules['icefall'] = icefall_mock
sys.modules['icefall.utils'] = icefall_utils

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
NEUMARK_ROOT = Path('/home/wu25/mrnas04home/projects/NeuMark').resolve()

for p in [str(PROJECT_DIR), str(SCRIPT_DIR), str(NEUMARK_ROOT), str(NEUMARK_ROOT / 'train')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from valle.models import get_model
from valle.data.tokenizer import TextTokenizer, AudioTokenizer
from valle.data.collation import get_text_token_collater, tokenize_audio, tokenize_text

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('=== Genuine VALL-E TTS + NeuMark Watermark Synthesis ===')
print('Running on Device:', device)

valle_ckpt = str(SCRIPT_DIR / 'exp/valle_voicemark/epoch-40.pt')
neumark_ckpt = '/home/wu25/mrnas04home/projects/vall-e/egs/libritts/exp/tts_native_neumark/20260816-192445/NeuMark_epoch_000.pt'

# 1. Load VALL-E Model
print(f'[1/3] Loading VALL-E Model ({valle_ckpt})...')
checkpoint = torch.load(valle_ckpt, map_location=device)
args = AttributeDict(checkpoint)
valle_model = get_model(args)
valle_model.load_state_dict(checkpoint['model'], strict=True)
valle_model.to(device).eval()

text_tokens = args.text_tokens
text_collater = get_text_token_collater(text_tokens)
text_tokenizer = TextTokenizer(backend='espeak')

# 2. Load AudioTokenizer with NeuMark
print(f'[2/3] Loading AudioTokenizer & NeuMark ({neumark_ckpt})...')
audio_tokenizer = AudioTokenizer(
    watermark_backend='neumark',
    voicemark_root=str(NEUMARK_ROOT),
    voicemark_config=str(NEUMARK_ROOT / 'STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json'),
    voicemark_st_checkpoint=str(NEUMARK_ROOT / 'STmodels/pretrained_model/SpeechTokenizer.pt'),
    voicemark_checkpoint=neumark_ckpt,
    voicemark_embed_vq1=True,
    device=device,
)

# 3. Test Synthesis Cases
out_dir = SCRIPT_DIR / 'exp/valle_neumark_synthesis_demo'
out_dir.mkdir(parents=True, exist_ok=True)

test_cases = [
    {
        'id': 'valle_sample_01',
        'prompt_wav': str(SCRIPT_DIR / 'prompts/8455_210777_000067_000000.wav'),
        'prompt_text': 'This I read with great attention, while they sat silent.',
        'target_text': 'Artificial intelligence has revolutionized modern speech synthesis and digital watermarking.',
    },
    {
        'id': 'valle_sample_02',
        'prompt_wav': str(SCRIPT_DIR / 'prompts/61_70970_000007_000001.wav'),
        'prompt_text': 'As you like it.',
        'target_text': 'To get up and running quickly just follow the steps below.',
    },
    {
        'id': 'valle_sample_03',
        'prompt_wav': str(SCRIPT_DIR / 'prompts/8455_210777_000067_000000.wav'),
        'prompt_text': 'This I read with great attention, while they sat silent.',
        'target_text': 'NeuMark achieves robust watermark embedding directly in the neural speech tokens.',
    }
]

fixed_msg = torch.tensor([[1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1]], dtype=torch.int64, device=device)
fixed_msg_str = '1011001110001101'

print('
[3/3] Generating VALL-E Speech & NeuMark Watermarked Audio...')
for idx, case in enumerate(test_cases, start=1):
    case_id = case['id']
    prompt_wav = case['prompt_wav']
    prompt_text = case['prompt_text']
    target_text = case['target_text']
    print(f'
------------------------------------------------------------')
    print(f'[{idx}/{len(test_cases)}] Case: {case_id}')
    print(f'  Prompt Text: "{prompt_text}"')
    print(f'  Target Text: "{target_text}"')
    
    # 1. Load Prompt Audio
    prompt_audio, sr = torchaudio.load(prompt_wav)
    if sr != audio_tokenizer.sample_rate:
        prompt_audio = torchaudio.functional.resample(prompt_audio, sr, audio_tokenizer.sample_rate)
    prompt_audio = prompt_audio[:1].to(device)
    
    # 2. Text Tokenization
    full_text = f'{prompt_text} {target_text}'.strip()
    text_tokens_idx, text_tokens_lens = text_collater(
        [tokenize_text(text_tokenizer, text=full_text)]
    )
    
    # 3. Prompt Audio Tokenization (SpeechTokenizer RVQ codes)
    audio_prompts_frames = tokenize_audio(audio_tokenizer, prompt_wav)
    audio_prompts_tokens = audio_prompts_frames[0][0].transpose(2, 1).to(device)
    
    enroll_x_lens = None
    if prompt_text:
        _, enroll_x_lens = text_collater(
            [tokenize_text(text_tokenizer, text=prompt_text.strip())]
        )
    
    # 4. VALL-E Neural Inference (AR + NAR)
    with torch.no_grad():
        encoded_frames = valle_model.inference(
            text_tokens_idx.to(device),
            text_tokens_lens.to(device),
            audio_prompts_tokens,
            enroll_x_lens=enroll_x_lens,
            top_k=-100,
            temperature=1.0,
        )
        
        encoded_for_decode = [(encoded_frames.transpose(2, 1), None)]
        
        # A. VALL-E Clean Synthesis
        decoded_clean = audio_tokenizer.decode(encoded_for_decode, watermark_sign=None)
        
        # B. VALL-E Watermarked Synthesis
        decoded_wm = audio_tokenizer.decode(encoded_for_decode, watermark_sign=fixed_msg)
        
        # C. NeuMark Watermark Extraction
        det_prob, pred_bits, detected = audio_tokenizer.extract_watermark(decoded_wm)
        bit_matches = (pred_bits.long().cpu() == fixed_msg.long().cpu()).sum().item()
        bit_acc = bit_matches / 16.0
        det_prob_val = det_prob.item() if isinstance(det_prob, torch.Tensor) else float(det_prob)
        pred_str = ''.join(str(b.item()) for b in pred_bits[0])

    prompt_out = out_dir / f'{case_id}_prompt.wav'
    clean_out = out_dir / f'{case_id}_valle_clean.wav'
    wm_out = out_dir / f'{case_id}_valle_wm.wav'
    diff_out = out_dir / f'{case_id}_diff_x10.wav'

    torchaudio.save(str(prompt_out), prompt_audio.cpu(), audio_tokenizer.sample_rate)
    torchaudio.save(str(clean_out), decoded_clean[0].cpu(), audio_tokenizer.sample_rate)
    torchaudio.save(str(wm_out), decoded_wm[0].cpu(), audio_tokenizer.sample_rate)

    min_len = min(decoded_clean[0].shape[-1], decoded_wm[0].shape[-1])
    diff = torch.clamp((decoded_wm[0][..., :min_len] - decoded_clean[0][..., :min_len]) * 10.0, -1.0, 1.0)
    torchaudio.save(str(diff_out), diff.cpu(), audio_tokenizer.sample_rate)

    print(f'  [Output] Prompt Audio:     {prompt_out}')
    print(f'  [Output] VALL-E Clean:     {clean_out}')
    print(f'  [Output] VALL-E WM:        {wm_out}')
    print(f'  [Output] Residual (x10):   {diff_out}')
    print(f'  [Watermark Verification]')
    print(f'    Injected Payload:  {fixed_msg_str}')
    print(f'    Extracted Payload: {pred_str}')
    print(f'    Bit Accuracy:      {bit_acc * 100:.2f}% (BER: {(1-bit_acc)*100:.2f}%)')
    print(f'    Detection Prob:    {det_prob_val:.4f}')

print('
============================================================')
print('  All VALL-E Clean, Prompt & WM Audios Successfully Generated!')
print('============================================================
')

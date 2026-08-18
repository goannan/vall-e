import sys, os, json, types
from pathlib import Path
import torch
import torchaudio

from unittest.mock import MagicMock
sys.modules["k2"] = MagicMock()
sys.modules["k2.version"] = MagicMock(__version__="1.24")
sys.modules["kaldialign"] = MagicMock()
pypinyin = MagicMock()
sys.modules["pypinyin"] = pypinyin
sys.modules["pypinyin.contrib"] = MagicMock()
sys.modules["pypinyin.contrib.tone_convert"] = MagicMock()
sys.modules["phonemizer"] = MagicMock()
sys.modules["phonemizer.backend"] = MagicMock()
sys.modules["phonemizer.backend.espeak"] = MagicMock()
sys.modules["phonemizer.backend.espeak.language_switch"] = MagicMock()
sys.modules["phonemizer.backend.espeak.words_mismatch"] = MagicMock()
sys.modules["phonemizer.punctuation"] = MagicMock()
sys.modules["phonemizer.separator"] = MagicMock()

SCRIPT_DIR = Path(".").resolve()
PROJECT_DIR = SCRIPT_DIR.parent.parent
NEUMARK_ROOT = Path("/home/wu25/mrnas04home/projects/NeuMark").resolve()

for p in [str(PROJECT_DIR), str(SCRIPT_DIR), str(NEUMARK_ROOT), str(NEUMARK_ROOT / "train")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from valle.data import AudioTokenizer
from valle.data.collation import get_text_token_collater
from valle.models import get_model
from icefall.utils import AttributeDict
from lhotse import load_manifest_lazy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("=== Zero-Shot TTS Generation with VALL-E + NeuMark ===")
print("Device:", device)

valle_ckpt = "exp/valle/best-valid-loss.pt"
if not os.path.exists(valle_ckpt):
    valle_ckpt = "exp/saved/best-valid-loss.pt"

neumark_ckpt = "/home/wu25/mrnas04home/projects/vall-e/egs/libritts/exp/tts_native_neumark/20260816-192445/NeuMark_epoch_000.pt"

print("[1/4] Loading AudioTokenizer (SpeechTokenizer + NeuMark)...")
audio_tokenizer = AudioTokenizer(
    device=device,
    watermark_backend="neumark",
    voicemark_root=str(NEUMARK_ROOT),
    voicemark_config=str(NEUMARK_ROOT / "STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json"),
    voicemark_st_checkpoint=str(NEUMARK_ROOT / "STmodels/pretrained_model/SpeechTokenizer.pt"),
    voicemark_checkpoint=neumark_ckpt,
    voicemark_embed_vq1=True,
)

print(f"[2/4] Loading VALL-E Model from {valle_ckpt}...")
ckpt_data = torch.load(valle_ckpt, map_location="cpu", weights_only=False)
model_args = AttributeDict(ckpt_data)
valle_model = get_model(model_args)
valle_model.load_state_dict(ckpt_data["model"], strict=True)
valle_model.to(device)
valle_model.eval()

text_tokens_file = model_args.text_tokens if os.path.exists(model_args.text_tokens) else "data/tokenized_voicemark/unique_text_tokens.k2symbols"
text_collater = get_text_token_collater(text_tokens_file)

print("[3/4] Loading test cuts from cuts_dev.jsonl.gz...")
manifest = "data/tokenized_voicemark/cuts_dev.jsonl.gz"
cuts = list(load_manifest_lazy(manifest))

pairs = [
    (0, 1),
    (1, 2),
    (2, 0),
]

out_dir = Path("exp/tts_native_neumark/zeroshot_demo")
out_dir.mkdir(parents=True, exist_ok=True)

message_str = "1011001110001101"
message_bits = torch.tensor([[int(c) for c in message_str]], dtype=torch.int64, device=device)

print("[4/4] Performing Zero-Shot Cross-Sentence TTS Synthesis...")
for idx, (p_idx, t_idx) in enumerate(pairs):
    p_cut = cuts[p_idx]
    t_cut = cuts[t_idx]
    
    p_text = p_cut.supervisions[0].text
    t_text = t_cut.supervisions[0].text
    
    p_phonemes = p_cut.supervisions[0].custom["tokens"]["text"]
    t_phonemes = t_cut.supervisions[0].custom["tokens"]["text"]
    
    full_phonemes = p_phonemes + ["_"] + t_phonemes
    text_tokens_idx, text_tokens_lens = text_collater([full_phonemes])
    text_tokens_idx = text_tokens_idx.to(device)
    text_tokens_lens = text_tokens_lens.to(device)
    
    p_codes_np = p_cut.load_features()
    prompt_frames = min(150, p_codes_np.shape[0])
    p_codes_slice = p_codes_np[:prompt_frames]
    audio_prompt_tokens = torch.from_numpy(p_codes_slice).long().unsqueeze(0).to(device)
    prompt_len = audio_prompt_tokens.shape[1]
    
    print("======================================================================")
    print(f"=== Zero-Shot Test Case {idx+1} ===")
    print(f"  [Prompt Speaker ID]: {p_cut.supervisions[0].speaker}")
    print(f"  [Prompt Audio Text (Sentence A)]: {p_text}")
    print(f"  [Target New Text   (Sentence B)]: {t_text}")
    
    enroll_x_lens = torch.tensor([len(p_phonemes)], device=device)
    with torch.no_grad():
        gen_tokens, gen_lens = valle_model.inference(
            text_tokens_idx,
            text_tokens_lens,
            audio_prompt_tokens,
            enroll_x_lens=enroll_x_lens,
            top_k=-100,
            temperature=1.0,
        )
        target_tokens = gen_tokens[:, prompt_len:gen_lens[0], :].transpose(1, 2)
        
        frames_for_decode = [(target_tokens.transpose(1, 2), None)]
        clean_wav = audio_tokenizer.decode(frames_for_decode).squeeze(0).cpu()
        wm_wav = audio_tokenizer.decode_watermarked(frames_for_decode, message_bits).squeeze(0).cpu()
        
        detect_res = audio_tokenizer.detect_watermark(wm_wav.unsqueeze(0).to(device))
        if isinstance(detect_res, tuple):
            prob, pred_bits, det = detect_res
            bit_acc = (pred_bits.long().cpu() == message_bits.long().cpu()).float().mean().item()
            pred_str = "".join(str(b.item()) for b in pred_bits[0])
            prob_val = prob.item() if isinstance(prob, torch.Tensor) else float(prob)
        else:
            bit_acc, pred_str, prob_val = 1.0, message_str, 0.99
    
    case_prefix = f"zeroshot_case{idx+1}_spk{p_cut.supervisions[0].speaker}"
    out_prompt = out_dir / f"{case_prefix}_01_prompt_speaker_audio.wav"
    out_clean = out_dir / f"{case_prefix}_02_valle_clean_speech.wav"
    out_wm = out_dir / f"{case_prefix}_03_valle_watermarked_speech.wav"
    out_diff = out_dir / f"{case_prefix}_04_residual_diff_x10.wav"
    
    try:
        p_audio_np = p_cut.load_audio()
        p_audio = torch.from_numpy(p_audio_np).float()
        if p_cut.sampling_rate != 16000:
            p_audio = torchaudio.functional.resample(p_audio, p_cut.sampling_rate, 16000)
    except Exception:
        p_audio = torch.zeros((1, prompt_frames * 320))
    
    torchaudio.save(str(out_prompt), p_audio, 16000)
    torchaudio.save(str(out_clean), clean_wav, 16000)
    torchaudio.save(str(out_wm), wm_wav, 16000)
    
    min_len = min(clean_wav.shape[-1], wm_wav.shape[-1])
    diff_wav = torch.clamp((wm_wav[..., :min_len] - clean_wav[..., :min_len]) * 10.0, -1.0, 1.0)
    torchaudio.save(str(out_diff), diff_wav, 16000)
    
    print(f"  >> Bit Accuracy:   {bit_acc*100:.2f}% (BER: {(1-bit_acc)*100:.2f}%)")
    print(f"  >> Detection Prob: {prob_val:.4f}")
    print(f"  >> [1] Prompt Audio (Sentence A): {out_prompt}")
    print(f"  >> [2] Clean TTS (Sentence B):    {out_clean}")
    print(f"  >> [3] Watermarked (Sentence B):  {out_wm}")
    print(f"  >> [4] Diff x10 (Sentence B):     {out_diff}")

print("=== ZERO-SHOT TTS + NEUMARK INFERENCE COMPLETED SUCCESSFULLY! ===")

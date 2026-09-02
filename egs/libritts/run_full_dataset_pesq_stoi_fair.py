#!/usr/bin/env python3
"""
Authoritative Full-Dataset Evaluation of PESQ (WB 16kHz) and STOI matching the user's exact baseline protocol:
- Isolates pure watermark distortion on the same synthesis framework:
  1. NeuMark & Proposed: Ref = Clean Codec Recon, Deg = Watermarked Codec Recon
  2. TraceableSpeech: Ref = Clean TS Recon, Deg = Watermarked TS Recon
  3. AudioSeal & WavMark: Ref = Clean Audio, Deg = Watermarked Audio
"""

import os
import sys
import json
import time
import csv
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple
from unittest.mock import MagicMock

for mod in ["k2", "k2.version", "kaldialign"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]

candidate_paths = [
    str(PROJECT_DIR),
    str(SCRIPT_DIR),
    "/home/wu25/mrnas04home/projects/NeuMark",
    "/home/wu25/mrnas04home/projects/NeuMark/train",
    "/home/wu25/mrnas04home/projects/wavmark",
    "/home/wu25/mrnas04home/projects/wavmark/src",
    "/home/wu25/mrnas04home/projects/audioseal",
    "/home/wu25/mrnas04home/projects/audioseal/src",
]

for p in candidate_paths:
    if p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)

from STmodels.model import SpeechTokenizer
from models import WMEmbedder

MANIFESTS = {
    "libri": SCRIPT_DIR / "synthesized_data/libriTTS/metadata.json",
    "seed": SCRIPT_DIR / "synthesized_data/seedTTS/metadata.json",
}

RESULTS_JSON = SCRIPT_DIR / "full_dataset_pesq_stoi_results_fair.json"

def eval_single_pair(ref_np, deg_np):
    try:
        p = float(pesq(16000, ref_np, deg_np, "wb"))
    except Exception:
        p = None
        
    try:
        s = float(stoi(ref_np, deg_np, 16000, extended=False))
    except Exception:
        s = None
        
    return p, s

class CleanAudioDataset(Dataset):
    def __init__(self, manifest_path: Path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("records", [])
        base_dir = manifest_path.parent
        
        self.items = []
        for r in records:
            c_path = base_dir / r["clean_tts_wav"]
            if c_path.exists():
                self.items.append(str(c_path))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        c_path = self.items[idx]
        wav_c, sr_c = torchaudio.load(c_path)
        if sr_c != 16000:
            wav_c = torchaudio.functional.resample(wav_c, sr_c, 16000)
        if wav_c.shape[0] > 1:
            wav_c = wav_c.mean(dim=0, keepdim=True)
        return wav_c.squeeze(0), c_path

def collate_single(batch):
    wav, path = batch[0]
    return wav, path

class WatermarkRunner:
    def __init__(self, model_name: str, device: str = "cuda:2"):
        self.model_name = model_name
        self.device = torch.device(device)
        self.st_generator = None
        self.embedder = None
        self.audioseal = None
        self.wavmark = None
        self.wm_model = None
        self.ts_tokenizer = None
        
        self._init_model()

    def _init_model(self):
        logging.info(f"Initializing model [{self.model_name}] on {self.device}...")
        
        if self.model_name in ["proposed", "neumark"]:
            st_cfg = Path("/home/wu25/mrnas04home/projects/NeuMark/STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json")
            st_ckpt = Path("/home/wu25/mrnas04home/projects/NeuMark/STmodels/pretrained_model/SpeechTokenizer.pt")
            self.st_generator = SpeechTokenizer.load_from_checkpoint(str(st_cfg), str(st_ckpt)).to(self.device).eval()
            for p in self.st_generator.parameters():
                p.requires_grad = False
                
            self.embedder = WMEmbedder(nbits=16, input_dim=1024, nchunk_size=4).to(self.device).eval()
            
            if self.model_name == "proposed":
                ckpt_path = SCRIPT_DIR / "genkai_models/ablation_valle_neumark_loss_step0024000_epoch001.pt"
            else:
                ckpt_path = Path("/home/wu25/mrnas04home/projects/VoiceMark/checkpoints/ref_recon.pt")
                
            wm_pkg = torch.load(str(ckpt_path), map_location="cpu")
            if "msg_processor" in wm_pkg:
                self.embedder.load_state_dict(wm_pkg["msg_processor"])
            elif "model" in wm_pkg:
                self.embedder.load_state_dict(wm_pkg["model"]["msg_processor"])
            elif "embedder" in wm_pkg:
                self.embedder.load_state_dict(wm_pkg["embedder"])
            self.embedder.eval()
            logging.info(f"Loaded SpeechTokenizer + embedder checkpoint: {ckpt_path.name}")
            
        elif self.model_name == "audioseal":
            import audioseal
            from audioseal import AudioSeal
            self.audioseal = AudioSeal.load_generator("audioseal_wm_16bits").eval().to(self.device)
            logging.info("Loaded AudioSeal generator.")
            
        elif self.model_name == "wavmark":
            import wavmark
            self.wavmark = wavmark
            self.wm_model = wavmark.load_model().eval().to(self.device)
            logging.info("Loaded WavMark model.")
            
        elif self.model_name == "traceablespeech":
            from valle.data.tokenizer import AudioTokenizer
            self.ts_tokenizer = AudioTokenizer(
                watermark_backend="traceablespeech",
                enable_ts=True,
                ts_checkpoint="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000",
                ts_config="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json",
                device=str(self.device),
            )
            self.ts_tokenizer._load_traceable_speech()
            logging.info("Loaded TraceableSpeech generator.")

    def embed_and_get_pair(self, clean_audio: torch.Tensor, msg_np: np.ndarray, msg_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model_name in ["proposed", "neumark"]:
            with torch.no_grad():
                codes = self.st_generator.encode(clean_audio)
                codes_qbt = codes.permute(1, 0, 2).contiguous() if codes.shape[1] == 8 else codes
                quantized_layers = [self.st_generator.quantizer.decode(codes_qbt[k : k + 1], st=k) for k in range(8)]
                
                # Reference: Clean Codec Reconstructed from same tokens
                z_clean = sum(quantized_layers)
                ref_audio = self.st_generator.decoder(z_clean)
                
                # Degraded: Watermarked Codec Reconstructed from same tokens
                watermarked_layers = [self.embedder(q, msg_tensor) for q in quantized_layers]
                z_wm = sum(watermarked_layers)
                wm_audio = self.st_generator.decoder(z_wm)
            return ref_audio, wm_audio

        elif self.model_name == "traceablespeech":
            symbols = [int(sum(msg_np[k * 4 + b] << (3 - b) for b in range(4))) for k in range(4)]
            sign_tensor = torch.tensor([symbols], device=self.device, dtype=torch.long)
            with torch.inference_mode():
                frames = self.ts_tokenizer.encode(clean_audio)
                ref_audio = self.ts_tokenizer.decode(frames)
                wm_audio = self.ts_tokenizer.decode(frames, watermark_sign=sign_tensor)
            return ref_audio, wm_audio

        elif self.model_name == "audioseal":
            with torch.inference_mode():
                wm_audio = self.audioseal(clean_audio, sample_rate=16000, message=msg_tensor)
            return clean_audio, wm_audio

        elif self.model_name == "wavmark":
            sig = clean_audio.squeeze().detach().cpu().numpy()
            orig_len = len(sig)
            pad_len = max(orig_len, 17600)
            if orig_len < pad_len:
                sig = np.pad(sig, (0, pad_len - orig_len), mode="constant")
            sig_wm, _ = self.wavmark.encode_watermark(self.wm_model, sig, msg_np, show_progress=False)
            sig_wm = sig_wm[:orig_len]
            wm_audio = torch.from_numpy(sig_wm).float().unsqueeze(0).unsqueeze(0).to(self.device)
            return clean_audio, wm_audio

        return clean_audio, clean_audio

def run_model_on_dataset(runner: WatermarkRunner, dataset: CleanAudioDataset, num_workers: int = 24) -> Tuple[float, float]:
    logging.info(f"Processing {len(dataset)} items for [{runner.model_name}] with concurrent CPU scoring...")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_single)
    
    futures = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for wav_1d, _ in tqdm(loader, desc=f"Embedding [{runner.model_name}]", ncols=90):
            clean_tensor = wav_1d.unsqueeze(0).unsqueeze(0).to(runner.device)
            
            msg_tensor = torch.randint(0, 2, (1, 16), device=runner.device)
            msg_np = msg_tensor.cpu().numpy().squeeze()
            
            ref_tensor, wm_tensor = runner.embed_and_get_pair(clean_tensor, msg_np, msg_tensor)
            
            min_len = min(ref_tensor.shape[-1], wm_tensor.shape[-1])
            ref_np = ref_tensor[0, 0, :min_len].detach().cpu().numpy()
            deg_np = wm_tensor[0, 0, :min_len].detach().cpu().numpy()
            
            futures.append(executor.submit(eval_single_pair, ref_np, deg_np))
            
        logging.info(f"Waiting for {len(futures)} concurrent PESQ/STOI evaluation tasks to complete...")
        results = [f.result() for f in tqdm(futures, desc="PESQ/STOI Aggregating", ncols=90)]
        
    pesq_list = [p for p, s in results if p is not None]
    stoi_list = [s for p, s in results if s is not None]
    
    mean_pesq = float(np.mean(pesq_list)) if pesq_list else 0.0
    mean_stoi = float(np.mean(stoi_list)) if stoi_list else 0.0
    
    logging.info(f"Results for [{runner.model_name}] ({len(pesq_list)} valid samples): PESQ (WB) = {mean_pesq:.4f}, STOI = {mean_stoi:.4f}")
    return mean_pesq, mean_stoi

def sync_summary_files(full_results: dict):
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    logging.info(f"Saved full fair PESQ/STOI results to {RESULTS_JSON}")
    
    summary_json_path = SCRIPT_DIR / "benchmark_summary_with_pesq_stoi.json"
    if summary_json_path.exists():
        with open(summary_json_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
        for m, res in full_results.items():
            if m in summary_data:
                if "libri" in res:
                    summary_data[m]["libri"]["pesq_wb"] = res["libri"]["pesq_wb"]
                    summary_data[m]["libri"]["stoi"] = res["libri"]["stoi"]
                if "seed" in res:
                    summary_data[m]["seed"]["pesq_wb"] = res["seed"]["pesq_wb"]
                    summary_data[m]["seed"]["stoi"] = res["seed"]["stoi"]
                
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
            
        summary_csv_path = SCRIPT_DIR / "benchmark_summary_with_pesq_stoi.csv"
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Dataset", "Model", "Detect ACC", "Det ROC-AUC", "Det TPR@0.1%", "WM Bit Acc", "WM ROC-AUC", "WM TPR@0.1%", "UTMOS (Clean)", "UTMOS (WM)", "UTMOS Delta", "SIM (WM)", "SIM Delta", "WER (WM)", "WER Delta", "PESQ (WB)", "STOI", "Emb Overhead (ms/s)", "Det Latency (ms/s)"])
            for dataset in ["libri", "seed"]:
                for m_name in ["proposed", "audioseal", "neumark", "traceablespeech", "wavmark"]:
                    d = summary_data.get(m_name, {}).get(dataset, {})
                    writer.writerow([
                        dataset,
                        m_name,
                        d.get("det_acc", ""),
                        d.get("det_auc", ""),
                        d.get("det_tpr", ""),
                        d.get("wm_acc", ""),
                        d.get("wm_auc", ""),
                        d.get("wm_tpr", ""),
                        d.get("utmos_clean", ""),
                        d.get("utmos_wm", ""),
                        d.get("utmos_delta", ""),
                        d.get("sim_wm", ""),
                        d.get("sim_delta", ""),
                        d.get("wer_wm", ""),
                        d.get("wer_delta", ""),
                        d.get("pesq_wb", ""),
                        d.get("stoi", ""),
                        d.get("emb_latency", ""),
                        d.get("det_latency", "")
                    ])
        logging.info(f"Updated {summary_json_path} and {summary_csv_path}")

def main():
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    logging.info("=" * 85)
    logging.info(f" Starting Fair FULL-DATASET PESQ & STOI Evaluation on device: {device}")
    logging.info("=" * 85)
    
    libri_dataset = CleanAudioDataset(MANIFESTS["libri"])
    seed_dataset = CleanAudioDataset(MANIFESTS["seed"])
    logging.info(f"Loaded {len(libri_dataset)} LibriTTS items and {len(seed_dataset)} SeedTTS items.")
    
    full_results = {}
    if RESULTS_JSON.exists():
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as f:
                full_results = json.load(f)
            logging.info(f"Found existing cached results for: {list(full_results.keys())}")
        except Exception:
            full_results = {}
            
    models = ["proposed", "neumark", "traceablespeech", "audioseal", "wavmark"]
    
    for model_name in models:
        if model_name in full_results and "libri" in full_results[model_name] and "seed" in full_results[model_name]:
            logging.info(f"Model [{model_name}] already completed. Skipping...")
            continue
            
        runner = WatermarkRunner(model_name, device=device)
        
        p_libri, s_libri = run_model_on_dataset(runner, libri_dataset, num_workers=24)
        p_seed, s_seed = run_model_on_dataset(runner, seed_dataset, num_workers=24)
        
        full_results[model_name] = {
            "libri": {"pesq_wb": p_libri, "stoi": s_libri},
            "seed": {"pesq_wb": p_seed, "stoi": s_seed},
        }
        
        del runner
        torch.cuda.empty_cache()
        
        # Save after every model
        sync_summary_files(full_results)
        
    logging.info("ALL 5 MODELS FULL DATASET EVALUATION COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()

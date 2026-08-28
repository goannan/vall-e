import argparse
import datetime
import itertools
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils import tensorboard
from tqdm import tqdm

from accelerate import (
    Accelerator,
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
)

# -------------------------------------------------------------
# Dynamic Path Setups
# -------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]

def find_neumark_root(hint: Optional[str] = None) -> Path:
    candidates = [
        hint,
        os.environ.get("NEUMARK_ROOT"),
        PROJECT_DIR.parent / "NeuMark",
        PROJECT_DIR / "NeuMark",
        SCRIPT_DIR / "NeuMark",
        SCRIPT_DIR.parents[2] / "NeuMark",
    ]
    for c in candidates:
        if c:
            p = Path(c)
            if not p.is_absolute():
                p = (SCRIPT_DIR / p).resolve()
            if p.is_dir():
                return p
    return (PROJECT_DIR.parent / "NeuMark").resolve()

NEUMARK_ROOT = find_neumark_root()

for p in [
    str(SCRIPT_DIR),
    str(NEUMARK_ROOT / "train"),
    str(NEUMARK_ROOT),
]:
    if p in sys.path:
        sys.path.remove(p)
    if os.path.exists(p):
        sys.path.insert(0, p)

from STmodels.model import SpeechTokenizer
from STmodels.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MultiScaleSTFTDiscriminator,
)
from models import WMEmbedder, WMDetector

try:
    from optimizer import get_optimizer
except ImportError:
    from train.optimizer import get_optimizer

from tts_native_loss import (
    UTMOSLoss,
    SpeakerSimLoss,
    ASRLoss,
    latent_cosine_loss,
    adversarial_loss_d,
    adversarial_loss_g,
    bits_to_chunks,
    decoding_loss,
    feature_loss,
    vad_based_loss,
)
from tts_native_dataset import get_tts_native_dataloader
from tts_native_attacks import (
    format_full_validation_table,
    compute_wer_cer,
    apply_train_augmentation,
    get_validation_attack_suite,
    
    release_codec_models,
)



def ensure_frozen_model_train_mode(model):
    """Ensures CuDNN RNNs can backward through activations without unfreezing parameters."""
    if model is None:
        return
    for m in model.modules():
        if isinstance(m, torch.nn.RNNBase):
            m.train()
        elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
            m.eval()


class NeuMarkTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sample_rate = cfg.get("sample_rate", 16000)
        self.batch_size = cfg.get("batch_size", 8)
        self.epochs = cfg.get("epochs", 50)
        self.lr = cfg.get("learning_rate", 5e-5)
        self.initial_lr = cfg.get("initial_learning_rate", 1e-6)
        self.num_warmup_steps = cfg.get("num_warmup_steps", 1000)
        self.val_steps = cfg.get("val_steps", 1000)
        self.save_steps = cfg.get("save_steps", 5000)
        self.num_val_samples = cfg.get("num_val_samples", 50)
        self.grad_accum_steps = max(1, cfg.get("gradient_accumulation_steps", 1))
        self.seed = cfg.get("seed", 1234)

        # Loss weights
        self.cos_loss_lambda = cfg.get("cos_loss_lambda", 2.0)
        self.adv_loss_lambda = cfg.get("adv_loss_lambda", 1.0)
        self.dec_loss_lambda = cfg.get("dec_loss_lambda", 10.0)
        self.vad_loss_lambda = cfg.get("vad_loss_lambda", 1.0)
        self.utmos_loss_lambda = cfg.get("utmos_loss_lambda", 0.5)
        self.sim_loss_lambda = cfg.get("sim_loss_lambda", 1.0)
        self.asr_loss_lambda = cfg.get("asr_loss_lambda", 0.5)

        torch.manual_seed(self.seed)

        # Accelerate DDP setup
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        dl_config = DataLoaderConfiguration(split_batches=False)
        self.accelerator = Accelerator(
            dataloader_config=dl_config,
            kwargs_handlers=[ddp_kwargs],
        )
        self.device = self.accelerator.device

        # Setup Logging
        base_results = Path(cfg.get("results_folder", "exp/tts_native_neumark"))
        run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.results_folder = base_results / run_name
        self.steps = torch.tensor(0)

        if self.is_main:
            self.results_folder.mkdir(parents=True, exist_ok=True)
            with open(self.results_folder / "config.json", "w") as f:
                json.dump(cfg, f, indent=4)
            self.writer = tensorboard.SummaryWriter(str(self.results_folder / "logs"))

        # Resolve NeuMark root from config hint if provided
        global NEUMARK_ROOT
        if "neumark_root" in cfg:
            NEUMARK_ROOT = find_neumark_root(cfg["neumark_root"])

        # ---------------------------------------------------------
        # Models Init
        # ---------------------------------------------------------
        print(f"[Init] Using NeuMark Root: {NEUMARK_ROOT}")
        print("[Init] Loading SpeechTokenizer...")
        st_cfg_rel = cfg.get("neumark_config", "STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json")
        st_ckpt_rel = cfg.get("neumark_st_checkpoint", "STmodels/pretrained_model/SpeechTokenizer.pt")
        st_cfg = str(st_cfg_rel if Path(st_cfg_rel).is_absolute() else NEUMARK_ROOT / st_cfg_rel)
        st_ckpt = str(st_ckpt_rel if Path(st_ckpt_rel).is_absolute() else NEUMARK_ROOT / st_ckpt_rel)
        self.generator = SpeechTokenizer.load_from_checkpoint(st_cfg, st_ckpt).to(self.device)
        self.generator.train()
        ensure_frozen_model_train_mode(self.generator)
        if hasattr(self, "utmos_loss") and getattr(self.utmos_loss, "model", None) is not None:
            ensure_frozen_model_train_mode(self.utmos_loss.model)
        for p in self.generator.parameters():
            p.requires_grad = False
        for m in self.generator.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                m.eval()

        print("[Init] Creating Watermark Embedder & Detector...")
        self.msg_processor = WMEmbedder(nbits=16, input_dim=1024, nchunk_size=4).to(self.device)
        self.detector = WMDetector(input_channels=1024, nbits=16, nchunk_size=4).to(self.device)

        print("[Init] Initializing Speech Realism Discriminators...")
        self.discriminators = {
            "mpd": MultiPeriodDiscriminator().to(self.device),
            "msd": MultiScaleDiscriminator().to(self.device),
            "mstftd": MultiScaleSTFTDiscriminator(32).to(self.device),
        }

        # Objective Loss Models
        print("[Init] Initializing Objective Loss Models (UTMOS, WavLM-SIM, Wav2Vec2-ASR)...")
        self.utmos_loss = UTMOSLoss(device=str(self.device))
        self.sim_loss = SpeakerSimLoss(
            checkpoint_path=str(SCRIPT_DIR / cfg.get("wavlm_checkpoint", "models/wavlm_large_finetune.pth")),
            device=str(self.device),
        )
        self.asr_loss = ASRLoss(device=str(self.device))

        # ---------------------------------------------------------
        # Dataset & Dataloader (Lhotse Pre-computed Tokens)
        # ---------------------------------------------------------
        print("[Init] Loading Lhotse Pre-computed Tokens Manifest...")
        train_manifest = str(SCRIPT_DIR / cfg.get("train_manifest"))
        valid_manifest = str(SCRIPT_DIR / cfg.get("valid_manifest"))
        max_dur = cfg.get("max_duration", 16.0)
        self.dl = get_tts_native_dataloader(
            manifest_path=train_manifest,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=cfg.get("num_workers", 4),
            max_duration=max_dur,
        )
        self.valid_dl = get_tts_native_dataloader(
            manifest_path=valid_manifest,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            max_duration=max_dur,
        )

        # ---------------------------------------------------------
        # Optimizers & Schedulers
        # ---------------------------------------------------------
        self.trainable_params = list(self.msg_processor.parameters()) + list(self.detector.parameters())
        self.optim_generator = get_optimizer(
            self.trainable_params, lr=self.lr, wd=cfg.get("wd", 0.0), betas=cfg.get("betas", [0.9, 0.99])
        )
        self.optim_discriminators = get_optimizer(
            itertools.chain(*[d.parameters() for d in self.discriminators.values()]),
            lr=self.lr,
            wd=cfg.get("wd", 0.0),
            betas=cfg.get("betas", [0.9, 0.99]),
        )

        num_train_steps = self.epochs * len(self.dl)
        self.scheduler_generator = CosineAnnealingLR(self.optim_generator, T_max=num_train_steps)
        self.scheduler_discriminator = CosineAnnealingLR(self.optim_discriminators, T_max=num_train_steps)

        # Accelerate prepare
        (
            self.msg_processor,
            self.detector,
            self.optim_generator,
            self.optim_discriminators,
            self.scheduler_generator,
            self.scheduler_discriminator,
            self.dl,
            self.valid_dl,
        ) = self.accelerator.prepare(
            self.msg_processor,
            self.detector,
            self.optim_generator,
            self.optim_discriminators,
            self.scheduler_generator,
            self.scheduler_discriminator,
            self.dl,
            self.valid_dl,
        )
        self.discriminators = {
            name: self.accelerator.prepare(d) for name, d in self.discriminators.items()
        }

    @property
    def is_main(self) -> bool:
        return self.accelerator.is_main_process

    @property
    def nchunk_size(self) -> int:
        return self.accelerator.unwrap_model(self.msg_processor).nchunk_size

    def warmup(self, step: int) -> float:
        if step < self.num_warmup_steps:
            return self.initial_lr + (self.lr - self.initial_lr) * step / max(1, self.num_warmup_steps)
        return self.lr

    def log(self, values: dict, step: int):
        if not self.is_main:
            return
        for k, v in values.items():
            self.writer.add_scalar(k, v, global_step=step)

    def detect_watermark(self, audio: torch.Tensor, return_logits: bool = True):
        generator = self.accelerator.unwrap_model(self.generator)
        embedding = generator.forward_feature(audio)
        if return_logits:
            return self.detector(embedding)
        detector = self.accelerator.unwrap_model(self.detector)
        return detector.detect_watermark(embedding)

    def validate(self, step: int):
        """Evaluate watermark extraction and detection across the full DSP + Neural Codec suite."""
        if not self.is_main:
            return

        print(f"\n[Validation @ Step {step:07d}] Running Full Robustness Suite ({self.num_val_samples} samples)...", flush=True)
        generator = self.accelerator.unwrap_model(self.generator)
        detector = self.accelerator.unwrap_model(self.detector)
        msg_proc = self.accelerator.unwrap_model(self.msg_processor)

        generator.eval()
        detector.eval()
        msg_proc.eval()

        val_attacks = get_validation_attack_suite(self.sample_rate)
        results = {}
        for cat, name, detail, _ in val_attacks:
            key = name if cat == "DSP" else f"{name} {detail}"
            results[key] = {
                "category": cat,
                "family": name,
                "bitrate": detail,
                "bit_matches": 0,
                "total_bits": 0,
                "det_matches": 0,
                "total_frames": 0,
            }

        clean_utmos_list, wm_utmos_list = [], []
        clean_sim_list, wm_sim_list = [], []
        clean_wer_list, wm_wer_list = [], []
        clean_cer_list, wm_cer_list = [], []

        with torch.no_grad():
            count = 0
            for batch in self.valid_dl:
                if count >= self.num_val_samples:
                    break

                codes = batch["codes"].to(self.device)  # [1, 8, T]
                prompt_audio = batch["prompt_audio"].to(self.device)  # [1, 1, T_p]
                texts = batch["texts"]
                batch_size = codes.size(0)
                message = torch.randint(0, 2, (batch_size, 16), dtype=torch.int64, device=self.device)

                codes_qbt = codes.permute(1, 0, 2).contiguous() if codes.shape[1] == 8 else codes
                quantized_layers = [generator.quantizer.decode(codes_qbt[k : k + 1], st=k) for k in range(8)]
                
                # 1. Clean TTS Reconstruction (without watermark)
                z_clean = sum(quantized_layers)
                clean_audio = generator.decoder(z_clean)

                # 2. Watermarked TTS Synthesis
                watermarked_layers = [msg_proc(q, message) for q in quantized_layers]
                z_wm = sum(watermarked_layers)
                wm_audio = generator.decoder(z_wm)

                # 3. UTMOS Evaluation (Clean vs WM)
                if getattr(self.utmos_loss, "model", None) is not None:
                    try:
                        c_u = self.utmos_loss.model(clean_audio.squeeze(1), self.sample_rate).mean().item()
                        w_u = self.utmos_loss.model(wm_audio.squeeze(1), self.sample_rate).mean().item()
                        clean_utmos_list.append(c_u)
                        wm_utmos_list.append(w_u)
                    except Exception:
                        pass

                # 4. Speaker SIM Evaluation (Clean vs WM)
                if hasattr(self, "sim_loss"):
                    try:
                        c_s = self.sim_loss.get_similarity(clean_audio, prompt_audio, self.sample_rate)
                        w_s = self.sim_loss.get_similarity(wm_audio, prompt_audio, self.sample_rate)
                        clean_sim_list.append(c_s)
                        wm_sim_list.append(w_s)
                    except Exception:
                        pass

                # 5. ASR WER / CER Evaluation (Clean vs WM)
                if getattr(self.asr_loss, "model", None) is not None:
                    try:
                        c_hyps = self.asr_loss.decode_greedy(clean_audio, self.sample_rate)
                        w_hyps = self.asr_loss.decode_greedy(wm_audio, self.sample_rate)
                        for ref_t, c_h, w_h in zip(texts, c_hyps, w_hyps):
                            c_wer, c_cer = compute_wer_cer(ref_t, c_h)
                            w_wer, w_cer = compute_wer_cer(ref_t, w_h)
                            clean_wer_list.append(c_wer)
                            clean_cer_list.append(c_cer)
                            wm_wer_list.append(w_wer)
                            wm_cer_list.append(w_cer)
                    except Exception:
                        pass

                # Robustness Evaluation across DSP + Codec attacks
                for cat, name, detail, atk_fn in val_attacks:
                    key = name if cat == "DSP" else f"{name} {detail}"
                    try:
                        attacked_audio = atk_fn(wm_audio)
                    except Exception as e:
                        attacked_audio = wm_audio

                    embedding = generator.forward_feature(attacked_audio)
                    logits, chunk_logits = detector(embedding)

                    # Extract bits
                    detect_prob, pred_bits, detected = detector.detect_watermark(embedding)
                    bit_correct = (pred_bits.long() == message.long()).sum().item()
                    total_b = message.numel()

                    # VAD detection accuracy
                    det_correct = (logits > 0.0).sum().item()
                    total_f = logits.numel()

                    results[key]["bit_matches"] += bit_correct
                    results[key]["total_bits"] += total_b
                    results[key]["det_matches"] += det_correct
                    results[key]["total_frames"] += total_f

                count += 1

        # Summary statistics
        summary = {}
        for key, stats in results.items():
            bit_acc = stats["bit_matches"] / max(1, stats["total_bits"])
            det_acc = stats["det_matches"] / max(1, stats["total_frames"])
            summary[key] = {
                "category": stats["category"],
                "family": stats["family"],
                "bitrate": stats["bitrate"],
                "bit_acc": bit_acc,
                "detect_acc": det_acc,
            }
            # TensorBoard logging
            clean_tag = key.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "_").replace("+", "p").replace("%", "pct").replace("-", "_")
            self.writer.add_scalar(f"val/{clean_tag}/bit_acc", bit_acc, global_step=step)
            self.writer.add_scalar(f"val/{clean_tag}/detect_acc", det_acc, global_step=step)

        # Compute average quality metrics (Clean vs Watermarked)
        c_ut = sum(clean_utmos_list) / max(1, len(clean_utmos_list)) if clean_utmos_list else 0.0
        w_ut = sum(wm_utmos_list) / max(1, len(wm_utmos_list)) if wm_utmos_list else 0.0
        c_sim = sum(clean_sim_list) / max(1, len(clean_sim_list)) if clean_sim_list else 0.0
        w_sim = sum(wm_sim_list) / max(1, len(wm_sim_list)) if wm_sim_list else 0.0
        c_wer = sum(clean_wer_list) / max(1, len(clean_wer_list)) if clean_wer_list else 0.0
        w_wer = sum(wm_wer_list) / max(1, len(wm_wer_list)) if wm_wer_list else 0.0
        c_cer = sum(clean_cer_list) / max(1, len(clean_cer_list)) if clean_cer_list else 0.0
        w_cer = sum(wm_cer_list) / max(1, len(wm_cer_list)) if wm_cer_list else 0.0

        quality_metrics = {
            "clean_utmos": c_ut, "wm_utmos": w_ut,
            "clean_sim": c_sim, "wm_sim": w_sim,
            "clean_wer": c_wer, "wm_wer": w_wer,
            "clean_cer": c_cer, "wm_cer": w_cer,
        }

        # Log quality metrics & deltas to TensorBoard
        self.writer.add_scalar("val/quality/clean_utmos", c_ut, global_step=step)
        self.writer.add_scalar("val/quality/wm_utmos", w_ut, global_step=step)
        self.writer.add_scalar("val/quality/delta_utmos", w_ut - c_ut, global_step=step)

        self.writer.add_scalar("val/quality/clean_speaker_sim", c_sim, global_step=step)
        self.writer.add_scalar("val/quality/wm_speaker_sim", w_sim, global_step=step)
        self.writer.add_scalar("val/quality/delta_speaker_sim", w_sim - c_sim, global_step=step)

        self.writer.add_scalar("val/quality/clean_asr_wer", c_wer, global_step=step)
        self.writer.add_scalar("val/quality/wm_asr_wer", w_wer, global_step=step)
        self.writer.add_scalar("val/quality/delta_asr_wer", w_wer - c_wer, global_step=step)

        self.writer.add_scalar("val/quality/clean_asr_cer", c_cer, global_step=step)
        self.writer.add_scalar("val/quality/wm_asr_cer", w_cer, global_step=step)
        self.writer.add_scalar("val/quality/delta_asr_cer", w_cer - c_cer, global_step=step)

        # Print formatted table
        table_str = format_full_validation_table(step, summary, quality_metrics=quality_metrics)
        print(table_str, flush=True)

        # Release cached attack models from GPU memory
        release_codec_models()
        print(f"[Validation @ Step {step:07d}] Codec cache released. Returning to training.\n", flush=True)

    def train(self):
        print(f"[Training] Starting NeuMark Training for {self.epochs} epochs...")
        self.generator.train()
        ensure_frozen_model_train_mode(self.generator)
        if hasattr(self, "utmos_loss") and getattr(self.utmos_loss, "model", None) is not None:
            ensure_frozen_model_train_mode(self.utmos_loss.model)
        for m in self.generator.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                m.eval()
        self.msg_processor.train()
        self.detector.train()
        for d in self.discriminators.values():
            d.train()

        steps = int(self.steps.item())
        for epoch in range(self.epochs):
            for i, batch in enumerate(tqdm(self.dl, desc=f"Epoch {epoch}", disable=not self.is_main, ncols=100)):
                codes = batch["codes"].to(self.device)  # [B, 8, T]
                real_audio = batch["audio"].to(self.device)  # [B, 1, T_samples]
                prompt_audio = batch["prompt_audio"].to(self.device)  # [B, 1, T_p]
                texts = batch["texts"]
                batch_size = codes.size(0)

                # Sample random 16-bit watermark message
                message = torch.randint(0, 2, (batch_size, 16), dtype=torch.int64, device=self.device)

                # -------------------------------------------------------------
                # 1. Decode Pre-computed Codes to Continuous Latents & Embed
                # -------------------------------------------------------------
                if i % self.grad_accum_steps == 0:
                    self.optim_generator.zero_grad()
                    if self.adv_loss_lambda > 0:
                        self.optim_discriminators.zero_grad()

                unwrapped_gen = self.accelerator.unwrap_model(self.generator)

                # Permute codes to [8, B, T] layout for SpeechTokenizer RVQ decode
                codes_qbt = codes.permute(1, 0, 2).contiguous() if codes.shape[1] == 8 else codes
                quantized_layers = [
                    unwrapped_gen.quantizer.decode(codes_qbt[k : k + 1], st=k)
                    for k in range(8)
                ]  # list of 8 tensors [B, 1024, T]
                z_q = sum(quantized_layers)

                # Layer-wise watermark injection
                watermarked_layers = [
                    self.msg_processor(q, message) for q in quantized_layers
                ]
                z_wm = sum(watermarked_layers)

                # Decode watermarked latent to waveform
                wm_audio = unwrapped_gen.decoder(z_wm)

                # Align temporal dimensions
                min_len = min(real_audio.shape[-1], wm_audio.shape[-1])
                real_audio_aligned = real_audio[..., :min_len]
                wm_audio_aligned = wm_audio[..., :min_len]

                # -------------------------------------------------------------
                # 2. Compute TTS-Native Losses
                # -------------------------------------------------------------
                # A. Latent Space Cosine Loss (Soft Anchor)
                loss_cos = latent_cosine_loss(z_wm, z_q) * self.cos_loss_lambda

                # B. UTMOS Naturalness Loss (Absolute MOS Maximization)
                loss_utmos = self.utmos_loss(wm_audio, self.sample_rate) * self.utmos_loss_lambda

                # C. Speaker Similarity Loss (WavLM vs Prompt)
                loss_sim = self.sim_loss(wm_audio, prompt_audio, self.sample_rate) * self.sim_loss_lambda

                # D. ASR Pronunciation Loss (CTC vs Target Text)
                loss_asr = self.asr_loss(wm_audio, texts, self.sample_rate) * self.asr_loss_lambda

                # E. GAN Adversarial & Feature Matching (Real = real_audio, Fake = wm_audio)
                loss_adv = torch.zeros((), device=self.device)
                loss_fm = torch.zeros((), device=self.device)
                adversarial_components = {}
                discriminator_components = {}

                if self.adv_loss_lambda > 0:
                    for discriminator in self.discriminators.values():
                        discriminator.requires_grad_(False)

                    fm_components = []
                    for name, discriminator in self.discriminators.items():
                        d_unwrapped = self.accelerator.unwrap_model(discriminator)
                        _, fake_preds, real_fmaps, fake_fmaps = d_unwrapped(
                            real_audio_aligned, wm_audio_aligned
                        )
                        comp = torch.stack([adversarial_loss_g(p) for p in fake_preds]).mean()
                        adversarial_components[name] = comp
                        fm_components.append(feature_loss(real_fmaps, fake_fmaps))

                    loss_fm = torch.stack(fm_components).mean()
                    loss_adv = (
                        torch.stack(list(adversarial_components.values())).mean() + 2.0 * loss_fm
                    ) * self.adv_loss_lambda

                # F. Watermark Decoding & Detection Loss (Strictly Encodec 3/6/12k, VC Masking, Clean)
                augmented_audio, vad_labels, attack_name = apply_train_augmentation(
                    wm_audio, sample_rate=self.sample_rate, orig_audio=real_audio
                )
                logits, chunk_logits = self.detect_watermark(augmented_audio, return_logits=True)

                target_chunks = bits_to_chunks(message.float(), chunk_size=self.nchunk_size)
                loss_dec = decoding_loss(chunk_logits, target_chunks) * self.dec_loss_lambda

                min_lens_pos = min(logits.shape[-1], vad_labels.shape[-1])
                loss_vad_pos = vad_based_loss(logits[..., :min_lens_pos], vad_labels[..., :min_lens_pos], from_logits=True) * self.vad_loss_lambda

                # Negative sample supervision (Real unwatermarked audio)
                augmented_neg, _, _ = apply_train_augmentation(
                    real_audio, sample_rate=self.sample_rate, orig_audio=None
                )
                neg_logits, _ = self.detect_watermark(augmented_neg, return_logits=True)
                vad_labels_neg = torch.zeros_like(neg_logits)
                min_lens_neg = min(neg_logits.shape[-1], vad_labels_neg.shape[-1])
                loss_vad_neg = vad_based_loss(neg_logits[..., :min_lens_neg], vad_labels_neg[..., :min_lens_neg], from_logits=True) * self.vad_loss_lambda

                # -------------------------------------------------------------
                # 3. Total Loss & Generator Backward
                # -------------------------------------------------------------
                total_loss = (
                    loss_dec
                    + loss_vad_pos
                    + loss_vad_neg
                    + loss_cos
                    + loss_adv
                    + loss_utmos
                    + loss_sim
                    + loss_asr
                )

                # Normalize loss for gradient accumulation
                self.accelerator.backward(total_loss / self.grad_accum_steps)

                # -------------------------------------------------------------
                # 4. Train Discriminators
                # -------------------------------------------------------------
                loss_D = torch.zeros((), device=self.device)
                if self.adv_loss_lambda > 0:
                    for discriminator in self.discriminators.values():
                        discriminator.requires_grad_(True)

                    for name, discriminator in self.discriminators.items():
                        real_preds, fake_preds, _, _ = discriminator(
                            real_audio_aligned, wm_audio_aligned.detach()
                        )
                        comp = torch.stack([adversarial_loss_d(r, f) for r, f in zip(real_preds, fake_preds)]).mean()
                        discriminator_components[name] = comp

                    loss_D = torch.stack(list(discriminator_components.values())).mean()
                    self.accelerator.backward(loss_D / self.grad_accum_steps)

                # Optimizer step on accumulation boundary
                if (i + 1) % self.grad_accum_steps == 0 or (i + 1) == len(self.dl):
                    self.optim_generator.step()
                    self.optim_generator.zero_grad()
                    if self.adv_loss_lambda > 0:
                        self.optim_discriminators.step()
                        self.optim_discriminators.zero_grad()

                    if steps < self.num_warmup_steps:
                        lr = self.warmup(steps)
                        for pg in self.optim_generator.param_groups:
                            pg["lr"] = lr
                        for pg in self.optim_discriminators.param_groups:
                            pg["lr"] = lr
                    else:
                        self.scheduler_generator.step()
                        self.scheduler_discriminator.step()

                if steps % self.cfg.get("log_steps", 50) == 0 or steps in {1, 5, 10}:
                    log_dict = {
                        "train/total_loss": total_loss.item(),
                        "train/dec_loss": loss_dec.item(),
                        "train/vad_pos": loss_vad_pos.item(),
                        "train/vad_neg": loss_vad_neg.item(),
                        "train/cos_loss": loss_cos.item(),
                        "train/utmos_loss": loss_utmos.item(),
                        "train/sim_loss": loss_sim.item(),
                        "train/asr_loss": loss_asr.item(),
                        "train/adv_loss": loss_adv.item(),
                        "train/fm_loss": loss_fm.item(),
                        "train/d_loss": loss_D.item(),
                    }
                    self.log(log_dict, step=steps)

                # Periodic Checkpoint Saving
                if steps > 0 and steps % self.save_steps == 0:
                    self.save(steps, epoch)

                # Validation Loop
                if steps > 0 and steps % self.val_steps == 0:
                    self.validate(steps)
                    self.generator.train()
                    ensure_frozen_model_train_mode(self.generator)
                    if hasattr(self, "utmos_loss") and getattr(self.utmos_loss, "model", None) is not None:
                        ensure_frozen_model_train_mode(self.utmos_loss.model)
                    self.msg_processor.train()
                    self.detector.train()
                    for d in self.discriminators.values():
                        d.train()

                self.steps += 1
                steps = int(self.steps.item())

            # Save checkpoint once every epoch
            self.save(steps, epoch)

        self.save(steps, self.epochs, final=True)
        print(f"[Done] Training complete after {steps} steps.")

    def save(self, steps: int, epoch: int, final: bool = False):
        if not self.is_main:
            return
        prefix = "NeuMark_final" if final else f"NeuMark_step_{steps:07d}_epoch_{epoch:03d}"
        ckpt_path = self.results_folder / f"{prefix}.pt"
        pkg = dict(
            msg_processor=self.accelerator.get_state_dict(self.msg_processor),
            detector=self.accelerator.get_state_dict(self.detector),
            discriminators={k: self.accelerator.get_state_dict(v) for k, v in self.discriminators.items()},
            optim_generator=self.optim_generator.state_dict(),
            optim_discriminators=self.optim_discriminators.state_dict(),
            steps=steps,
            epoch=epoch,
        )
        torch.save(pkg, ckpt_path)
        print(f"[Checkpoint] Saved model checkpoint: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config_tts_native.json"), help="Config path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    trainer = NeuMarkTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()

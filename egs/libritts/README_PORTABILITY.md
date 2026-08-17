# 🚀 NeuMark 跨设备部署与迁移指南 (Cross-Device Portability Guide)

本文档说明如何在**全新的 Linux 设备**（或其他未配置过环境的主机）上一站式配置并运行 **NeuMark** 的模型训练与语音合成推理。

---

## 1. 目录结构推荐规范

在工作空间根目录下，推荐将 `vall-e` 和 `NeuMark` 作为平级目录放置：

```text
workspace/
├── vall-e/                                # 当前仓库
│   └── egs/libritts/
│       ├── config_tts_native.json         # 训练主配置文件
│       ├── prepare_assets.py              # 🚀 跨设备资产自检与诊断工具
│       ├── infer_neumark.py               # 🔊 水印语音合成与听感对比脚本
│       ├── tts_native_train.py            # 训练脚本
│       ├── tts_native_train.slurm         # Slurm 批处理调度脚本
│       ├── models/
│       │   └── wavlm_large_finetune.pth   # (大文件) WavLM 说话人模型 (1.2 GB)
│       └── data/
│           └── tokenized_voicemark/       # (大文件) Lhotse 离线声学特征数据
│               ├── cuts_train.jsonl.gz    # (38.6 MB)
│               └── cuts_dev.jsonl.gz      # (0.7 MB)
└── NeuMark/                               # 外部依赖仓库
    ├── models/
    │   └── __init__.py                    # WMEmbedder, WMDetector 模型定义
    └── STmodels/
        └── pretrained_model/
            ├── SpeechTokenizer.pt         # (大文件) SpeechTokenizer 预训练权重 (459 MB)
            └── speechtokenizer_hubert_avg_config.json
```

> [!TIP]
> **自定义路径**：如果 `NeuMark` 仓库不在默认的平级目录，只需设置环境变量：
> ```bash
> export NEUMARK_ROOT="/path/to/your/custom/NeuMark"
> ```
> 脚本会自动优先识别该环境变量！

---

## 2. Python 依赖环境安装

建议使用 Python 3.10+ 环境：

```bash
# 核心依赖库
pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install accelerate lhotse julius tensorboard

# 音频编解码攻击库
pip install encodec
pip install descript-audio-codec  # dac
pip install snac
```

---

## 3. 大文件资产准备（不可通过 Git 追踪的文件）

从源机器拷贝或解压以下 3 个关键资产到对应位置：

1. **SpeechTokenizer 权重（459 MB）**：
   * 放置于 `NeuMark/STmodels/pretrained_model/SpeechTokenizer.pt`
   * 配置文件放置于 `NeuMark/STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json`
2. **WavLM 说话人模型（1.2 GB）**：
   * 放置于 `vall-e/egs/libritts/models/wavlm_large_finetune.pth`
3. **数据 Manifest（~40 MB）**：
   * 放置于 `vall-e/egs/libritts/data/tokenized_voicemark/cuts_train.jsonl.gz` 和 `cuts_dev.jsonl.gz`

---

## 4. 运行一键就绪诊断工具

在新设备上进入 `egs/libritts/` 目录，执行诊断自检：

```bash
cd vall-e/egs/libritts
python3 prepare_assets.py
```

若全部依赖和文件就绪，将输出：
```text
===========================================================================
  🎉 ALL CHECKS PASSED! The environment is 100% ready for training & inference.
===========================================================================
```

---

## 5. 启动训练与推理

### 启动训练（Slurm 或直接运行）：
```bash
# Slurm 集群调度
sbatch tts_native_train.slurm

# 或单卡本地直接运行
python3 tts_native_train.py --config config_tts_native.json
```

### 启动推理与生成听感对比：
```bash
python3 infer_neumark.py \
    --checkpoint exp/tts_native_neumark/NeuMark_epoch_000.pt \
    --manifest data/tokenized_voicemark/cuts_dev.jsonl.gz \
    --sample_index 0 \
    --message "1011001110001101" \
    --output_dir exp/neumark_listening_demo
```

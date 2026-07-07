# Persian Whisper Fine-Tuning with LoRA + CPU Deployment (GGUF)

Fine-tuning OpenAI's `whisper-large-v3-turbo` for Persian (Farsi) automatic speech recognition (ASR), optimized for efficient CPU-only inference via `whisper.cpp`.

## 🎯 Motivation

Off-the-shelf Whisper models perform inconsistently on Persian audio due to limited representation in pretraining data. This project fine-tunes the model specifically on Persian speech, then compresses it into a lightweight format that runs on consumer CPUs without requiring a GPU — making it practical for local/offline deployment.

## 🧩 Pipeline

```
Common Voice (Persian) Dataset
        │
        ▼
Feature Extraction & Tokenization
        │
        ▼
LoRA Fine-Tuning (8-bit base model)
        │
        ▼
Merge LoRA Adapters → Full Model
        │
        ▼
Convert to GGML/GGUF Format
        │
        ▼
Q8_0 Quantization (whisper.cpp)
        │
        ▼
CPU Inference + WER Evaluation
```

## 🛠️ Tech Stack

- **Model**: `openai/whisper-large-v3-turbo`
- **Fine-tuning**: LoRA (PEFT) with 8-bit quantized base model (LoRA-style)
- **Dataset**: [hezarai/common-voice-13-fa](https://huggingface.co/datasets/hezarai/common-voice-13-fa)
- **Deployment**: `whisper.cpp` (GGUF, Q8_0 quantization) for CPU inference
- **Evaluation**: Word Error Rate (WER) via `jiwer`
- **Libraries**: `transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`, `librosa`

## 📁 Repository Structure

```
persian-whisper-finetune/
├── data/
│   └── prepare_dataset.py       # Downloads & preprocesses Common Voice (Persian)
├── train/
│   └── train_lora.py            # LoRA fine-tuning script
├── convert/
│   └── merge_and_convert_gguf.sh # Merges LoRA weights, converts to GGUF, quantizes
├── evaluate/
│   └── eval_wer.py              # Benchmarks WER and inference time on a test set
├── results/
│   ├── loss_curve.png
│   └── wer_results.csv
├── requirements.txt
└── README.md
```

## 📊 Results

| Model | WER (%) | Avg. Inference Time (s) | Notes |
|---|---|---|---|
| Baseline (whisper-large-v3-turbo) | _TBD_ | _TBD_ | Pretrained, no fine-tuning |
| Fine-tuned (LoRA) | _TBD_ | _TBD_ | After Persian fine-tuning |
| Fine-tuned + GGUF Q8_0 (CPU) | _TBD_ | _TBD_ | Quantized for CPU deployment |

> Fill in actual numbers from `evaluate/eval_wer.py` before publishing.

**Training loss curve:**

![Loss Curve](results/loss_curve.png)

## 🚀 Usage

### 1. Prepare the dataset
```bash
python data/prepare_dataset.py
```

### 2. Fine-tune with LoRA
```bash
python train/train_lora.py
```

### 3. Merge & convert to GGUF
```bash
bash convert/merge_and_convert_gguf.sh
```

### 4. Evaluate
```bash
python evaluate/eval_wer.py
```

## 🔑 Key Highlights

- End-to-end pipeline: data preparation → fine-tuning → model compression → deployment → evaluation
- Memory-efficient training using 8-bit quantization + LoRA (trains large model on limited GPU memory)
- Production-ready output: quantized GGUF model runs fully offline on CPU
- Quantitative evaluation using WER, not just qualitative transcription samples



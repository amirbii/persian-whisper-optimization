# Persian Whisper Optimization

بهینه‌سازی، تنظیم دقیق و ارزیابی مدل‌های **Whisper** برای بازشناسی گفتار فارسی، با تمرکز بر آموزش کم‌هزینه با **LoRA و 8-bit quantization** و اجرای مدل روی **CPU** با `whisper.cpp` و `OpenVINO`.

> این مخزن یک روند چندمرحله‌ای را پوشش می‌دهد: آماده‌سازی داده، Fine-tuning، ادغام LoRA، تبدیل و Quantization مدل، اجرای inference و مقایسه‌ی نتایج بر اساس زمان پردازش و WER.

## نمای کلی فرایند

```mermaid
flowchart LR
    A[Common Voice 13 Persian] --> B[Preprocessing at 16 kHz]
    B --> C[Whisper input features and labels]
    C --> D[8-bit Fine-tuning with LoRA]
    D --> E[Merge LoRA with base model]
    E --> F[GGML / Q8_0 with whisper.cpp]
    E --> G[OpenVINO export]
    F --> H[CPU Evaluation]
    G --> H
    H --> I[WER and inference-time reports]
```

## اهداف پروژه

- آماده‌سازی دیتاست فارسی Mozilla Common Voice برای Whisper
- تنظیم دقیق `openai/whisper-large-v3-turbo` با LoRA
- کاهش مصرف حافظه‌ی آموزش با بارگذاری 8-bit
- ادامه‌ی خودکار آموزش از آخرین checkpoint
- ادغام LoRA با مدل پایه و تبدیل به فرمت سازگار با `whisper.cpp`
- ساخت مدل Quantized با قالب `Q8_0` برای CPU
- آزمایش مسیر OpenVINO برای inference روی CPU
- مقایسه‌ی مدل‌ها با معیار **Word Error Rate (WER)** و زمان پردازش

## ساختار مخزن

```text
persian-whisper-optimization/
├── notebooks/
│   ├── data_preparation.ipynb
│   ├── fine_tune_model.ipynb
│   ├── test.ipynb
│   └── whisper_ov_test.ipynb
├── results/
│   ├── loss.png
│   ├── whisper_fa_q8_test_results_colab_CPU.xlsx
│   ├── whisper_large_v3_turbo_test_results_colab_CPU.xlsx
│   └── whisper_ov_fp16_test_results_colab_CPU .xlsx
└── README.md
```



## اجزای پروژه

### 1. آماده‌سازی داده‌ها

فایل `notebooks/data_preparation.ipynb`:

- دریافت دیتاست `hezarai/common-voice-13-fa`
- استفاده از splitهای `train`، `validation` و `test`
- Resample کردن صوت‌ها به `16 kHz`
- استخراج `input_features` با پردازشگر Whisper
- Tokenize کردن متن فارسی و ساخت `labels`
- ذخیره‌ی داده‌های پردازش‌شده در Google Drive با `save_to_disk`

تعداد نمونه‌های ثبت‌شده در خروجی نوت‌بوک:

| Split | تعداد نمونه |
|---|---:|
| Train | 28,024 |
| Validation | 10,440 |
| Test | 10,440 |

### 2. Fine-tuning مدل

فایل `notebooks/fine_tune_model.ipynb` مدل پایه را با ترکیب **BitsAndBytes + LoRA** آموزش می‌دهد.

| پارامتر | مقدار |
|---|---|
| Base model | `openai/whisper-large-v3-turbo` |
| Language / task | Persian / transcribe |
| Model loading | 8-bit |
| LoRA rank | `32` |
| LoRA alpha | `64` |
| Target modules | `q_proj`, `v_proj` |
| LoRA dropout | `0.05` |
| Batch size | `4` |
| Gradient accumulation | `4` |
| Effective batch size | `16` |
| Learning rate | `1e-3` |
| Maximum steps | `1000` |
| Evaluation interval | هر `100` step |
| Checkpoint interval | هر `100` step |
| Maximum saved checkpoints | `2` |
| Mixed precision | FP16 |

آموزش در صورت وجود checkpoint به‌صورت خودکار ادامه پیدا می‌کند. در اجرای ذخیره‌شده، آموزش از checkpointهای `800` و `900` شناسایی و تا step `1000` ادامه یافته است.

آخرین مقادیر ثبت‌شده در خروجی آموزش:

| Step | Training loss | Validation loss |
|---:|---:|---:|
| 1000 | 0.533144 | 0.238728 |

### 3. ادغام و Quantization

پس از آموزش:

1. آداپتور LoRA با مدل پایه ادغام می‌شود.
2. مدل ادغام‌شده در قالب Hugging Face ذخیره می‌شود.
3. مدل با ابزارهای `whisper.cpp` به GGML تبدیل می‌شود.
4. نسخه‌ی `Q8_0` برای اجرای CPU تولید می‌شود.

خروجی ثبت‌شده‌ی Quantization:

| مدل | اندازه |
|---|---:|
| مدل پیش از Quantization | 3,085.62 MB |
| مدل Q8_0 | 833.08 MB |
| کاهش اندازه | حدود 73٪ |

فایل نهایی در نوت‌بوک با نام زیر ساخته می‌شود:

```text
whisper-fa-q8_0.bin
```

### 4. ارزیابی مدل‌ها

فایل `notebooks/test.ipynb` یک `mini_corpus.zip` شامل پوشه‌ی `clips` و فایل `validated.tsv` دریافت می‌کند و برای هر نمونه موارد زیر را محاسبه می‌کند:

- متن مرجع
- متن پیش‌بینی‌شده
- زمان پردازش
- WER

نتایج به‌صورت CSV/XLSX ذخیره می‌شوند. نوت‌بوک روی تمام ردیف‌های موجود در `validated.tsv` حلقه می‌زند؛ فایل‌های فعلی پوشه‌ی `results` شامل 20 نمونه‌ی ارزیابی و یک ردیف میانگین هستند.

## نتایج فعلی

مقادیر زیر مستقیماً از فایل‌های موجود در پوشه‌ی `results` استخراج شده‌اند. مقدار کمتر برای WER بهتر است.

| مدل / فرمت | محیط گزارش‌شده | میانگین زمان هر فایل | میانگین WER |
|---|---|---:|---:|
| Persian fine-tuned Q8 | Colab CPU | 60.07 ثانیه | 91.90٪ |
| Whisper large-v3-turbo | Colab CPU | 45.40 ثانیه | 96.87٪ |
| OpenVINO FP16 | Colab CPU | 37.86 ثانیه | 94.69٪ |

بر اساس این اجرای کوچک، نسخه‌ی Fine-tuned Q8 کمترین WER را داشته، اما کندترین زمان inference را ثبت کرده است. OpenVINO سریع‌ترین میانگین زمان را در میان سه گزارش فعلی دارد.

### نمودار Loss

![Training and validation loss](results/loss.png)

## نصب پیش‌نیازها

محیط اصلی نوت‌بوک‌ها Google Colab است.

```bash
pip install transformers datasets librosa evaluate jiwer gradio
pip install peft "bitsandbytes>=0.46.1" accelerate
pip install optimum optimum-intel openvino
```

برای نوت‌بوک ارزیابی مبتنی بر OpenAI Whisper:

```bash
pip install git+https://github.com/openai/whisper.git
pip install torchaudio jiwer
```

## ترتیب اجرای پیشنهادی

1. `notebooks/data_preparation.ipynb`
2. `notebooks/fine_tune_model.ipynb`
3. تبدیل و Quantization در بخش‌های پایانی همان نوت‌بوک
4. `notebooks/test.ipynb` برای ارزیابی مدل‌ها
5. `notebooks/whisper_ov_test.ipynb` برای آزمایش OpenVINO روی CPU

## مسیرهای Google Drive

نوت‌بوک‌های Colab از ساختار زیر استفاده می‌کنند:

```text
/content/drive/MyDrive/whisper_fa_dataset/
├── cache/
├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
├── models/
├── final_lora_model/
├── merged_model/
└── gguf_models/
```

در صورت تغییر محل پروژه، متغیرهای `BASE_DIR`، `PROCESSED_PATH`، `CHECKPOINT_DIR` و مسیرهای خروجی را به‌روزرسانی کنید.

## اجرای OpenVINO

فایل `notebooks/whisper_ov_test.ipynb` برای محیط محلی ویندوز نوشته شده و دارای مسیرهای مطلق است. پیش از اجرا این مقادیر را تغییر دهید:

```python
model_id = r"PATH_TO_HUGGINGFACE_MODEL"
save_dir = r"PATH_TO_OPENVINO_MODEL"
MODEL_PATH = r"PATH_TO_OPENVINO_MODEL"
AUDIO_PATH = r"PATH_TO_16KHZ_AUDIO.wav"
```

نمونه‌ی inference:

```python
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor
import librosa

processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = OVModelForSpeechSeq2Seq.from_pretrained(MODEL_PATH, device="CPU")

audio, _ = librosa.load(AUDIO_PATH, sr=16000)
inputs = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
).input_features

predicted_ids = model.generate(inputs)
text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(text)
```









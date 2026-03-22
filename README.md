# Heart Failure Risk Assessment - QLoRA Fine-tuned Phi-3 Mini

A fine-tuned clinical decision support model that accepts natural language patient descriptions and outputs mortality risk assessments with reasoning. Built on Microsoft Phi-3 Mini using QLoRA on a free Kaggle T4 GPU.

---

## What It Does

Input - plain natural language:
```
Patient is a 72-year-old male with high blood pressure and diabetes.
Ejection fraction is 22%, serum creatinine is 2.4 mg/dL,
serum sodium is 128 mEq/L, CPK is 1400 mcg/L. Follow-up: 30 days.
```

Output - risk assessment with reasoning:
```
This patient is at high risk of mortality.

Key contributing factors:
- Severely reduced ejection fraction (22%) indicating poor cardiac function
- Elevated serum creatinine - reduced kidney function
- Critically low serum sodium (hyponatremia)
- Elevated CPK suggesting possible muscle damage

Outcome: The patient did not survive the follow-up period.
```

If clinical data is incomplete, the model flags exactly what is missing:
```
⚠️ Incomplete clinical context detected.
Serum creatinine and CPK level were not provided.
These are critical for a complete assessment.
```

---

## Dataset

- **Source:** UCI Heart Failure Clinical Records Dataset (299 patients, 13 clinical features)
- **Pipeline:** Each CSV row converted to a natural language paragraph with clinical context added per feature (normal ranges, severity labels)
- **Training samples:** 399 total
  - 299 full patient records with complete clinical profiles
  - 100 partial records with deliberately dropped features to teach the model to flag missing context
- **Format:** Alpaca-style JSONL `{instruction, input, output}` → formatted into Phi-3 instruct template

---

## Model

| Property | Value |
|---|---|
| Base model | microsoft/Phi-3-mini-4k-instruct |
| Parameters | 3.8B |
| Quantization | 4-bit NF4 (QLoRA) |
| Trainable params | 3,145,728 (0.08%) |
| Target modules | qkv_proj, o_proj |
| LoRA rank | 16 |
| LoRA alpha | 32 |

---

## Training Config

| Hyperparameter | Value |
|---|---|
| Epochs | 3 |
| Batch size | 4 |
| Gradient accumulation | 4 (effective batch = 16) |
| Learning rate | 2e-4 |
| Warmup steps | 10 |
| Optimizer | adamw_8bit |
| Max sequence length | 512 |
| Hardware | Kaggle T4 x2 GPU |
| Training time | ~20 minutes |

---

## Stack

```
unsloth       - QLoRA training, 4-bit model loading
peft          - LoRA adapter management
trl           - SFTTrainer
transformers  - model architecture, tokenizer
datasets      - JSONL loading and formatting
```

---

## Project Structure

```
heart_lora_ready.jsonl        training data (399 samples)
heart_lora_train.py           full training + inference script
heart-disease-lora.ipynb      Kaggle notebook
/heart-lora-final/
    adapter_model.safetensors   LoRA weights (12.6 MB)
    adapter_config.json
    tokenizer.json
```

---

## Key Design Decisions

**CSV → Text conversion** - Raw tabular data is unreadable to LLMs. Each row was converted to a clinical paragraph with domain context added (e.g. "ejection fraction of 20%, which is severely reduced - critically below the normal range of 55–70%"). This lets the model reason about values rather than memorize numbers.

**Incomplete sample training** - 100 of the 399 samples had clinical features deliberately dropped during data preparation. This teaches the model to recognize when input is incomplete and respond with specific warnings rather than overconfident assessments.

**Natural language inference** - At inference time, input is free-form text. No structured fields required. The model extracts what it can and flags what's missing.

---

## Adapter Size

The full Phi-3 Mini base model is 3.8B parameters (~7.6 GB). The trained LoRA adapter is **12.6 MB** - only the delta from fine-tuning is stored.

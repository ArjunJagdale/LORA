### Cell 1 
```
!pip install -q unsloth datasets
```
This installs two Python libraries:

* unsloth — A library that makes fine-tuning large language models (LLMs) faster and more memory-efficient. It has special optimizations for training on consumer/limited GPUs (like on Kaggle's free tier). The "LoRA" in the notebook title is enabled through this.
* datasets — A library by HuggingFace for easily loading, processing, and managing datasets (including JSONL files, CSVs, etc.).

The -q flag means quiet mode — suppresses verbose installation logs so the output stays clean.

### Cell 2
```
from unsloth import FastLanguageModel
import os
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
```
* FastLanguageModel (from unsloth) — The main class used to load and fine-tune the LLM efficiently. Think of it as the "engine" of this whole notebook.
* os — Standard Python library for interacting with the file system — listing files, building paths, etc.
* torch — PyTorch, the deep learning framework everything runs on. Handles tensors, GPU computation, and model training under the hood.
* load_dataset (from datasets) — The function that will load the heart disease JSONL file into memory as a usable dataset object.
* SFTTrainer (from trl) — "Supervised Fine-Tuning Trainer". This is the actual training loop manager. trl is HuggingFace's library for fine-tuning LLMs with human feedback or supervised data.
* TrainingArguments — A config object where you define how training happens — learning rate, batch size, epochs, etc.
* TextStreamer — Used during inference to stream output token by token (like how ChatGPT types word by word). Not heavily used in this notebook but imported for potential use.

### Cell 3
```
JSONL_PATH = "/kaggle/input/datasets/lightvvcx/heart-lora-jsonl/heart_lora_ready.jsonl"
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = "/kaggle/working/heart-lora-final"
```

JSONL_PATH — Where is the training data?
* This points to the heart disease dataset stored on Kaggle.
* It's a .jsonl file — meaning JSON Lines format
* Each line in this file is one training example, like:
```
{"instruction": "You are a clinical decision support assistant specialized in heart failure risk assessment. Given a patient's clinical profile in natural language, assess their risk of mortality, explain the key contributing factors, and clearly flag if any critical clinical information is missing that would affect the confidence of your assessment.", "input": "Patient is a 61-year-old male. He has a known medical history of anaemia, diabetes. He is a non-smoker. Ejection fraction is 38.0%, which is reduced. This is below the normal range of 55\u201370% and suggests moderate cardiac dysfunction. Creatinine phosphokinase (CPK) level is 80 mcg/L, which is normal. Within the expected range, suggesting no significant muscle damage. Serum creatinine is 1.4 mg/dL, which is mildly elevated. Slightly above normal, suggesting early-stage kidney stress. Serum sodium is 137.0 mEq/L, which is normal. Within the healthy electrolyte range. Platelet count is 282,000, which is normal. Within the expected range. The patient was monitored over a follow-up period of 213 days.", "output": "Assessment: This patient carries a very high risk of mortality from heart failure.\n\nProtective factors include: normal electrolyte balance. However, the following risk factors require monitoring: reduced ejection fraction; mildly elevated serum creatinine; diabetes; anaemia.\n\nOutcome: The patient survived the follow-up period."}
{"instruction": "You are a clinical decision support assistant specialized in heart failure risk assessment. Given a patient's clinical profile in natural language, assess their risk of mortality, explain the key contributing factors, and clearly flag if any critical clinical information is missing that would affect the confidence of your assessment.", "input": "Patient is a 85-year-old male. He has a known medical history of anaemia. He is a non-smoker. Ejection fraction is 50.0%, which is mildly below normal. This is slightly under the normal range of 55\u201370% and warrants monitoring. CPK level is 910 mcg/L, which is mildly elevated. Slightly above normal, which may suggest minor cardiac or muscular stress. Serum creatinine is 1.3 mg/dL, which is mildly elevated. Slightly above normal, suggesting early-stage kidney stress. Follow-up period: 121 days.", "output": "Assessment: Based on the available information, this patient's risk appears moderate. Survival was observed during follow-up.\n\n\u26a0\ufe0f Incomplete clinical context detected. The following information was not provided: platelet count, serum sodium (electrolyte balance). A complete and confident risk assessment requires all clinical markers. Please provide the missing values for a more accurate evaluation.\n\nOutcome: The patient survived the follow-up period."}
```
* The model will read thousands of these examples and learn from them

MODEL_ID — Which base model are we fine-tuning?
"microsoft/Phi-3-mini-4k-instruct" is the starting point — a pre-trained LLM made by Microsoft.

* Phi-3-mini — small but powerful, only ~3.8 billion parameters
* 4k — can handle up to 4096 tokens (roughly 3000 words) in one go
* instruct — already trained to follow instructions (not just predict text)

We're not training from scratch. We're taking this already-smart model and teaching it to specialise in heart failure assessment. That's what fine-tuning means.

OUTPUT_DIR — Where do we save the result?
After training, the fine-tuned LoRA adapter weights get saved here.

* This is a Kaggle working directory (persists after the session)
* Later in the notebook, inference loads from this exact path
* It won't save the full model — just the small LoRA adapter (a few MBs, not GBs)

### Cell 4
```
LORA_R = 16
LORA_ALPHA = 32
EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 2e-4
MAX_SEQ_LEN = 512
```


EPOCHS = 3 — How Many Times to See the Data
* The model will go through your entire dataset 3 times during training.
* Too few → underfitting, model doesn't learn enough
* Too many → overfitting, model memorises training data and fails on new inputs
* 3 is conservative and safe for small medical datasets

BATCH_SIZE = 4 — Samples Per Step
* How many training examples the model sees at once before updating weights.
* Larger batches → more stable training, but needs more GPU memory
* 4 is small — chosen because medical LLM training on Kaggle's free GPU is memory-constrained

GRAD_ACCUM = 4 — Gradient Accumulation
* A clever workaround for small batch sizes.
* Instead of updating weights after every batch of 4, it accumulates gradients over 4 batches and then updates — effectively simulating a batch size of 4 × 4 = 16.
* Same result as a bigger batch
* Without needing more GPU memory
* Just takes a bit longer per update

LR = 2e-4 — Learning Rate
* How big each weight update step is. 2e-4 = 0.0002.
* Too high → model overshoots, training becomes unstable
* Too low → model learns painfully slowly
* 2e-4 is the standard recommended rate for LoRA fine-tuning

MAX_SEQ_LEN = 512 — Maximum Token Length
* The longest input+output the model will process. Anything longer gets cut off (truncated).
* 512 tokens ≈ ~380 words
* Enough for a patient description + clinical assessment
* Keeping it low saves GPU memory significantly



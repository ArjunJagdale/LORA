# Titanic Survival - LoRA Fine-tuned Reasoning Model

Fine-tuned `Qwen2.5-0.5B-Instruct` with LoRA to reason in natural language over Titanic passenger data and predict survival outcomes.

---

## What it does

Takes a passenger profile as input and produces a natural language reasoning paragraph before predicting survival - not just a label, but a human-readable explanation of *why*.

**Input**
```
- Name: Astor, Mr. John Jacob
- Sex: male
- Age: 47
- Class: 1
- Fare: £227.52
- Siblings/Spouse aboard: 1
- Embarked from: Cherbourg
```

**Output**
```
Astor was an adult male passenger travelling first class, boarding at Cherbourg.
Men had lower evacuation priority. As a first class passenger, they had priority
access to lifeboats. Paid a high fare of £227.5. Travelled with 1 family member(s).
Prediction: Did not survive.
```

---

## Training details

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Method | LoRA (r=16, alpha=32) |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Trainable params | 2,162,688 / 496,195,456 (0.44%) |
| Dataset | Titanic train.csv - 891 rows |
| Training examples | 2,673 (3 question variants per row) |
| Train / Val split | 90 / 10 |
| Epochs | 4 |
| Batch size | 2 (grad accum 8, effective batch 16) |
| Learning rate | 1e-4 with cosine scheduler |
| Hardware | Kaggle Tesla T4 (16GB) |
| Training time | ~57 minutes |

**Loss progression**

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 0.1579 | 0.1636 |
| 2 | 0.0782 | 0.0860 |
| 3 | 0.0498 | 0.0571 |
| 4 | 0.0448 | 0.0554 |

---

## Project structure

```
├── titanic_cot_lora.ipynb   ← full training notebook (Kaggle)
├── train.csv                ← Titanic training data
├── test.csv                 ← Titanic test data (not used in training)
└── output/
    └── lora/final/          ← saved LoRA adapter weights
```

---

## How to run

1. Open on Kaggle, add the Titanic competition dataset
2. Set accelerator to GPU T4
3. Run all cells top to bottom - training takes ~57 minutes

```
Cell 1  : GPU setup (single GPU enforced)
Cell 2-5: Load + clean CSV, build CoT examples
Cell 6  : Save training JSONL
Cell 7  : Load base model + attach LoRA adapter
Cell 8  : Format dataset for instruction tuning
Cell 9  : Train + save adapter
Cell 10 : Run inference and print reasoning
```

---

## Key design decisions

**Natural language reasoning over rigid steps** - instead of forcing a Step 1/2/3 format, the model is trained to produce a single coherent paragraph. This is more natural for a 0.5B model and produces more fluent output.

**3 question variants per row** - each passenger row generates 3 training examples with different question phrasings, tripling the effective dataset size to 2,673 examples without any additional data.

**No quantization** - removed 4-bit quantization to avoid compatibility issues with newer transformers versions. At 0.5B, the model fits comfortably in bfloat16 on a T4.

**Reasoning is grounded in historical rules** - evacuation priority (women first, class priority, family dynamics) is baked into the training labels, so the model learns to cite real factors rather than hallucinate.

---

> Research and portfolio project. Not a production system.

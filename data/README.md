# Datasets

This directory contains training and evaluation datasets for Alexthia.

## Current Datasets (Alexthia v0.5)

### 1. MATH-Hard - Competition-Level Mathematics
- **Source:** https://huggingface.co/datasets/lighteval/MATH-Hard
- **Training examples:** 2,304 (entire dataset)
- **Usage:** 100% of available data
- **Format:** Problem-solution pairs with detailed steps
- **Topics:** Advanced algebra, number theory, geometry, calculus, combinatorics
- **Difficulty:** Competition-level (AMC, AIME, USAMO)
- **Why this dataset:** Teaches rigorous proof construction and multi-step reasoning

### 2. GSM8K - Grade School Math Word Problems
- **Source:** https://huggingface.co/datasets/openai/gsm8k
- **Training examples:** 3,000 (of 7,473 available)
- **Usage:** ~40% of dataset
- **Format:** Natural language questions with step-by-step solutions
- **Topics:** Arithmetic, basic algebra, word problems
- **Difficulty:** Grade school level (5-8th grade)
- **Why this dataset:** Teaches problem parsing and clear explanatory reasoning

### 3. SciQ - Science Reasoning Questions
- **Source:** https://huggingface.co/datasets/allenai/sciq
- **Training examples:** 2,000 (of 11,679 available)
- **Usage:** ~17% of dataset
- **Format:** Multiple-choice science questions with explanations
- **Topics:** Physics, chemistry, biology, earth science
- **Difficulty:** Middle school to high school level
- **Why this dataset:** Teaches scientific reasoning and cross-domain thinking

### Training Summary
- **Total examples:** 7,304
- **Difficulty distribution:** 31.5% hard, 41.1% medium, 27.4% easier
- **Training time:** 6-8 hours on Kaggle T4 GPU
- **Result:** Alexthia v0.5 (multi-domain math reasoning)

---

## Dataset Format

All datasets are converted to a unified format for training:

```json
{
  "problem": "If $x^2 + 2x - 15 = 0$, what are the possible values of $x$?",
  "solution": "We can factor this as $(x+5)(x-3) = 0$. Therefore, $x = -5$ or $x = 3$."
}
```

**Prompt template:**
```
Below is a math problem. Write a solution that appropriately solves the problem.

### Problem:
{problem}

### Solution:
{solution}
```

---

## Planned Datasets (Alexthia v1.0 - January 2026)

### Game Theory Dataset (Tenet)
- **Source:** Custom-generated using Tenet solver
- **Target size:** 10,000 examples
- **Format:** Tenet game definition + Nash equilibrium + proof
- **Coverage:**
  - 2-player strategic games (5,000)
  - 3+ player games (3,000)
  - Cooperative games (1,000)
  - Auction mechanisms (1,000)
- **Purpose:** Teach game-theoretic reasoning and equilibrium concepts

### Formal Verification Dataset (Flux)
- **Source:** Custom-generated using Flux verifier
- **Target size:** 5,000 examples
- **Format:** Theorem + Flux proof + verification steps
- **Coverage:**
  - Algebra (2,000)
  - Calculus (1,500)
  - Number theory (1,000)
  - Logic (500)
- **Purpose:** Teach formal proof construction and verification

### Alexthia v1.0 Total
- **Combined examples:** 22,304
- **Domains:** 5 (competition math, word problems, science, game theory, formal verification)
- **Result:** Full neuro-symbolic AI with verifier-optimizer loop

---

## Loading Datasets

### From Hugging Face (Current Method)

```python
from datasets import load_dataset

# MATH-Hard
math_data = load_dataset("lighteval/MATH-Hard", split="train[:5000]")

# GSM8K
gsm8k_data = load_dataset("openai/gsm8k", "main", split="train[:3000]")

# SciQ
sciq_data = load_dataset("allenai/sciq", split="train[:2000]")
```

### From Local Files (Future Custom Datasets)

```python
# Load local JSON
dataset = load_dataset("json", data_files="data/tenet_games.json")

# Load multiple files
dataset = load_dataset("json", data_files={
    "train": "data/train.json",
    "test": "data/test.json"
})
```

---

## Do Not Commit Large Files

⚠️ **Important:** Large dataset files (.json.gz, .csv.gz, .parquet) are in `.gitignore`.

**Recommended approaches:**
- Load directly from Hugging Face Datasets (current method)
- Use Git LFS for files >100MB
- Host on cloud storage (S3, Azure Blob) and download in code
- For custom datasets: provide generation script, not raw data

---

**Last Updated:** December 26, 2025  
**Current Version:** Alexthia v0.5 (7,304 examples)  
**Next Update:** January 2026 (add Tenet + Flux datasets for v1.0)

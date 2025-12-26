# Alexthia: LLM with Formal Verification

**Reducing hallucination in language models through self-verification and formal reasoning framework.**

![Status](https://img.shields.io/badge/status-in--training-orange)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## Overview

Alexthia is a fine-tuned version of Qwen 2.5 7B (7 billion parameters), demonstrating the **"Verifier-Optimizer" architecture** where:
1. An LLM generates mathematical/logical solutions
2. The LLM questions its own reasoning (Socratic method)
3. Formal verifiers (Tenet+Flux) check correctness
4. The model learns to self-correct through iterative verification

**Key insight:** *Architecture beats scale.* A small model (7B) with verification can outperform larger unverified models (14B+) by rejecting hallucinations before they propagate taking advantage of shorter runtimes.

This is part of the **Axiom Stack**, a complete neuro-symbolic computing platform integrating AI with game theory and formal verification.

## Architecture

```
┌─────────────────────────────────────────────┐
│           Alexthia (Qwen 7B)                │
│         Fine-tuned on reasoning tasks       │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │  Generate Response  │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │     Verifier        │
         │  (Formal checking)  │
         └──────────┬──────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
    VERIFIED              HALLUCINATION
         │                     │
    ✅ Accept           ❌ Reject & Retrain
```

## Quick Start

### 1. Training on Kaggle (Recommended)

```bash
# 1. Sign up for Kaggle: https://www.kaggle.com
# 2. Upload notebooks/training_multi_dataset.ipynb
# 3. Enable GPU (Settings → Accelerator → GPU T4 x2)
# 4. Enable Internet (Settings → Internet → On)
# 5. Run all cells
# 6. Wait 1-2 hours (7,304 examples)
# 7. Download trained LoRA adapters (~100MB)
```

### 2. Local Inference (After Training)

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference
python scripts/inference.py --model-path ./models/alexthia-qwen-7b-lora
```

### 3. Evaluation

```bash
# Run benchmark suite
python scripts/evaluate.py --dataset MATH --num-samples 100
```

## Repository Structure

```
alexthia/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
│
├── notebooks/
│   ├── training.ipynb       # Main training notebook (Kaggle)
│   └── demo.ipynb           # Inference demonstrations
│
├── scripts/
│   ├── inference.py         # Run model inference
│   ├── evaluate.py          # Benchmark evaluation
│   └── export.py            # Export to GGUF/ONNX
│
├── data/
│   └── README.md            # Dataset information
│
├── models/
│   └── .gitkeep             # Model weights go here (not committed)
│
└── docs/
    ├── TRAINING.md          # Training guide
    ├── RESULTS.md           # Benchmark results
    └── INTEGRATION.md       # Flux integration (future)
```

## Features

### Current (v0.5 - In Training)
- Multi-domain training (MATH-Hard + GSM8K + SciQ)
- 7,304 examples across 3 reasoning domains
- QLoRA optimization for consumer GPU (T4/P100)
- Optimized for Kaggle free tier 


### Planned (v1.0 - January 2026)
- Tenet dataset integration (10k game theory examples)
- Flux dataset integration (5k formal verification examples)
- Total 22,304 examples across 5 domains
- Socratic self-questioning mechanism
- Verifier-optimizer loop implementation
- Comprehensive hallucination benchmarks

## Technical Details

**Base Model:** Qwen 2.5 7B (7 billion parameters)  
**Training Method:** QLoRA (Quantized Low-Rank Adaptation)  
**Quantization:** 4-bit (fits in 16GB VRAM)  
**LoRA Rank:** 16, Alpha: 16  

**Training Data (v0.5):**
- MATH-Hard: 2,304 examples (competition-level)
- GSM8K: 3,000 examples (word problems)  
- SciQ: 2,000 examples (science reasoning)
- **Total:** 7,304 examples

**Training Configuration:**
- GPU: Tesla T4 (Kaggle)
- Batch size: 2, Gradient accumulation: 4 (effective: 8)
- Epochs: 3
- Training time: ~6-8 hours
- Optimizer: AdamW 8-bit
- Learning rate: 2e-4 with linear warmup

**Model Size:**
- LoRA adapters: ~100MB
- Full merged model: ~14GB (16-bit)
- Trainable parameters: ~67M (0.95% of total)

## Performance

| Metric | Base Qwen 7B | Alexthia v0.5 (Target) | Expected Improvement |
|--------|--------------|------------------------|----------------------|
| MATH Accuracy | ~75% | ~78-80% | +3-5% |
| MATH-Hard Accuracy | ~60% | ~65-68% | +5-8% |
| GSM8K Accuracy | ~85% | ~88-90% | +3-5% |
| Training Loss | - | ~0.5-0.7 (final) | - |
| Inference Speed | ~45 tok/s | ~45 tok/s | 0% (same) |

*Benchmarks will be updated after training completion (December 26-27, 2025).*

**Key Hypothesis:** With v1.0 (Tenet/Flux verification), expect Alexthia 7B + verifier to match or exceed standalone 14B+ models on reasoning tasks.

## Part of the Axiom Stack

Alexthia is one component of the **Axiom Stack**:

1. **Tenet** - Game theory language and Nash equilibrium solver
2. **Flux** - Math DSL with formal verification
3. **Alexthia** - LLM with verified reasoning (this project)
4. **Axiom OS** - Minimal Linux distribution optimized for mathematical computing

Learn more: [Axiom Stack Documentation](#)

## Use Cases

- **Education:** Teaching formal mathematical reasoning
- **Research:** Studying LLM hallucination reduction
- **Game Theory:** Solving strategic games with verified equilibria
- **Finance:** Risk modeling with formal guarantees
- **AI Safety:** Demonstrating verifiable AI reasoning

## Citation

If you use Alexthia in your research, please cite:

```bibtex
@software{alexthia2025,
  title={Alexthia: LLM with Formal Verification},
  author={Fawaz Ishola},
  year={2025},
  url={https://github.com/fawazishola/alexthia}
}
```

## Contributing

This is currently a solo research project, but feedback is welcome! Open an issue or reach out.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- **Qwen Team** - Base model (Alibaba Cloud)
- **Unsloth** - Fast training library
- **Hugging Face** - Transformers and datasets
- **Kaggle** - Free GPU compute

---

**Author:** Fawaz Ishola  
**Institution:** Carleton University (Aerospace Engineering, Math Minor)  
**Project:** Part of Axiom Stack  
**Status:** In Training (v0.5) → Production (v1.0 - January 2026)

---

<p align="center">
  <i>"Show me the equilibrium, and I'll show you the truth."</i>
</p>

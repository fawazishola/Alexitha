# Alexthia: LLM with Formal Verification

**Reducing hallucination in language models through game-theoretic reasoning and formal verification.**

![Status](https://img.shields.io/badge/status-proof--of--concept-yellow)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## Overview

Alexthia is a fine-tuned version of Qwen 2.5 7B, designed to demonstrate the "verifier-optimizer" architecture where:
1. An LLM generates mathematical/logical reasoning
2. A formal verifier (Flux) checks correctness
3. The model learns to prefer verified outputs

This is part of the **Axiom Stack**, a suite of tools for verified computational reasoning.

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
         │   Flux Verifier     │
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
# 2. Upload notebooks/training.ipynb
# 3. Enable GPU (Settings → Accelerator → GPU T4 x2)
# 4. Run all cells
# 5. Wait 6-8 hours
# 6. Download trained model
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

### Current (v0.1)
- ✅ Fine-tuned Qwen 2.5 7B on mathematical reasoning
- ✅ QLoRA optimization for consumer hardware
- ✅ Trained on 5,000 competition-level math problems
- ✅ Inference ready
- ✅ Exportable model weights

### Planned (v0.2+)
- 🚧 Flux verifier integration
- 🚧 DPO training with verified/hallucinated pairs
- 🚧 Tenet-generated game theory dataset
- 🚧 Nash equilibrium solver integration
- 🚧 Comprehensive hallucination benchmarks

## Technical Details

**Base Model:** Qwen 2.5 7B (7 billion parameters)  
**Training Method:** QLoRA (Quantized Low-Rank Adaptation)  
**Quantization:** 4-bit (fits in 16GB VRAM)  
**LoRA Rank:** 16  
**Training Data:** MATH dataset (5,000 examples)  
**Training Time:** ~6-8 hours on P100 GPU  
**Trainable Parameters:** ~67M (0.95% of total)

## Performance

| Metric | Base Qwen 7B | Alexthia v0.1 | Improvement |
|--------|--------------|---------------|-------------|
| MATH Accuracy | TBD | TBD | TBD% |
| Training Loss | - | 0.XX | - |
| Inference Speed | ~X tok/s | ~X tok/s | 0% |

*Benchmarks will be updated after training completion.*

## Part of the Axiom Stack

Alexthia is one component of the **Axiom Stack**:

1. **Tenet** - Game theory language and Nash equilibrium solver
2. **Flux** - Math DSL with formal verification
3. **Alexthia** - LLM with verified reasoning (this project)
4. **Axiom OS** - Minimal Linux distribution optimized for mathematical computing

Learn more: [Axiom Stack Documentation](#)

## Use Cases

- 🎓 **Education:** Teaching formal mathematical reasoning
- 🔬 **Research:** Studying LLM hallucination reduction
- 🎯 **Game Theory:** Solving strategic games with verified equilibria
- 💼 **Finance:** Risk modeling with formal guarantees
- 🤖 **AI Safety:** Demonstrating verifiable AI reasoning

## Citation

If you use Alexthia in your research, please cite:

```bibtex
@software{alexthia2025,
  title={Alexthia: LLM with Formal Verification},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/alexthia}
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

**Author:** [Your Name]  
**Contact:** [Your Email]  
**Project:** Part of Atlas (YC W26 application)  
**Status:** Proof of Concept (v0.1)

---

<p align="center">
  <i>"Show me the equilibrium, and I'll show you the truth."</i>
</p>

# Applying CoVT to Korean Cultural Image Captioning with Knowledge Injection

> Independent experiments by an undergraduate researcher, built on top of [CoVT (Chain-of-Visual-Thought)](https://arxiv.org/abs/2511.19418) by Qin et al. (UC Berkeley, 2025).

---

## Overview

This repository documents two sets of experiments exploring the capabilities and limitations of [CoVT](https://arxiv.org/abs/2511.19418) — a visual reasoning framework for VLMs — when applied to **Korean cultural heritage image captioning**, a domain characterized by sparse training coverage and high sensitivity to cultural hallucination.

The experiments proceed in two stages:

1. **Experiment 1**: Validate whether CoVT's visual reasoning advantage transfers to a culturally specific, underrepresented domain.
2. **Experiment 2**: Design and evaluate an external knowledge injection framework to address the cultural knowledge gap that visual reasoning alone cannot fill.

The final output is a curated training dataset of **24,923 Korean cultural images** with model-selected captions, intended for subsequent knowledge-injection fine-tuning.

---

## Research Motivation

### Problem: Three Failure Modes in VLMs for Domain-Specific Captioning

Applying general-purpose VLMs to Korean cultural imagery surfaces three distinct failure modes:

| Failure Mode | Description | Example |
|---|---|---|
| **Object Hallucination** | Generating entities absent from the image | Labeling a Korean kite-flying scene as "Chinese Dragon Boat Festival" |
| **Language Prior Dominance** | Captions driven by pre-trained text patterns rather than visual evidence | Describing any traditional gathering as "symbolizing unity and harmony" |
| **Subjective Injection** | Inserting unverifiable symbolic or emotional content | "The ritual carries deep ancestral meaning" (image shows men in suits) |

These issues are amplified in underrepresented domains where pre-trained models lack localized knowledge.

### Why CoVT? And Why Is It Insufficient Alone?

CoVT introduces continuous visual tokens (segmentation, depth, DINO, edge) into VLM reasoning chains, improving perceptual grounding on general benchmarks. This raised a natural question: does better visual grounding reduce cultural hallucination?

**Short answer: partially, but not sufficiently.**

- CoVT reduces hallucination driven by visual ambiguity
- CoVT *cannot* supply missing semantic knowledge — it cannot identify an unfamiliar cultural artifact by name, even when it describes its shape correctly
- This gap motivated designing a knowledge injection mechanism that complements CoVT's perceptual strengths

---

## Method

### Experiment 1 — Baseline Validation

**Setup**: 50 Korean cultural images across 32 categories. Blind pairwise comparison (CoVT-7B-seg_depth_dino vs. Qwen2.5-VL-7B baseline) using GPT-4.1-mini as judge with five evaluation criteria (visual grounding, hallucination absence, neutrality, detail, generalization).

**Key design decision — prompt sensitivity**:

| Prompt | Strategy | CoVT Win Rate (Korean) |
|---|---|---|
| P0 | No category info | ~0.50 |
| P1 | Category name + description | 0.28 (**baseline wins 0.72**) |
| P2 | Visual element extraction | — |
| P3 | Structural verification (training-format-aligned) | **0.56** |

Prompt P1 degraded CoVT performance significantly, while P3 — designed to match CoVT's internal training format — recovered it. This established that **prompt design must account for model-internal reasoning mechanisms**, not just task objectives.

### Experiment 2A — Extending to Cultural Context (without KB)

Extended prompt space to P0–P6 (adding culturally framed instructions). P6 ("Describe the visual appearance and cultural context...") achieved CoVT win-rate of **0.60**.

Analysis of 30 CoVT-winning captions:

| Pattern | Frequency |
|---|---|
| Suppression of baseline hallucinations | 93.3% |
| More visually grounded details | 60.0% |
| Correct use of cultural terminology within visual bounds | 50.0% |

Despite these wins, qualitative analysis confirmed persistent **cultural hallucination** in both models — unverifiable claims generated from textual priors rather than image content. This motivated external knowledge injection.

### Experiment 2B — Knowledge-Injected Captioning

#### Knowledge Base Design

- 32 categories mapped to dictionary-style, **visually grounded definitions** (what is visible in an image, not encyclopedic background)
- Integrated via RAG-style prompting — no fine-tuning required
- Core design principle: treat KB as a **naming guide**, not a content source

**Negative constraint**: prompts explicitly prohibited the model from including KB content not directly visible in the image.

#### Systematic Prompt Exploration (P7–P16)

Three iteration rounds across 10 prompt variants:

| Round | Prompts | Focus |
|---|---|---|
| v2 | P7–P13 | KB presentation format: parenthetical, reference block, naming guide, conditional application |
| v3 | P14–P16 | Combining best v2 elements: image-first + KB isolation, visual detail density, double grounding |
| Final | P15 | Large-scale validation on 640 images (20 per category) |

**Final prompt (P15)**:
```
Reference: {category} — {definition}.
Describe what you actually see in this image in exactly one sentence,
noting colors, shapes, and spatial arrangement.
Use the correct Korean cultural term if visually supported.
Do not include any reference details that are not visible.
```

P15 was selected after identifying a key failure in earlier prompts: CoVT captions were 9–19 characters shorter than baseline captions, with reduced visual detail. P15's explicit instructions ("colors, shapes, and spatial arrangement") addressed this gap while preserving grounding constraints.

---

## Key Results and Failure Mode Analysis

### Large-Scale Evaluation (640 images, P15)

Domain-level breakdown revealed systematic CoVT underperformance in three categories:

| Domain | CoVT Win Rate | Root Cause |
|---|---|---|
| History | **0.359** | Monuments, statues, archival photos — minimal perceptual signal from visual tokens |
| Architecture | 0.483 | Single-subject buildings — limited spatial variation |
| Folk | 0.473 | Costume/movement interpretation requires cultural semantics, not spatial perception |

### Identified Failure Modes

**1. Domain mismatch**: CoVT's visual tokens (Seg/Depth/DINO/Edge) are optimized for spatial tasks (counting, depth ordering). In knowledge-intensive categories, they add no marginal value.

**2. Catastrophic forgetting**: CoVT was fine-tuned on vision-centric data (LLaVA-OneVision, TallyQA, ADE20K-Depth) with no Korean cultural imagery. Fine-tuning partially diluted Qwen2.5-VL's pre-trained cultural knowledge, causing systematic underperformance in knowledge-intensive domains.

**3. Caption convergence**: For visually homogeneous categories (e.g., Haenggung palace images), CoVT produced near-identical captions with >0.9 SequenceMatcher similarity across 62 image pairs. When visual tokens fail to encode domain-specific distinctions, the `<think>...<visual tokens>...</think>` reasoning chain becomes a passthrough, and generation collapses toward the fine-tuning distribution mean.

### Solution: Winner-Model Selection with Deduplication (Strategy C)

To construct a high-quality, diverse training dataset despite these failure modes:

1. **Per-image GPT-based selection**: Use GPT-4.1 pairwise evaluation to select the better caption per image (CoVT or baseline). Automatically adapts to domain-level performance variation without manual rules.
2. **SequenceMatcher deduplication**: Remove captions with similarity > 0.85 within each category; retain the more specific caption (by word count).

**Result**: 24,923 training images with quality-selected, deduplicated captions across 32 Korean cultural categories.

---

## Technical Stack

| Component | Technology |
|---|---|
| Base VLM | Qwen2.5-VL-7B-Instruct |
| Visual Reasoning | CoVT (Chain-of-Visual-Thought) — Seg/Depth/DINO tokens |
| Knowledge Integration | Dictionary-style KB with RAG-style prompting (no fine-tuning) |
| Evaluation | GPT-4.1 / GPT-4.1-mini pairwise judging via OpenAI API |
| Deduplication | Python `SequenceMatcher` (threshold: 0.85) |
| Framework | PyTorch, Hugging Face Transformers |

---

## Repository Structure

```
gradio/
├── gen_scripts/      # Caption generation scripts (CoVT + baseline)
└── eval_scripts/     # GPT-based pairwise evaluation scripts
train/                # CoVT training code (forked from original repo)
VLMEvalKit/           # Evaluation framework (forked)
docs/                 # Experiment logs and analysis
```

---

## What This Work Is (and Is Not)

This is an independent undergraduate research project — not a paper submission or formal study. The contribution is primarily:

- Systematic empirical investigation of where and why CoVT generalizes (or fails to) in a culturally underrepresented domain
- A structured knowledge injection design that separates visual grounding from semantic naming
- Concrete failure mode analysis (domain mismatch, catastrophic forgetting, caption convergence) with traceable causes
- A 24,923-image training dataset with principled construction methodology

The original CoVT framework was developed by Qin et al. (2025) at UC Berkeley. This work is an application and extension study built on their publicly released code and models.

---

## Original CoVT Paper

```bibtex
@article{qin2025chain,
  title={Chain-of-Visual-Thought: Teaching VLMs to See and Think Better with Continuous Visual Tokens},
  author={Qin, Yiming and Wei, Bomin and Ge, Jiaxin and Kallidromitis, Konstantinos and Fu, Stephanie and Darrell, Trevor and Wang, Xudong},
  journal={arXiv preprint arXiv:2511.19418},
  year={2025}
}
```

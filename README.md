# Build an LLM (from Scratch)

This repository documents the foundational research and implementation efforts in understanding and constructing Large Language Models (LLMs) from fundamental principles. The primary objective is to bypass high-level abstractions in favor of rigorous, mathematically grounded implementations of state-of-the-art transformer architectures.

## Ongoing Research

### Attention Calibration Distillation for Demonstration-Free Text Reasoning
This directory contains the codebase and empirical findings for my current, active research investigating attention calibration distillation methodologies. The work aims to enhance the inherent text reasoning capabilities of language models in a zero-shot, demonstration-free context.

## Repository Structure

### Core Architectures & Implementations
Ground-up implementations of established LLM architectures to empirically analyze their structural and functional paradigms:
- **`GPT_2_from_scratch_Sebastiane_Build_an_llm_from_scratch/`**: Implementation of the GPT-2 architecture, building upon Sebastian Raschka's foundational literature.
- **`gemma_from_scratch_in_python/`**: A native Python implementation of the Gemma model architecture.
- **`qwen_3_from_scratch_in_pytorch/`**: PyTorch-based construction of the Qwen 3 model architecture.

### Core Mechanisms & Primitives
Isolated implementations of critical transformer sub-components to facilitate deeper mechanistic interpretability and study:
- **[`SELF_ATTENTION_FROM_SCRATCH.PY`](SELF_ATTENTION_FROM_SCRATCH.PY)**: A foundational implementation of the scaled dot-product Self-Attention mechanism.
- **[`ROPE.PY`](ROPE.PY)**: Implementation of Rotary Positional Embeddings (RoPE), a standard methodology for positional encoding in modern architectures such as LLaMA, Gemma, and Qwen.

## Technical Environment
- **Python**
- **PyTorch**

## Project Objectives
1. Achieve a rigorous, mathematically sound comprehension of contemporary Transformer architectures.
2. Conduct comparative structural analyses between foundational models (GPT-2, Gemma, Qwen).
3. Implement and evaluate modern advancements in attention mechanisms and positional embeddings.
4. Advance ongoing research in attention calibration and demonstration-free reasoning paradigms.

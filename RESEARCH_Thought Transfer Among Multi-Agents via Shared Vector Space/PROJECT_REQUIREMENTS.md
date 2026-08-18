# Project Requirements

## Purpose

This document defines the engineering requirements, coding standards, architecture, milestones, and implementation roadmap for the THOUGHTCOMM reimplementation.

The objective is **not** merely to produce working code, but to create a clean, modular, maintainable, and research-friendly codebase that is easy to extend and understand.

This document should be treated as the primary engineering specification for the project.

---

# Development Philosophy

The project should prioritize:

* Clean architecture
* Readability
* Modularity
* Type safety
* Documentation
* Reproducibility
* Unit testing
* Research extensibility

Code should be written as though it may eventually become an open-source research repository.

---

# Repository Structure

```text
thoughtcomm/

│

├── thoughtcomm/
│   ├── __init__.py
│   ├── autoencoder.py
│   ├── adapter.py
│   ├── router.py
│   ├── losses.py
│   ├── extractor.py
│   ├── prefixes.py
│   ├── utils.py
│   ├── config.py
│   └── visualization.py
│
├── scripts/
│   ├── extract_hidden_states.py
│   ├── train_autoencoder.py
│   ├── train_adapter.py
│   ├── evaluate_autoencoder.py
│   └── run_two_agent_demo.py
│
├── experiments/
│   ├── exp01_hidden_state_extraction.py
│   ├── exp02_sparse_autoencoder.py
│   ├── exp03_latent_visualization.py
│   ├── exp04_adapter.py
│   ├── exp05_two_agent_demo.py
│   └── exp06_ablation.py
│
├── notebooks/
│   ├── 01_hidden_states.ipynb
│   ├── 02_autoencoder.ipynb
│   ├── 03_visualization.ipynb
│   └── 04_demo.ipynb
│
├── configs/
│
├── tests/
│
├── figures/
│
├── checkpoints/
│
├── requirements.txt
│
├── README.md
│
└── LICENSE
```

---

# Coding Standards

All Python code should:

* Use Python 3.11+
* Follow PEP 8
* Include type hints
* Include comprehensive docstrings
* Avoid global mutable state
* Prefer composition over inheritance
* Keep functions small and focused
* Use descriptive variable names

Avoid:

* Hardcoded paths
* Notebook-only logic
* Duplicate code
* Large monolithic classes

---

# Configuration

All configurable values should live in configuration objects or configuration files.

Examples:

* model name
* latent dimension
* batch size
* learning rate
* sparsity coefficient
* prefix length
* hidden layer sizes

Never hardcode experiment-specific values inside model implementations.

---

# Module Requirements

## extractor.py

Responsible for:

* Loading Hugging Face models
* Tokenization
* Running inference
* Extracting hidden states
* Returning the final hidden representation

Should expose a simple interface such as:

```python
extract_hidden_state(prompt)
```

No training logic belongs here.

---

## autoencoder.py

Contains:

* SparseAutoencoder class
* Encoder
* Decoder
* Forward pass
* Latent extraction
* Reconstruction

Should support configurable:

* input dimension
* latent dimension
* hidden layers
* activation functions

---

## losses.py

Responsible for all loss functions.

Should include:

* Reconstruction loss
* Sparsity loss
* Combined loss

Losses should remain independent of model implementations whenever possible.

---

## adapter.py

Responsible for:

* Mapping latent vectors into transformer embedding space
* Producing prefix embeddings

The adapter should remain lightweight.

---

## router.py

Responsible for future communication routing.

Initially it may contain placeholder implementations.

Eventually it should support:

* shared latent routing
* private latent routing
* masking
* communication policies

---

## visualization.py

Responsible for:

* PCA
* t-SNE
* UMAP (optional)
* Activation plots
* Latent neuron statistics
* Reconstruction plots

Visualization code should never be mixed into training loops.

---

# Experiment Requirements

Every experiment should live inside its own script.

Each experiment must have:

* Objective
* Configuration
* Logging
* Saved outputs
* Reproducible execution

Experiments should never modify library code.

---

# Milestones

## Milestone 1

Extract hidden states from a small language model.

Deliverables:

* hidden state extractor
* validation notebook
* unit tests

---

## Milestone 2

Train a standard autoencoder.

Deliverables:

* reconstruction loss
* reconstruction visualization

---

## Milestone 3

Train a sparse autoencoder.

Deliverables:

* sparsity regularization
* latent visualization
* activation statistics

---

## Milestone 4

Implement adapter network.

Deliverables:

* adapter model
* prefix embeddings
* embedding validation

---

## Milestone 5

Prototype two-agent communication.

Deliverables:

* two frozen agents
* latent exchange
* qualitative evaluation

---

## Milestone 6

Perform ablation studies.

Possible studies:

* latent dimension
* sparsity coefficient
* prefix length
* hidden layer depth

---

# Unit Testing

Every module should include tests where appropriate.

Examples:

Autoencoder:

* output shape
* latent shape
* reconstruction shape

Loss:

* correct scalar output
* gradient computation

Extractor:

* hidden state dimensions
* deterministic output

Adapter:

* embedding dimensions
* prefix length

---

# Logging

Training should log:

* epoch
* loss
* reconstruction loss
* sparsity loss
* learning rate

Support:

* TensorBoard
* CSV logging

Weights & Biases support may be added later.

---

# Checkpointing

Training scripts should save:

* model weights
* optimizer state
* configuration
* training epoch

Checkpoints should be resumable.

---

# Visualization Requirements

The project should generate figures showing:

* loss curves
* reconstruction quality
* latent activation histograms
* PCA projections
* t-SNE projections
* sparsity statistics

Saved figures should be publication quality whenever possible.

---

# Performance Goals

The project should remain executable on Google Colab using a T4 GPU.

Models should remain frozen whenever possible.

Training should primarily involve:

* sparse autoencoder
* adapter

The implementation should avoid unnecessary GPU memory consumption.

---

# Documentation Requirements

Every public class should include:

* purpose
* arguments
* return values
* usage example

Every script should include:

* objective
* expected inputs
* expected outputs

---

# Future Extensions

The architecture should make it easy to replace components.

Possible future replacements include:

* Variational Autoencoder
* Top-K Sparse Autoencoder
* Dictionary Learning
* LoRA Adapter
* Different routing algorithms
* Different sparse penalties
* Different visualization techniques

No module should be tightly coupled to a specific implementation.

---

# Success Criteria

This project will be considered successful if:

* Every module is independently understandable.
* Every experiment is reproducible.
* Components are interchangeable.
* The codebase is well documented.
* The architecture can support future research extensions.
* A new contributor can understand and extend the project with minimal onboarding.

The long-term objective is to produce a repository that serves as both a learning resource and a solid foundation for future research into latent communication between language models.

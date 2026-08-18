# Project Details

# Project Name

THOUGHTCOMM: Reimplementation and Research Exploration

---

# Project Objective

This project is **NOT** intended to simply reproduce the results of the THOUGHTCOMM paper.

Instead, the goal is to **understand, reconstruct, validate, and extend** the ideas proposed in the paper from first principles.

Every component should be implemented in a modular and educational way.

The implementation should prioritize:

* correctness
* readability
* modularity
* reproducibility
* research friendliness

over reproducing benchmark numbers.

---

# Background

Current multi-agent LLM systems communicate through natural language.

Example:

Agent A

↓

"I think the answer is ..."

↓

Agent B

↓

Reads text

↓

Continues reasoning

The paper argues that this communication protocol is fundamentally inefficient because the language is generated **after** reasoning has already happened.

Instead of exchanging natural language, the authors propose exchanging **latent internal representations**.

---

# Core Hypothesis

The paper assumes that an LLM's hidden state is generated from a set of latent reasoning variables.

Instead of

Prompt

↓

Hidden State

the paper proposes

Prompt

↓

Latent Variables (Thoughts)

↓

Hidden State

The hidden state is observable.

The latent variables are not.

These latent variables are referred to as "Thoughts" throughout the paper.

**Important:**

This project should treat these as **latent representations** rather than assuming they literally correspond to human thoughts.

The implementation should remain scientifically conservative.

---

# Hidden State

The implementation will extract the hidden representation corresponding to the **last generated token**.

Reason:

By the time the final token is produced, the hidden representation already incorporates information from all previous tokens.

This hidden state is used as the observable representation **H**.

---

# Latent Representation

The latent representation is denoted as **Ẑ** and is produced by the encoder of a sparse autoencoder.

Pipeline:

Hidden State

↓

Encoder

↓

Ẑ

↓

Decoder

↓

Reconstructed Hidden State

The encoder produces **Ẑ**.

The decoder exists only to force **Ẑ** to preserve useful information through reconstruction.

---

# Why an Autoencoder?

The hidden state is a high-dimensional mixed representation.

Example:

4096-dimensional vector

↓

Contains:

* arithmetic
* planning
* syntax
* memory
* reasoning
* confidence
* context

all mixed together.

The autoencoder attempts to learn a compressed representation capable of reconstructing the hidden state.

This compressed representation is interpreted as the latent communication space.

---

# Sparsity

This is the most important concept in the project.

Without sparsity:

every latent variable influences almost every hidden-state feature.

Example:

z1 → everything

z2 → everything

z3 → everything

This produces highly entangled representations.

With sparsity:

each latent variable is encouraged to explain only a small subset of the hidden-state features.

Example:

z1 → arithmetic

z2 → planning

z3 → geometry

This **does not prove** these variables are literal thoughts.

Instead,

sparsity encourages specialization and disentanglement.

Throughout the implementation these should be described as:

**specialized latent features**

rather than guaranteed semantic thoughts.

---

# Autoencoder Objective

The encoder compresses.

The decoder reconstructs.

The training objective should include:

* Reconstruction loss
* Sparsity regularization

The purpose is not compression alone.

The purpose is to produce structured latent variables that are easier to separate and communicate.

---

# Adapter

The transformer cannot directly consume **Ẑ**.

Therefore another lightweight neural network is introduced.

Pipeline:

Ẑ

↓

Adapter MLP

↓

Prefix Embeddings

↓

Frozen Transformer

The adapter converts the latent representation into vectors compatible with the transformer's embedding space.

---

# Frozen LLM

The language model itself remains frozen.

Only the following modules should be trainable:

* Sparse Autoencoder
* Adapter
* (Later) Routing components

This dramatically reduces computational requirements and makes the project feasible on limited hardware.

---

# Communication

Traditional systems:

Text

↓

Agent

THOUGHTCOMM:

Hidden State

↓

Latent Representation

↓

Adapter

↓

Prefix Embeddings

↓

Agent

Natural language is replaced with latent representations as the communication medium.

---

# Research Philosophy

This project should clearly distinguish between:

## Hypothesis

The latent representation corresponds to underlying reasoning variables.

## Implementation

A sparse autoencoder approximates these latent variables from observable hidden states.

The implementation should **never** claim that literal thoughts have been recovered.

---

# Project Scope

### Primary Goal

Understand every component of THOUGHTCOMM and if possible extend to a new research direction based on it.

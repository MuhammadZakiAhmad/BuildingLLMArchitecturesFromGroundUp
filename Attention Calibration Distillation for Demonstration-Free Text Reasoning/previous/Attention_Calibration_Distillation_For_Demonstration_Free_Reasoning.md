A Simplified Research Plan: Attention Calibration Distillation for Demonstration-Free Text Reasoning
1. Motivation

Large Language Models (LLMs) often perform much better when provided with few-shot demonstrations. These demonstrations help the model understand the task by showing examples before the actual question.

For example,

Example 1
Q: 2 + 3 = ?
A: 5

Example 2
Q: 4 + 7 = ?
A: 11

Now answer:

Q: 6 + 8 = ?

The problem is that demonstrations increase the context length, making inference slower and more expensive.

Our objective is to build a student model that can imitate the behaviour of a few-shot teacher without receiving the demonstrations.

Instead of modifying the model weights, we will modify where the transformer attends.

2. Attention Calibration

A transformer computes attention scores using

S=
d
	​

QK
T
	​


where

Q (Query) represents what each token is searching for.
K (Key) represents what information each token provides.
S is the attention logit matrix before Softmax.

Normally,

Q,K
   │
   ▼
QKᵀ
   │
   ▼
Softmax
   │
   ▼
Attention

Our idea is extremely simple.

Instead of using only

S

we compute

S
′
=S+Δ

where

Δ

is a learnable attention correction.

Conceptually,

Δ represents how the attention scores should change so that the student behaves like a teacher that has seen demonstrations.

3. Low-Rank Attention Bias

Learning an entire matrix

Δ

is impractical because its size is

Sequence Length × Sequence Length

For long sequences this becomes millions of values.

Instead, we construct Δ indirectly.

First we project the Queries.

A=QU
q
	​


where

Q is the original Query matrix.
Uq is a small trainable projection matrix.

Intuitively,

A is simply a compressed representation of the Queries.

Similarly,

B=KU
k
	​


where

K is the original Key matrix.
Uk is another trainable projection matrix.

Again,

B is a compressed representation of the Keys.

Finally,

Δ=AB
T

This is simply an outer product between the compressed Query representation and compressed Key representation.

The beautiful property is that

if the question changes,
Q changes,
therefore A changes,
therefore Δ changes.

Likewise,

if the input changes,
K changes,
therefore B changes,
therefore Δ changes.

Hence,

Δ is generated dynamically for every input rather than being a fixed matrix.

4. Query-Adaptive Gate

Not every question benefits equally from attention correction.

Instead of always adding the complete Δ, we learn a gate

g=σ(MLP(Q))

where

0 ≤ g ≤ 1

The modified attention becomes

S
′
=S+gΔ

Intuitively,

g = 0.95

→ Apply almost all of Δ

whereas

g = 0.10

→ Apply only a small fraction of Δ

The gate therefore learns

When should the attention be corrected?

5. Hidden State Distillation

After attention is computed,

Attention

↓

Context Vector (Z)

↓

Residual Connection

↓

LayerNorm

↓

Feed Forward Network

↓

Residual Connection

↓

LayerNorm

↓

Hidden State

The hidden state is simply the output of an entire transformer block.

Suppose we have two models.

Teacher

Receives

Few-shot demonstrations

+

Question

and produces

Hidden State
Student

Receives only

Question

and also produces

Hidden State

The training objective is not to supervise Δ or the gate directly.

Instead, we compare

Teacher Hidden State

vs

Student Hidden State

The loss is computed between these hidden representations.

During backpropagation, gradients naturally flow through

Hidden State

↓

Attention

↓

gΔ

↓

Δ

↓

Uq, Uk

↓

Gate Parameters

Thus,

Uq, Uk, and the gate parameters gradually learn how to modify the student's attention so that its hidden states resemble those of the teacher.

Proposed Research Project

The complete Hyper-ICL framework is multimodal and requires substantial computational resources. Instead, we propose a lightweight text-only implementation that can be trained on a free Google Colab GPU.

Phase 1 — Teacher-Student Framework

We begin with a pretrained GPT-2 Small (124M) model.

The model is duplicated into

Teacher (Frozen)
Student (Frozen initially)

The teacher receives

4-shot prompt

while the student receives

0-shot prompt

Our first objective is simply to verify that the teacher and student produce different hidden states.

Phase 2 — Attention Calibration

We introduce the low-rank attention adapter.

The student now computes

S
′
=S+Δ

where

Δ=AB
T

Only the projection matrices

U
q
	​

,U
k
	​


are trainable.

The GPT-2 backbone remains completely frozen.

The objective is to determine whether modifying the attention logits makes the student's hidden states more similar to those of the teacher.

Phase 3 — Query-Adaptive Gating

After validating the attention calibration module, we introduce the gate

S
′
=S+gΔ

The gate learns whether a particular input requires strong or weak attention correction.

Performance will be compared against the previous phase to determine whether adaptive calibration provides additional benefits.

Phase 4 — Chain-of-Thought Distillation

If the attention calibration framework successfully transfers the benefits of few-shot prompting, we will extend the same architecture to Chain-of-Thought reasoning.

The teacher will receive reasoning demonstrations, while the student will observe only the original question.

The objective is to investigate whether the student can learn to internally reproduce the reasoning behaviour induced by Chain-of-Thought examples without explicitly seeing those demonstrations.
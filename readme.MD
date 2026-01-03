Decoder-Only Transformer Language Model (From Scratch)

A minimal GPT-style decoder-only Transformer language model implemented from scratch in PyTorch, built to deeply understand how modern LLMs work at the systems and algorithmic level.

This project focuses on fundamentals rather than scale: attention mechanics, tokenization, training stability, and efficient inference using KV caching.

🚀 Project Motivation

Most LLM tutorials hide critical details behind frameworks.
This project was built to answer questions like:

How does attention actually scale with sequence length?

Why does naive generation collapse into repetition?

What really changes when you add KV cache?

Why do models become memory-bound before compute-bound?

The goal was learning-by-building, not benchmarking against large models.

🧩 Model Overview

Architecture: GPT-style, Decoder-only Transformer

Layers: 6 Transformer blocks

Attention: Multi-Head Causal Self-Attention

Embedding Dimension: 128

Heads: 4

Context Window: 128 tokens

Normalization: Pre-LayerNorm

Weight Tying: Token embedding ↔ output projection

Inference: Autoregressive decoding with KV cache

🔤 Tokenization

Tokenizer: SentencePiece (BPE)

Vocab Size: 3,000

Special Tokens: BOS, EOS, PAD, UNK

Trained directly on the dataset text

📚 Dataset

Source: Wikipedia / custom text corpus (wiki.train.tokens)

Train / Validation split: 90% / 10%

Objective: Next-token prediction (causal LM)

🏋️ Training Details

Framework: PyTorch

Loss: Cross-Entropy Loss

Optimizer: AdamW

Learning Rate: 3e-4

Weight Decay: 0.1

Batching: Random contiguous token blocks

Training Length: ~1k steps (educational scale)

Metrics: Loss, Perplexity

⚡ Key Features Implemented
✅ Transformer from Scratch

Manual implementation of:

QKV projections

Causal masking

Multi-head attention

Residual connections

Feed-forward networks

✅ KV Cache for Fast Inference

Cached keys and values across generation steps

Reduced attention complexity during decoding

Learned why incorrect positional handling breaks generation

✅ Repetition Mitigation

Sliding-window repetition penalty

Token-level logit penalties

Analysis of degeneration vs model capacity

✅ Debugged Real LLM Failure Modes

Infinite repetition loops

Token collapse

Positional embedding overflow

Incorrect KV cache concatenation

📈 Sample Generation
Input: "A human"

Output:
"A human language model processes text by predicting the next token
based on context, representations, and learned statistical structure..."


(Outputs improve with longer training and larger models.)

🧠 What I Learned (Core Takeaways)

KV cache changes memory behavior, not model quality

Small models repeat because of capacity + data limitations

Attention is often memory-bound, not compute-bound

Tokenization quality matters more than expected

Efficient inference requires architectural awareness

🔬 Limitations (Intentionally Acknowledged)

Small parameter count (educational model)

No FlashAttention / fused kernels

No large-scale distributed training

Not instruction-tuned

This project is a foundation, not a product model.

🧪 Future Work

Structured attention pruning experiments

Activation / KV sparsity

Longer context via memory-efficient attention

Quantization-aware inference

Profiling FLOPs vs memory bandwidth

🛠 Tech Stack

Python

PyTorch

SentencePiece

NumPy

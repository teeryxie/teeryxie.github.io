# Correct While Verifying: Intermediate Target Predictions for Speculative Draft Correction

Speculative decoding accelerates large language model inference by letting a small draft model propose tokens and asking the target model to verify them in parallel. Its rejection handling, however, remains reactive: the system learns that a token is wrong only after final verification, then discards the suffix following the first rejection.

**Correct While Verifying** asks whether the target model can repair a draft before verification finishes. The key observation is that intermediate target-model states already contain useful predictions about the correct continuation, even before final logits are available.

## Core method

- A trainable decoder layer maps intermediate target states to preliminary token distributions.
- Top candidate tokens seed correction branches while final target verification continues from cached intermediate states.
- A frozen draft model expands those branches in parallel under a custom attention mask.
- Final target verification evaluates the original draft and corrected branches, then commits the longest valid continuation under the target model's verification rule.

## Why it matters

The method moves speculative decoding from reactive rejection handling toward proactive correction. Across Llama, Qwen2.5, and Vicuna target-draft pairs, the submission reports consistent improvements over PEARL and up to **4.53x speedup** over autoregressive decoding.

## Submission

- AAAI 2027 Conference Submission
- Submitted on July 20, 2026; modified on August 9, 2026
- Authors: Haocheng Sun, Ruilin Wang, Yuexiao Ma, Yuxin Zhang, Tianyu Xie, Jianxin Lin, Zhixiang Ren, Xiawu Zheng
- Primary topic: Efficient, Edge, Green & Hardware-aware ML

The previous FOLD title and ICLR 2026 submission status are obsolete. This page keeps the existing URL so earlier links remain valid.

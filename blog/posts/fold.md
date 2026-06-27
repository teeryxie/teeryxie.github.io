# FOLD: Fast Correct Speculative Decoding

Large language model inference is often limited by decoding latency. Speculative decoding is attractive because it can accelerate generation while preserving correctness when verification is handled carefully.

**FOLD** studies fast correct speculative decoding through online learning and drafting. The important part is not just speed, but the combination of faster generation with an explicit correctness path.

## Research angle

- Draft tokens quickly with an auxiliary process.
- Verify outputs so acceleration does not silently change model behavior.
- Use online learning to adapt the drafting process.
- Treat efficient inference as a system problem, not only a model architecture problem.

## Link

- [OpenReview](https://openreview.net/forum?id=QNEsvRqub6)

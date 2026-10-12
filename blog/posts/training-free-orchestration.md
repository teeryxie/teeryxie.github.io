# Training-Free Multimodal LLM Orchestration

Modern omni-modal assistants should not be rebuilt from scratch every time a new vision, audio, speech, or language expert appears. This work studies a more modular path: coordinate existing modality experts through an LLM-based orchestration layer, then make the system interactive through routing, memory, and streaming control.

The key idea is **training-free composition**. Instead of paying the cost of end-to-end multimodal retraining, the system treats specialized models as callable experts and lets a language model decide when and how to use them. The design allows experts to be replaced independently and makes routing decisions available for inspection.

## Why it matters

- Multimodal systems often need fast iteration across speech, image, video, and text components.
- Training-free orchestration lowers the barrier for building new omni-modal workflows.
- Explicit routing and memory make system behavior more inspectable than a monolithic model.
- Streaming and interruption handling support evaluation in interactive settings.

## Links

- [OpenReview](https://openreview.net/forum?id=V2fTGbSD7Q)
- [arXiv](https://arxiv.org/abs/2508.10016)
- [Code](https://github.com/MAC-AutoML/Trainingfree-LLM-Orchestration)

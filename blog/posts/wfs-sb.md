# WFS-SB: Frame Selection for Long-Video Understanding

Long-video understanding is not only a context-length problem. If a system feeds too many redundant frames into a multimodal model, inference becomes expensive and important transitions can be diluted by noise.

**WFS-SB** approaches the problem through semantic boundaries. The method uses wavelet-based signals to identify meaningful changes in video structure, then selects frames that preserve the reasoning-relevant parts of the video.

## Why this direction is useful

- It reduces redundant visual input before expensive LVLM reasoning.
- It focuses the model on event transitions and semantic boundaries.
- It makes long-video inference more efficient without relying only on larger context windows.
- It connects naturally with event-anchored frame selection work.

## Links

- [arXiv](https://arxiv.org/abs/2603.00512)
- [Code](https://github.com/MAC-AutoML/WFS-SB)

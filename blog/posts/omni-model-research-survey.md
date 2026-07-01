# From GPT-4o to Omni Models: 从实时多模态交互到通用智能体的研究路线

如果要分享 omni model 的研究过程，我建议不要把它讲成“多模态模型又多接了几个输入”。真正的变化是：模型从“能看图、能听音频、能读文本”的能力拼接，走向“在同一个交互过程中看、听、说、想、行动”的系统形态。GPT-4o 之所以成为一个分界点，不只是因为它把 audio、vision、text 放在一起，而是因为它把实时性、自然语音、视觉理解和对话推理放在一个体验里展示出来。此后，开源社区和学术界开始围绕几个问题快速展开：如何做端到端 speech-to-speech？如何让音频和视频在时间上对齐？如何处理打断、重叠语音和连续视频流？如何评测模型是否真的理解多模态互动，而不是只会在单模态 benchmark 上刷分？

这篇文章按 2024 年 5 月 GPT-4o 发布之后的脉络梳理 omni model 的研究：它是什么，为什么会出现，中间经历了哪些技术变化，目前有哪些研究方向，未来可以怎么研究，以及如果做一次分享应该重点讲哪些论文。

![Omni timeline](/images/blog/omni-timeline.svg "图 1：从 GPT-4o 到开源 omni model、实时 API、Qwen2.5/3.5-Omni 和 omni benchmark 的时间线。")

## 什么是 omni model

最窄的定义是：omni model 是能同时处理多种输入模态，并能以多种输出模态回应的统一模型。输入通常包括 text、image、video、audio；输出至少包括 text 和 speech，进一步可能包括 image、audio、tool action 或 embodied action。GPT-4o System Card 对 GPT-4o 的描述很典型：它是 autoregressive omni model，接受 text、audio、image、video 的任意组合输入，并生成 text、audio、image 的任意组合输出，而且训练是 end-to-end across text, vision, and audio。这个定义把 omni 和传统多模态模型区分开来。

传统 VLM 的主要形式是 image/video + text input，text output。它可以回答图片问题、做 OCR、理解视频，但交互仍然是文本中心。Audio-language model 则把 speech 或 sound 接进 LLM，通常输出文本。Speech pipeline 系统会把 ASR、LLM、TTS 串起来，看起来像语音助手，但中间丢失了很多信息：语气、停顿、环境声、重叠语音、情绪、非语言声音和实时 turn-taking。Omni model 的目标是减少这种中间损失，让模型直接在多模态信号之间推理和生成。

更宽的定义里，omni model 不是一个模型文件，而是一种交互系统。它需要低延迟输入、连续流式状态、语音输出、视觉输入、工具调用、记忆、打断处理和安全控制。OpenAI 的 Realtime API 和 Google Gemini Live API 都体现了这一点：用户不是把一个完整音频文件上传后等待结果，而是持续把 audio/video/text 流给模型，模型持续产生响应，并能处理实时对话中的打断和上下文变化。因此，omni 的关键不是“模态数量”，而是“实时多模态闭环”。

## 为什么 GPT-4o 是分界点

GPT-4o 之前，多模态研究已经有很长积累。CLIP、Flamingo、BLIP、LLaVA、Qwen-VL、Gemini 1.5、GPT-4V 等都推动了视觉语言理解；Whisper、Qwen-Audio、AudioPaLM、SpeechGPT 等推动了语音和音频理解；Video-MME 等 benchmark 推动了视频评测。但这些方向大多仍然是分块发展的：视觉是一条线，音频是一条线，语音合成是一条线，工具调用又是一条线。

GPT-4o 的冲击在于体验层面把这些线合到了一起。官方发布强调它能实时跨 audio、vision、text reasoning；System Card 披露它可以在 232ms 最低、320ms 平均延迟响应音频输入，接近人类对话节奏。这说明研究问题从“模型能否理解图片或音频”变成“模型能否像人一样边听、边看、边想、边说”。这也是为什么后续论文反复把 GPT-4o 作为参照物：VITA、Mini-Omni、Mini-Omni2、LLaMA-Omni、Baichuan-Omni 都在论文摘要或动机里强调 GPT-4o 展示的实时交互和 omni 能力。

另一个变化是安全和产品形态同步进入研究议题。GPT-4o 的语音能力带来了 voice cloning、情绪影响、拟人化、隐私和身份误用风险。System Card 对 speech-to-speech、persuasion、voice safety 等做了专门评估。也就是说，omni model 从诞生起就不是纯 benchmark 模型，而是直接面向真实用户、实时对话和社会风险的系统。

## 研究过程的第一阶段：从 pipeline 到端到端语音

GPT-4o 之后，最先被开源社区追赶的是 speech-to-speech。因为这是最容易被用户感知到的差距：传统语音助手需要 ASR -> LLM -> TTS，延迟高，容易丢语气，不能自然打断。Mini-Omni、LLaMA-Omni、Moshi 都可以放在这一阶段讲。

Mini-Omni 的目标是让语言模型在 streaming 场景里“hear, talk while thinking”。它提出 text-instructed speech generation 和 batch-parallel inference，试图在保持语言能力的同时让模型直接生成流式音频输出。Mini-Omni2 进一步加入视觉输入和 duplex interaction，朝“开源 GPT-4o”靠近。LLaMA-Omni 则基于 Llama-3.1-8B-Instruct，整合 speech encoder、speech adaptor、LLM 和 streaming speech decoder，直接从 speech instruction 同时生成 text 和 speech，并报告低至 226ms 的响应延迟。

Moshi 是另一条很重要的路线。它不是简单把 speech encoder 接到 LLM 上，而是把 spoken dialogue 看成 speech-to-speech generation，用神经音频 codec 的 token 来建模语音，并为用户 speech 和模型 speech 建立并行流，从而支持 full-duplex spoken dialogue。它提出 Inner Monologue，让模型先预测 time-aligned text tokens，再生成 audio tokens，提高语言质量并支持流式 ASR/TTS。这条路线的研究价值在于，它明确处理了真实对话里的 overlapping speech、interruption、interjection 和 speaker turn，而不是假设对话总是完整轮次。

这一阶段的核心问题是：语音不是文字的外壳。音频里有情绪、节奏、停顿、环境声、说话人状态和社会信号。如果模型只把语音转成文字再推理，很多信息会丢失。端到端 speech LLM 的目标就是把 audio token 作为可推理的对象，而不仅仅是输入输出接口。

## 第二阶段：从 speech omni 到 video-audio-text omni

仅有语音交互还不够。真正的 omni model 需要同时处理视觉、音频和文本，尤其要理解音画同步。VITA、Baichuan-Omni、Qwen2.5-Omni 是这一阶段的重要节点。

VITA 的定位是 open-source interactive omni multimodal LLM，处理 Video、Image、Text、Audio，同时强调 advanced multimodal interactive experience。它从 Mixtral 8x7B 作为语言基础出发，通过扩展中文词表、双语 instruction tuning、多模态 alignment 和 instruction tuning，逐步赋予视觉和音频能力。VITA 的意义不只在性能，而是提出开源社区也可以探索类似 GPT-4o 的多模态交互。

Baichuan-Omni 则提供了 7B 级别的开源 omni baseline，支持 image、video、audio、text 并发处理。Baichuan-Omni-1.5 进一步加入 end-to-end audio generation，并强调数据清洗与合成、多模态数据管线和音频 tokenizer。这个方向说明，omni 能力不只是模型结构问题，也是大规模多模态数据工程问题。

Qwen2.5-Omni 是目前开源 omni 研究中非常适合重点讲的模型。它是 end-to-end multimodal model，输入包括 text、image、audio、video，输出 text 和 natural speech，并支持 streaming。它提出 TMRoPE，用 Time-aligned Multimodal RoPE 同步 video timestamp 和 audio；提出 Thinker-Talker 架构，让 Thinker 负责文本推理，Talker 利用 Thinker hidden states 生成 audio tokens；同时使用 block-wise audio/vision encoders 和 sliding-window DiT 降低流式语音输出初始延迟。

这一阶段的核心问题变成：如何把不同时间尺度的模态放在同一坐标系里。图像是二维空间，视频是时间 + 空间，音频是高频时间信号，文本是离散 token。若只是拼接 token，模型知道序列顺序，却不一定知道“哪段声音对应哪一帧”。TMRoPE 这类设计说明，omni model 需要显式处理时间对齐。

![Architecture shifts](/images/blog/omni-architecture-shifts.svg "图 2：omni model 架构从 ASR-LLM-TTS pipeline，走向统一 token、Thinker-Talker 和实时 agent 系统。")

## 第三阶段：从模型能力到实时系统

2024 年下半年到 2025 年，产品和 API 也开始推动研究方向变化。OpenAI Realtime API 把 GPT-4o 的 speech-to-speech 能力开放给开发者，支持低延迟、多模态体验、语音输入输出、打断和工具调用。Google 的 Gemini Live API 也强调低延迟实时 voice/vision interaction，处理连续 audio、images/text 或 video stream，产生自然 spoken response。Project Astra 则把这个方向推向“universal AI assistant”，强调看见世界、听见声音、记住上下文、辅助用户完成任务。

这意味着 omni model 的研究对象不再只是 single-turn QA，而是 continuous interaction。模型需要在输入还没结束时就更新状态，需要在说话时被打断，需要结合屏幕、摄像头、麦克风、工具和记忆。传统 benchmark 往往假设输入是完整的，答案是一次性文本输出；实时 omni 场景则要求模型边输入边输出，甚至要主动判断什么时候该说话、什么时候该等待。

这也是为什么 OmniMMI、SocialOmni、MMOU、MMAO-Bench 等新 benchmark 开始出现。它们试图从 streaming video、audio-visual social interactivity、multi-task omni understanding 等角度评估模型。OmniBench 强调 image、audio、text 的互补信息推理；MMAU 强调 speech、environmental sounds、music 的专家级音频理解与推理；Video-MME 强调长视频和多领域视频理解；SocialOmni 则进一步关注音视频社交互动理解。这些 benchmark 共同说明，omni 评测正在从“单模态能力总和”转向“跨模态、跨时间、交互式推理”。

## 现在的研究现状

截至 2026 年 7 月 1 日，我认为 omni model 的研究现状可以概括为六条主线。

第一，native audio 和 speech-to-speech。代表工作包括 GPT-4o、Mini-Omni、LLaMA-Omni、Moshi、Qwen2.5-Omni。核心问题是低延迟、流式输出、音色自然度、情绪表达、打断和 full-duplex。未来要解决的是 speech reasoning，而不只是 speech response。

第二，audio-video temporal alignment。代表工作包括 Qwen2.5-Omni 的 TMRoPE、音视频 interleaving、长视频 frame selection 和 event boundary 研究。核心问题是让声音、画面、文本在同一时间坐标里推理。你的 long-video understanding 方向可以自然接到这里。

第三，Thinker-Talker 和多头生成架构。Qwen2.5-Omni 把 reasoning 和 speech generation 拆成 Thinker/Talker；Qwen3.5-Omni 又把 Hybrid Attention MoE 用到 Thinker 和 Talker，面向长序列推理。这个方向的核心问题是：如何让文本推理和语音生成互相支持，而不是互相干扰。

第四，长上下文和高效推理。Omni 输入天然很长，尤其是长音频、长视频和持续对话。Qwen3.5-Omni 的 Hybrid Attention MoE、vLLM-Omni 的 rollout 后端、Gemini Live 的低延迟系统都说明，未来竞争不只是模型大小，还包括 KV cache、流式编码、长视频压缩、memory 和推理调度。

第五，评测和数据。当前很多 omni 模型在短 demo 上很强，但在真实复杂交互中仍然难评估。OmniBench、MMAU、Video-MME、OmniMMI、SocialOmni、MMOU 都在补这个空白。未来需要更接近真实场景的 benchmark：多人对话、嘈杂环境、长视频事件、社交信号、跨轮记忆、工具使用和安全边界。

第六，post-training 和 agentic omni。Omni model 最终要服务于真实交互，必须通过 RLHF、DPO、GSPO、reward model 和用户反馈持续改进。VeRL-Omni 这类框架说明，多模态生成 RL 的训练基础设施正在形成。未来的研究可能不只是训练一个 omni backbone，而是训练一个可以看、听、说、行动、调用工具并被 reward 约束的 agent。

![Research map](/images/blog/omni-research-map.svg "图 3：当前 omni model 的研究地图：流式交互、时间对齐、长视频、评测、后训练与安全。")

## 分享时建议讲哪些论文

如果你要做一次完整分享，我建议按“体验突破 -> 开源复现 -> 架构深化 -> 评测与未来”的顺序讲，而不是按时间硬排。

第一组是起点和定义：

- GPT-4o release and GPT-4o System Card。讲清楚 omni 的定义、实时 audio/vision/text、端到端训练、低延迟和安全风险。
- Gemini Live / Project Astra。讲 closed-source 体系如何把 omni 推向 universal assistant、实时摄像头和屏幕共享。

第二组是 open-source speech interaction：

- Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming。讲实时 speech interaction、text-instructed speech generation 和 streaming output。
- LLaMA-Omni: Seamless Speech Interaction with Large Language Models。讲 speech encoder + adaptor + LLM + streaming decoder，以及低延迟 speech response。
- Moshi: a speech-text foundation model for real-time dialogue。讲 full-duplex、parallel speech streams、Inner Monologue 和 turn-taking。

第三组是 open-source omni understanding：

- VITA: Towards Open-Source Interactive Omni Multimodal LLM。讲 Video/Image/Text/Audio 同时处理，以及开源 omni interaction 的第一批尝试。
- Baichuan-Omni and Baichuan-Omni-1.5。讲 7B omni baseline、端到端 audio generation、数据清洗和多模态训练管线。
- AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling。虽然早于 GPT-4o，但非常适合讲“统一离散 token”路线，说明多模态可以像语言一样进入 LLM。

第四组是 Qwen omni 系列：

- Qwen2-Audio。作为 audio-language model 的前置工作，讲 voice chat/audio analysis 和 audio instruction following。
- Qwen2.5-Omni。重点讲 TMRoPE、Thinker-Talker、block-wise encoding、sliding-window DiT 和 streaming speech。
- Qwen3.5-Omni。重点讲 Hybrid Attention MoE、长音频/视频理解、Thinker/Talker 的高效长序列推理。

第五组是 benchmark：

- OmniBench。讲 image/audio/text 互补信息推理。
- MMAU。讲音频理解不只是 ASR，而是 speech、environmental sound、music 的专家级 reasoning。
- Video-MME。讲长视频评测和多领域视频理解。
- OmniMMI / SocialOmni / MMOU。讲 streaming interaction、social interactivity 和 massive omni understanding。

第六组是训练系统：

- VeRL-Omni。讲多模态生成 RL 后训练基础设施，rollout、async reward、vLLM-Omni、GSPO 和 reward service。

这样组织的好处是听众能看到研究问题如何一步步变形：先是“模型能说话”，然后是“模型能实时说话”，再到“模型能看着视频听着声音说话”，最后变成“模型能在复杂交互中被评测和后训练”。

## 未来应该怎么研究

我认为未来 omni model 可以沿着八个方向推进。

第一，真正的实时音视频推理。当前很多模型能处理音频和视频，但不一定能在 streaming 场景里稳定推理。未来需要模型在 partial input 下做可更新判断，而不是等完整视频结束后回答。

第二，长视频中的事件级时间结构。Omni model 不应该只均匀抽帧，而应该理解事件边界、音画同步、说话人动作、情绪变化和场景转折。Frame selection、semantic boundary、TMRoPE、event-anchored selection 可以结合起来。

第三，社交互动理解。真实交互中，语音语调、表情、动作、沉默、打断、视线都很重要。SocialOmni 这类 benchmark 说明，omni model 需要理解 social cues，而不是只识别物体和文字。

第四，speech reasoning。很多 speech LLM 能自然说话，但复杂推理仍然弱。未来需要研究模型如何一边听一边思考，如何在 speech token 和 text reasoning token 之间分配计算，如何避免语音流式输出牺牲推理质量。

第五，多模态 memory。持续交互要求模型记住之前看过、听过、说过的内容。Memory 不能只是文本摘要，还需要保存音视频事件、时间戳和跨模态引用。

第六，omni post-training。多模态 RLHF 还很早期。如何构造 audio-video reward、如何避免 reward hacking、如何服务化 reward model、如何在长视频和实时对话中做 on-policy training，都是开放问题。

第七，安全和身份边界。Voice cloning、情绪操控、隐私泄露、摄像头实时输入、环境声音采集都会带来风险。Omni model 的安全不能只靠文本拒答，还要有语音、视觉和实时交互层面的策略。

第八，efficient omni orchestration。不是所有任务都需要一个巨大的端到端模型。未来系统可能会结合 omni backbone、specialist experts、tool agents、retrieval 和 streaming memory。你的 training-free multimodal LLM orchestration 可以放在这个方向里：在不昂贵重训的情况下，让不同模态专家高效协同。

## 我的理解

从 GPT-4o 到 2026 年的 omni model 研究，可以看成三次转变。

第一次转变是从文本中心到交互中心。模型不再只是回答文本问题，而是要在用户说话、看屏幕、摄像头输入、环境声音和工具调用之间建立闭环。

第二次转变是从多模态输入到多模态生成。VLM 主要解决“看懂并说文字”，omni model 还要“听懂并说声音”，甚至未来要生成图像、视频、动作和工具操作。

第三次转变是从 benchmark 能力到系统能力。实时性、打断、音画同步、长上下文、reward、memory、安全和部署吞吐，都会影响模型是否真的可用。

所以，omni model 的研究不应该只问“哪一个模型分数最高”，而应该问：它是否能持续感知世界？是否能在低延迟下保持推理？是否能理解音视频里的社会信号？是否能通过后训练持续改进？是否能在安全边界内和人自然协作？这也是我认为 omni model 会成为多模态 AI 下一阶段核心方向的原因。

## 参考资料

- [Hello GPT-4o](https://openai.com/index/hello-gpt-4o/)
- [GPT-4o System Card](https://openai.com/index/gpt-4o-system-card/)
- [OpenAI Realtime API](https://openai.com/index/introducing-the-realtime-api/)
- [Gemini Live API](https://ai.google.dev/gemini-api/docs/live-api)
- [Project Astra](https://deepmind.google/models/project-astra/)
- [AnyGPT](https://arxiv.org/abs/2402.12226)
- [VITA](https://arxiv.org/abs/2408.05211)
- [Mini-Omni](https://arxiv.org/abs/2408.16725)
- [Mini-Omni2](https://arxiv.org/abs/2410.11190)
- [LLaMA-Omni](https://arxiv.org/abs/2409.06666)
- [Moshi](https://arxiv.org/abs/2410.00037)
- [Baichuan-Omni](https://arxiv.org/abs/2410.08565)
- [Baichuan-Omni-1.5](https://arxiv.org/abs/2501.15368)
- [Qwen2-Audio](https://arxiv.org/abs/2407.10759)
- [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215)
- [Qwen3.5-Omni](https://arxiv.org/abs/2604.15804)
- [OmniBench](https://m-a-p.ai/OmniBench/)
- [MMAU](https://arxiv.org/abs/2410.19168)
- [Video-MME](https://video-mme.github.io/home_page.html)
- [VeRL-Omni](https://github.com/verl-project/verl-omni)

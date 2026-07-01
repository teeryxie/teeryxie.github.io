# How to Train Omni Models: 从模态对齐到 RL 后训练

Omni model 的训练方法不能简单理解成“把图像、音频、视频数据都喂给 LLM”。真正困难的是：不同模态有不同采样率、不同 token 形式、不同输出空间、不同延迟约束和不同 reward。文本是离散 token，图像是空间 patch 或视觉 embedding，音频是高频连续信号或 codec token，视频是时间和空间的组合，语音输出还要求低延迟、自然 prosody、可被打断。训练一个 omni model，本质上是在训练一个能够跨模态感知、跨时间对齐、跨输出形式生成、并能在真实交互中持续优化的系统。

从 GPT-4o 之后的公开研究看，omni model 的训练大致形成了五层流程：模态编码与 tokenization、模态对齐预训练、多模态指令微调、流式语音/音视频生成训练、以及 reward-based post-training。不同论文会强调不同层。例如 AnyGPT 关注离散 token 统一建模，VITA 和 Baichuan-Omni 关注多阶段 multimodal alignment + instruction tuning，Mini-Omni 和 LLaMA-Omni 关注 speech interaction 的低延迟训练，Moshi 关注 full-duplex speech-to-speech，Qwen2.5-Omni 关注 TMRoPE、Thinker-Talker 和 streaming，VeRL-Omni 则把 RL 后训练工程化。

![Omni training pipeline](/images/blog/omni-training-pipeline.svg "图 1：omni model 的训练通常分为编码、对齐、SFT、流式生成和 RL 后训练几个阶段。")

## 第一层：模态表示和 tokenization

训练 omni model 的第一步是决定每个模态如何进入模型。文本最简单，直接使用 tokenizer。图像通常经过视觉 encoder 或 ViT-like patch encoder，再通过 projector/adaptor 映射到 LLM embedding space。视频可以看作多帧图像，但真正训练时还要处理 frame sampling、temporal position、spatiotemporal compression 和长上下文成本。音频最复杂，因为它既可以被当成理解输入，也可以被当成生成输出。

对于音频理解，常见做法是使用 pretrained speech/audio encoder，比如 Whisper-style speech encoder、BEATs、Whisper-large 或模型自带 audio tower，把音频压成 frame-level representation，再用 adaptor 接进 LLM。对于语音生成，很多工作会使用 neural audio codec，把连续音频离散化成 codec tokens。Moshi 使用 Mimi streaming neural audio codec，并把用户 speech 和模型 speech 建模成并行 token stream；Mini-Omni、LLaMA-Omni 等也都围绕 audio token 或 streaming decoder 设计低延迟语音输出。

AnyGPT 提供了另一种很清晰的思想：把 speech、text、image、music 都转成 unified discrete representations，然后用 next-token prediction 训练 LLM。它的优点是可以不改变 LLM 架构，把新模态当成“新语言”加入；缺点是 codec/tokenizer 的质量直接决定上限，且不同模态的离散化粒度很难统一。对于 omni model 来说，tokenization 不是预处理细节，而是决定模型是否能稳定学习多模态关系的基础。

## 第二层：模态对齐预训练

有了模态表示后，下一步是 alignment。视觉、音频、视频 encoder 输出的 embedding 并不天然位于 LLM 可理解的语义空间，必须通过 projector、adapter 或 cross-modal pretraining 对齐。VITA 的训练流程很典型：先做 LLM instruction tuning，再做 multimodal alignment，最后做 multimodal instruction tuning。Baichuan-Omni 也采用从 7B base model 出发，经过 multimodal alignment 和 multitask fine-tuning 的两阶段方案。

Alignment 阶段通常有几类目标。第一类是 captioning 或 QA 式语言建模：给图片、音频或视频，让模型预测描述文本或答案。第二类是 contrastive 或 matching：让模态表示和文本语义靠近。第三类是 modality adapter pretraining：冻结 LLM 和 encoder，只训练 projector，让外部模态先能被 LLM 接收。第四类是联合 LM loss：把特殊模态 token 插入文本序列，让模型在上下文中预测后续文本。

这个阶段的关键不是追求最终对话效果，而是让模型“看得懂、听得懂”。如果 alignment 不稳，后续 SFT 会变成用 instruction data 强行补语义，样本效率很差，也容易出现 hallucination。很多开源 omni 模型会先保留强大的 pretrained LLM，再逐步训练 visual/audio adaptor，就是为了避免一开始就破坏语言模型能力。

## 第三层：多模态指令微调

SFT 阶段把模型从“能感知”变成“会按用户意图行动”。训练数据通常包含图文问答、视频问答、音频问答、语音指令、跨模态推理、多轮对话和工具式任务。VITA、Baichuan-Omni、Mini-Omni2 都强调 instruction tuning，因为 omni model 的难点不是单个模态 benchmark，而是用户会用自然方式混合输入：“看这个视频，听他说话的语气，解释他为什么这么反应。”

多模态 SFT 有几个实际问题。第一，数据格式必须统一。不同数据集可能是 image-text pair、audio caption、video QA、speech instruction、OCR task、music understanding，需要转换成统一 chat template 或 conversation format。第二，任务比例要平衡。如果图像数据太多，模型可能忽视音频；如果语音数据太少，语音输出会退化。第三，negative 或 refusal 数据也要覆盖多模态场景，否则模型可能在摄像头、声音、身份识别和隐私问题上表现不安全。

Speech interaction 的 SFT 更特殊。LLaMA-Omni 构造了 InstructS2S-200K，把 speech instructions 和 speech responses 配对，用来训练模型直接从语音指令生成文本和语音响应。Mini-Omni2 用三阶段训练对齐视觉和音频模态，并加入 command-based interruption mechanism，让用户可以在模型说话时打断。这说明 omni SFT 不只是内容对齐，还包括 interaction protocol 的学习。

## 第四层：语音生成和流式输出训练

Omni model 和普通 VLM 最大的差异之一是输出不再只是文本。语音输出要求模型生成连续、自然、低延迟的 audio token。Pipeline 系统可以把 LLM 文本输出交给 TTS，但这会损失语气、节奏和实时性。端到端 speech-to-speech 模型则需要在训练中同时处理语义内容和声学细节。

Moshi 的 Inner Monologue 是很有代表性的训练方法。它让模型先预测 time-aligned text tokens，再预测 audio tokens。这样做有两个好处：一是保留文本语言模型的语义能力，二是让 audio generation 有一个可解释的语义中间层。Moshi 同时建模用户语音流和模型语音流，支持 full-duplex conversation，不需要严格的 turn segmentation。

Qwen2.5-Omni 采用 Thinker-Talker 架构。Thinker 负责多模态理解和文本推理，Talker 使用 Thinker hidden states 生成 speech tokens。这个拆分很重要，因为 reasoning 和 speech synthesis 的优化目标不同。Thinker 要回答对，Talker 要说得自然、稳定、低延迟。如果把两者完全绑在一个输出 head 上，训练可能互相干扰。Qwen3.5-Omni 进一步在 Thinker 和 Talker 上使用 Hybrid Attention MoE，说明长上下文和语音输出都需要更高效的结构。

流式输出训练还涉及 chunking 和 latency。模型不能等整段输入结束后才生成语音，而要 block-wise processing。Qwen2.5-Omni 的 audio/vision encoder 使用 block-wise 处理，Talker 使用 sliding-window DiT 和 modified BigVGAN decoder，以控制 streaming speech 的初始延迟。训练时必须模拟这种 chunked inference，否则离线训练得到的模型在实时场景中可能不稳定。

## 第五层：时间对齐训练

Omni model 的核心难点之一是音视频时间对齐。音频采样率高，视频帧率低，文本 token 更稀疏。如果训练时只是把音频 token 和视觉 token 串起来，模型很难知道“哪个声音对应哪一帧”。Qwen2.5-Omni 的 TMRoPE 正是为此提出：它把 rotary position 拆成 temporal、height、width 组件，并让 audio/video 在真实时间轴上对齐。

训练时的 time interleaving 也很关键。对于带音频的视频，Qwen2.5-Omni 按时间 chunk 组织 audio/video representation，而不是先放完所有 video token 再放 audio token。这样相邻时间段内的视觉和音频 token 在序列中更近，同时 TMRoPE 给它们一致的 temporal reference。这个设计对长视频理解尤其重要，因为动态 frame sampling 会改变帧间距离，position encoding 必须反映真实时间，而不是简单帧编号。

未来的训练方法可以进一步把 event boundary 引入 temporal objective。长视频里的重要信息不均匀，某些事件转折点比大量平稳帧更重要。训练模型时，如果能把 frame selection、semantic boundary、audio event 和 speech timestamp 结合起来，模型可能更适合 long-video reasoning 和 social interaction。

![Training objectives](/images/blog/omni-training-objectives.svg "图 2：omni model 训练目标并不单一，而是由 alignment、autoregressive generation、temporal grounding、SFT 和 preference/RL 组成。")

## 第六层：高效训练和长上下文结构

Omni 输入天然很长。长音频、长视频、多轮对话、实时屏幕流都会让上下文迅速变大。训练上需要考虑三个层面：数据长度怎么采样，模型结构怎么处理长序列，分布式系统怎么承受显存和吞吐。

Qwen3.5-Omni 的 Hybrid Attention MoE 是一个重要信号：omni model 不可能每层都用昂贵 full attention 处理所有音视频 token。Hybrid Attention 让模型在部分层做全局交互，在其他层用更高效的 sequence mixing；MoE 则让不同 token 激活不同专家，提高总容量但控制 active compute。训练这类模型还要处理 expert load balancing、长上下文 batch packing、KV cache、activation checkpointing 和序列并行。

开源训练通常会用 LoRA 或 partial tuning 控制成本。Qwen3-Omni Thinker GSPO recipe 只训练 Thinker 的 LoRA，冻结或剥离 talker、vision tower、audio tower 等模块。这样做不是因为其他模块不重要，而是因为 omni model 的全量训练成本太高，必须分阶段优化。对研究者来说，合理选择 freeze/unfreeze 策略，本身就是训练方法的一部分。

## 第七层：偏好对齐和 RL 后训练

SFT 只能模仿数据，不能直接优化用户偏好、实时交互质量和安全边界。Omni model 最后需要 post-training。文本 LLM 中常见 RLHF、DPO、PPO、GRPO；在 omni 场景里，reward 更复杂。语音任务要评估内容、语气、自然度、延迟、是否可打断；视频任务要评估 grounding、时间对齐、事件理解；社交互动任务要评估情绪、关系、非语言线索；安全任务要评估隐私、身份、声音克隆和视觉敏感内容。

VeRL-Omni 的意义在这里非常明显。它把 rollout、reward loop、trainer、vLLM-Omni 后端和 async reward 组织起来，为多模态生成 RL 提供基础设施。对于 Qwen3-Omni Thinker，它提供 GSPO + LoRA recipe；对于 Qwen-Image，它支持 FlowGRPO、Flow-DPPO、DPO、DiffusionNFT 等 diffusion RL 方法。虽然这些算法不完全等同于 omni speech model RL，但它们说明多模态生成的后训练正在从“概念”走向可复现实验栈。

Omni RL 的难点是 reward design。一个模型可能学会让语音更讨好 reward model，但内容变差；也可能在 OCR reward 上过拟合，牺牲图像美感；还可能在社交互动里迎合错误情绪标签。因此，未来 post-training 需要 multi-reward、human preference、model judge、rule verifier 和安全 reward 的组合，并持续监控 reward hacking。

![RL post-training loop](/images/blog/omni-training-rl-loop.svg "图 3：omni model 的 RL 后训练需要 rollout、多模态 reward、actor 更新和评测闭环。")

## 一个实用训练 recipe

如果从研究实现角度设计一个 omni model training recipe，我会按以下顺序推进。

第一步，确定基础 LLM 和模态 encoder。先选一个强语言模型，再选视觉 encoder、audio encoder、audio codec/tokenizer。不要一开始就追求所有模态端到端训练，先保证每个输入模态能被稳定映射到 LLM 空间。

第二步，做单模态到语言的 alignment。图像 caption、video caption、audio caption、ASR、speech QA、OCR 都可以作为早期任务。此阶段可以冻结 LLM，只训 projector/adaptor，也可以逐步解冻少量层。

第三步，做多模态混合 SFT。把 image/audio/video/text 任务统一成 conversation format，加入跨模态推理数据和多轮交互数据。训练时要控制任务比例，避免某个模态主导。

第四步，加入语音输出。可以采用 two-stage 方法：先训练 text response，再训练 speech token generation；也可以像 Thinker-Talker 那样拆分 reasoning 和 speech synthesis。要在训练中模拟 streaming chunk，不能只训练离线整段输出。

第五步，加入 temporal alignment。视频音频任务需要真实 timestamp、frame time、audio chunk time，并用 TMRoPE、interleaving 或事件边界让模型学习同步关系。长视频任务还要加入 frame selection 或 event-anchor 数据。

第六步，做 instruction + safety tuning。加入拒答、隐私、身份、声音克隆、敏感视觉、摄像头场景等安全数据。Omni 安全必须覆盖语音和视觉，不只是文本安全。

第七步，做 reward-based post-training。先从容易验证的任务开始，例如数学语音问答、OCR、temporal grounding、long-video QA；再逐步加入 human preference 和 social interaction reward。工程上需要 reward service、async reward 和可复现评测。

## 当前方法的主要问题

第一，数据仍然是瓶颈。高质量 audio-video-text 对齐数据很少，真实多轮语音视频交互数据更少。很多模型依赖合成数据或 pipeline 生成数据，这会带来噪声和分布偏移。

第二，训练目标不统一。理解任务、语音生成、视频 grounding、实时交互和安全对齐的 loss 不在同一个尺度上。如何动态平衡这些目标，是 omni training 的关键问题。

第三，评测不够贴近训练目标。很多 benchmark 仍然是离线 QA，而训练目标希望模型实时交互。未来需要把 latency、interruption、multi-turn memory、audio-visual grounding 和 user satisfaction 放进评测闭环。

第四，RL reward 还不成熟。文本 RLHF 已经很难，多模态 RLHF 更难。Reward model 可能看不懂视频细节，也可能对语音情绪误判。没有可靠 reward，就很难稳定优化 omni interaction。

第五，训练成本极高。一个真正的 omni model 需要 LLM、vision encoder、audio encoder、speech decoder、video pipeline、reward model、rollout backend 和 distributed trainer。研究者需要更多参数高效、training-free 或 modular orchestration 方法。

## 未来研究方向

我认为 omni model training 未来有几个重要方向。

第一，event-aware multimodal training。训练数据和 loss 不应只按固定时间切块，而应该围绕事件边界、说话人 turn、音视频同步点和用户意图组织。

第二，speech reasoning pretraining。当前很多 speech model 会说话，但复杂推理仍依赖文本中间层。未来可以研究 speech token 与 text reasoning token 的联合 pretraining，让模型保留语音细节同时具备强推理。

第三，long-video RLHF。长视频任务的 reward 可以结合 temporal grounding、event retrieval、caption consistency 和 human preference。这个方向和 long-video understanding 非常接近。

第四，omni-modal orchestration training。不是所有能力都要端到端塞进一个模型。可以训练 router 或 controller，协调 vision expert、audio expert、LLM、speech generator 和 memory。你的 training-free orchestration 可以作为低成本起点。

第五，safe full-duplex interaction。模型边听边说时，安全策略也必须实时生效。未来需要训练模型识别危险语音、隐私泄露、身份冒充、视觉敏感内容，并能在不中断正常交互的情况下处理。

## 我的理解

Omni model 的训练方法正在从“多模态 SFT”走向“实时交互系统训练”。早期训练关注的是把视觉和音频接进 LLM；现在训练要同时考虑流式输入、自然语音输出、音视频时间对齐、长上下文、高效推理、用户偏好和安全边界。

如果用一句话概括：训练 omni model 不是训练一个会接收多种模态的模型，而是训练一个能在不确定、连续、嘈杂、实时的人类环境中稳定协作的系统。这也是为什么它和普通 VLM 不同。VLM 的核心问题是理解，omni model 的核心问题是交互。

## 参考资料

- [GPT-4o System Card](https://openai.com/index/gpt-4o-system-card/)
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
- [VeRL-Omni](https://github.com/verl-project/verl-omni)

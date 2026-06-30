# Hybrid Attention MoE: 面向长上下文 Omni 模型的高效架构

Hybrid Attention Mixture-of-Experts，简称 Hybrid Attention MoE，可以理解为新一代长上下文大模型在“注意力成本”和“模型容量”之间做出的系统级折中。它不是一个单独模块，而是一组架构思想的组合：一方面，模型不再把所有层都设计成昂贵的全局 self-attention，而是把全局注意力、滑动窗口注意力、线性或递归状态更新、门控注意力等机制混合起来；另一方面，前馈网络部分不再让每个 token 都激活全部参数，而是用稀疏 MoE 路由到少量专家，让模型拥有更大的总容量，但保持相对可控的每 token 计算量。

这类设计在纯文本长上下文模型里已经很重要，在 omni 模型里更是几乎不可避免。原因很直接：omni 输入不是普通文本，而是文本、图像、视频、音频、语音输出历史、工具调用、用户交互状态共同组成的长流。一个 400 秒视频、一段 10 小时音频或一个持续多轮的实时助手会话，都可能迅速把上下文推到几十万 token 级别。如果继续依赖每层完整 self-attention，成本会非常高；如果完全放弃全局注意力，又容易丢掉跨时间、跨模态、跨事件的关键依赖。Hybrid Attention MoE 的核心价值就在这里：把“全局建模能力”保留下来，同时把长序列处理的主要负担交给更高效的混合层和稀疏专家。

![Hybrid Attention MoE overview](/images/blog/hybrid-attention-moe-overview.svg "图 1：Hybrid Attention MoE 在长上下文 omni 模型中的基本位置。")

## 为什么长上下文需要 Hybrid Attention

Transformer 的标准 self-attention 很强，因为任意 token 都可以直接看见任意其他 token。对于短文本、短图文上下文，这种全局连接非常有效。但它的计算和缓存成本会随着序列长度急剧上升。长上下文模型真正痛苦的不是“多读一点文本”，而是每一层都要维护大量 key-value cache，并在推理时持续处理越来越长的历史。当上下文从 8k 增长到 128k、256k，甚至更长时，注意力成本会成为系统瓶颈。

在 omni 场景里，这个问题被进一步放大。文本 token 通常还算稀疏，但音频是高频信号，视频包含连续帧，图像和视频帧还会被切成视觉 token。即使前端编码器做了压缩，进入 LLM 的表示仍然可能很长。更麻烦的是，这些 token 的重要性高度不均匀：很多音频帧只是背景，很多视频帧几乎没有变化，但某些瞬间可能包含用户问题的关键线索。模型需要既能低成本地扫过长流，又能在必要时进行跨模态全局对齐。

Hybrid Attention 的直觉就是：不要让每一层都承担同一种职责。部分层负责高效传播局部和长程状态，部分层负责关键的全局交互，部分层用门控机制决定哪些信息应该进入状态。这样做的结果不是简单降低成本，而是把不同类型的依赖关系分配给不同机制。对于长视频和长音频，连续时间结构可以由高效 mixing 或状态更新承载；对于问题相关的片段、跨模态同步、指代表达和推理跳转，则仍然需要全局 attention 层提供连接。

![Hybrid attention cost](/images/blog/hybrid-attention-moe-cost.svg "图 2：Dense attention、efficient mixing 和 hybrid attention 的成本直觉。")

## 从 Dense Attention 到 Hybrid Attention

Dense attention 的优势是表达力强。它不预设哪些 token 重要，也不限制注意力连接范围。用户问“刚才那个人说这句话之前发生了什么”时，模型理论上可以从当前问题跳到语音片段，再跳到对应视频帧，再跳到更早的事件。它的问题是成本高，而且在超长上下文中，很多全局连接并不必要。

局部 attention 或 sliding-window attention 试图降低成本，让 token 只看附近窗口。它适合流式输入和局部连续信号，但对跨段依赖不友好。线性 attention、state-space model、recurrent memory、DeltaNet 这类机制则进一步把序列处理转化为状态更新，优势是更适合极长序列，推理缓存压力小，吞吐更好。问题在于，如果只有这类机制，模型可能难以处理稀疏但重要的全局跳转，尤其是多模态问答中那些跨很远时间的引用关系。

Hybrid Attention 的设计把两者结合起来。公开资料中，Qwen3-Next 系列就把 Gated DeltaNet 与 Gated Attention 结合，并配合高稀疏 MoE。Qwen3.5-Omni 技术报告也提到，其 Thinker 和 Talker 都采用 Hybrid-Attention MoE 架构，以支持更长上下文和高效推理。这里需要谨慎区分：不同模型的具体层比例、路由策略和训练细节不一定完全相同；但从架构趋势看，核心方向是一致的，即用混合注意力降低长序列成本，用稀疏专家提升容量。

我认为 Hybrid Attention 最重要的不是某个具体公式，而是它改变了“每层都做全局注意力”的默认假设。长上下文不是短上下文的线性扩展。真正有效的架构应该承认不同层有不同职责：一些层像扫描器，快速维护时间流；一些层像检索器，在关键位置建立全局连接；一些层像对齐器，把音频、视频和文本放进统一推理空间。Hybrid Attention 的价值就是让这些职责可以在同一个 backbone 中共存。

## MoE 解决的是容量，而不是上下文长度本身

Mixture-of-Experts 常被简单理解为“参数更多但计算不同比例增加”。这个说法没错，但还不够。MoE 的本质是 conditional computation：同一个输入序列中的不同 token 可以激活不同专家。对于每个 token，router 会给专家打分，只选择 top-k 个专家参与计算。这样模型可以拥有很大的专家池，但每个 token 只使用其中很小一部分。

在多模态模型里，这个性质很自然。文本推理 token、视觉事件 token、音频情绪 token、数学符号 token、工具调用 token，它们对前馈网络的需求并不完全相同。让所有 token 都走同一套 dense FFN，等价于要求一个固定容量模块同时处理所有模式；MoE 则允许模型把不同模式映射到不同专家组合。即使专家不是显式按模态命名，训练也可能形成某种隐式 specialization。

![Sparse MoE routing](/images/blog/hybrid-attention-moe-router.svg "图 3：Sparse MoE 通过 router 为不同 token 选择少量专家。")

不过，MoE 不是免费午餐。第一，路由需要负载均衡，否则少数专家会过载，其他专家利用不足。第二，分布式训练和推理会引入通信开销，尤其当 expert parallelism 跨设备时，all-to-all 通信可能成为瓶颈。第三，MoE 提升的是条件容量，不直接解决长上下文注意力的二次复杂度。因此，MoE 和 Hybrid Attention 的关系应该是互补：Hybrid Attention 解决“长流怎么处理”，MoE 解决“每个 token 的语义变换容量怎么扩展”。

这也是为什么 Hybrid Attention MoE 是一个组合词，而不是两个互不相关的模块。对于长上下文 omni 模型，仅有 Hybrid Attention 可能高效但容量不足；仅有 MoE 可能容量很大但上下文成本仍然高。二者结合以后，模型既可以用混合注意力处理长序列，又可以用稀疏专家承载复杂语义和模态差异。

## 为什么 omni 模型尤其需要这套组合

Omni 模型的目标不是简单把图像、音频、视频接到 LLM 上，而是让系统在真实交互中持续感知、推理和生成。Qwen2.5-Omni 引入 TMRoPE 来处理音视频时间对齐；Qwen3.5-Omni 进一步强调更长上下文、更强的音频视频理解和更高效的端到端交互。从这个脉络看，Hybrid-Attention MoE 是很自然的下一步：当输入时长和模态复杂度上来以后，位置对齐只是基础，架构本身也必须能承受长流。

以长视频理解为例，模型需要同时处理三个层次。第一是局部连续性，例如动作如何从前一帧过渡到后一帧，语音音素如何组成词。第二是事件结构，例如一个动作开始、发展、结束，或者某段对话对应某个视觉变化。第三是任务相关的全局依赖，例如用户最后问的问题可能指向几分钟前的一个细节。Dense attention 擅长第三类，但对第一类和第二类成本太高；高效状态更新擅长第一类和部分第二类，但不一定擅长第三类。Hybrid Attention 正好把这些能力拆开组合。

再看实时交互。一个交互式助手不能等所有输入结束才做推理。它需要在音频输入还在流动、视频画面还在变化时维持内部状态，并在必要时生成文本或语音回复。这要求模型的推理缓存、增量更新和延迟都可控。纯 dense attention 在长会话中会越来越重；纯局部机制又可能遗忘早期重要信息。Hybrid Attention MoE 通过高效层处理大多数流式状态，通过全局层保留关键跳转能力，再用 sparse experts 增加条件容量，更接近实际系统需求。

## Thinker-Talker 视角下的 Hybrid-Attention MoE

Omni 模型通常不只有一个语言生成模块。以 Qwen 系列 omni 架构为例，Thinker 负责多模态理解和文本侧推理，Talker 负责生成自然语音。Thinker 更像一个统一推理核心，需要理解文本、音频、图像、视频之间的关系；Talker 则要在 Thinker 的隐状态基础上生成连续、自然、低延迟的语音 token。两者面对的序列结构并不相同，但都受长上下文和实时性的约束。

Thinker 使用 Hybrid-Attention MoE 的意义在于，它需要在长上下文里保持推理能力。用户可能基于很早之前的视频细节提问，也可能要求模型比较不同时间段的事件，或者结合语音语调和画面动作判断意图。Thinker 不能只做局部匹配，它需要跨模态全局推理；但它也不能每层都用昂贵全局 attention，否则长输入会拖垮推理效率。

Talker 使用类似思想则更多服务于低延迟语音生成。语音输出是连续信号，对节奏、音色、停顿和语义一致性都有要求。如果 Talker 完全依赖短窗口，可能语音连贯性不足；如果每一步都做高成本全局建模，实时性又会受影响。Hybrid Attention 可以让 Talker 在维持局部语音动态的同时，周期性接入更长范围的语义状态；MoE 则为不同语音模式、语言、情绪和上下文提供更大的条件容量。

这说明 Hybrid Attention MoE 在 omni 模型里不是只为“跑得更快”，而是为了让理解和生成都能在长流中稳定运行。Thinker 侧重跨模态推理，Talker 侧重连续语音生成，它们都需要效率和表达力之间的平衡。

## 和 TMRoPE 的关系：坐标对齐之后，还要高效推理

如果把 TMRoPE 和 Hybrid Attention MoE 放在一起看，前者解决的是“token 在哪里、何时发生”，后者解决的是“这么长、这么复杂的 token 流如何被模型处理”。TMRoPE 给音频和视频建立共享时间坐标，使模型更容易知道哪个声音对应哪个画面；Hybrid Attention MoE 则让模型在这个坐标化的长流上进行高效建模。

这两个方向并不冲突，反而互相依赖。没有时间对齐，Hybrid Attention 即使高效，也可能在跨模态同步上学得很吃力；没有高效架构，TMRoPE 即使提供了清晰坐标，也难以支撑更长视频、更长音频和持续交互。换句话说，位置编码提供结构先验，混合注意力提供长序列处理机制，MoE 提供条件容量。一个成熟的 omni 系统需要三者同时成立。

这也给 long-video understanding 研究带来启发。过去很多工作集中在 frame selection、temporal grounding 或 video-language alignment。它们当然重要，但如果后端模型的长上下文机制本身不够高效，那么选择再好的帧也可能被上下文长度限制住。未来的长视频系统可能需要同时设计采样策略、时间编码、混合注意力和专家路由，而不是把它们当作孤立模块。

## 对研究的启发

我觉得 Hybrid Attention MoE 对多模态研究至少有三个启发。

第一，长上下文能力不应该只用最大 token 数来衡量。很多模型宣称支持 128k 或 256k 上下文，但真正关键的是在这么长的上下文中能否保持有效检索、跨模态对齐、事件推理和低延迟生成。Hybrid Attention 的存在说明，长上下文是一种系统能力，而不是简单扩展 position embedding。

第二，专家路由可以和模态结构结合。现在很多 MoE 模型的专家 specialization 是隐式形成的，未来可以研究更可解释的路由机制，例如让不同专家更好地服务于视频事件、音频情绪、文本推理、工具调用或跨模态对齐。当然，这不能变成硬编码模态专家，否则会损失泛化能力；更好的方式可能是弱约束、可解释分析或动态路由正则。

第三，Hybrid Attention 可以和事件边界结合。长视频中的关键不是每一秒都同等重要，而是事件发生变化的位置更重要。如果模型能够在事件边界附近使用更强的全局注意力，在平稳片段使用更高效的状态更新，那么计算资源会更符合视频本身的结构。这与 frame selection、semantic boundary detection 和 long-video grounding 都有天然联系。

从我自己的研究兴趣看，Hybrid Attention MoE 和 training-free multimodal orchestration 也有关联。Orchestration 关注的是系统层面如何选择、组合和调用不同模型或专家；MoE 则是在模型内部做 token-level expert routing。一个在外部调度专家，一个在内部路由专家。二者都指向同一个趋势：未来多模态系统不会是单一 dense 模型包打天下，而会越来越依赖条件计算、动态选择和结构化协同。

## 局限与需要谨慎的地方

Hybrid Attention MoE 很有吸引力，但不能把它神化。首先，混合注意力的设计空间很大，不同模型的具体实现差异可能非常大。Gated attention、linear attention、sliding window、DeltaNet、state-space layer、global attention 的组合比例都会影响性能。公开博客或模型卡通常只给出高层描述，不能据此推断所有内部细节。

其次，MoE 的实际收益取决于训练和系统实现。路由不均衡、专家退化、通信开销、推理框架支持不足，都可能抵消理论收益。尤其在多模态场景里，token 分布比纯文本更复杂，专家负载也可能随输入类型大幅变化。如果没有良好的负载均衡和部署优化，MoE 可能在论文指标上好看，但在线推理体验不稳定。

再次，Hybrid Attention MoE 不能替代数据、任务和评测。一个模型即使用了先进架构，如果缺少高质量音视频数据、缺少交互式评测，仍然可能无法真正理解复杂多模态场景。因此，架构只是基础设施，最终能力还要靠数据构造、训练目标、benchmark 设计和系统调优共同决定。

## 我对 Hybrid Attention MoE 的理解

我倾向于把 Hybrid Attention MoE 看作长上下文 omni 模型的“计算分层协议”。TMRoPE 这类位置编码告诉模型不同 token 的时间与空间坐标；Hybrid Attention 告诉模型哪些依赖用全局连接处理，哪些依赖用高效状态传播处理；MoE 告诉模型不同 token 应该调用哪些语义变换能力。三者合起来，才比较接近一个可扩展的多模态智能系统。

如果只追求 benchmark 上的短输入准确率，Dense Transformer 仍然很强。但如果目标是十小时音频、几百秒高清视频、多轮实时交互和自然语音输出，那么架构必须改变。Hybrid Attention MoE 的意义不只是降低 FLOPs，而是让模型的计算结构更接近真实任务结构：大多数时间是连续流，少数位置需要全局跳转；大多数 token 不需要全部专家，少数 token 需要特化能力；大多数输入是局部演化，关键事件需要跨时间整合。

这也是我认为它值得关注的原因。未来的 omni 模型不会只靠更大的 dense 参数量竞争，而会在时间对齐、长上下文机制、专家路由、流式生成和系统调度上共同进化。Hybrid Attention MoE 正处在这个交叉点上：它既是模型架构问题，也是系统效率问题，更是多模态智能如何从“能看能听”走向“持续理解和实时交互”的关键路径。

## 参考资料

- [Qwen3.5-Omni Technical Report](https://arxiv.org/abs/2604.15804)
- [Qwen3-Next model card on Hugging Face](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- [Qwen3-Next model card on ModelScope](https://modelscope.cn/models/Qwen/Qwen3-Next-80B-A3B-Instruct)

# Qwen3.5-Omni 论文精读：256K 长上下文、ARIA 与原生 Omni Agent

如果说 Qwen3-Omni 解决的是“多种模态能否在一个模型里共存而不整体降智”，那么 Qwen3.5-Omni 进一步追问：**这样的全模态模型能否扩展到数十万 token、数小时音频、实时语音和工具调用，并在长时间交互中保持稳定？**

论文给出的答案不是单纯增加参数，而是同时改造四个瓶颈：Thinker 与 Talker 都切换到 Hybrid-Attention MoE，用 Gated Delta Net 降低长序列 KV-cache 压力；预训练扩到约 4T 多模态 token，并把最长上下文推进到 262,144；用显式文本时间戳弥补 TMRoPE 在极长音视频上的稀疏位置问题；用 ARIA 将文本与语音从双轨生成改成自适应单流交错，从根源上减少漏词、错读数字和跨语言节奏失配。

我的判断是：Qwen3.5-Omni 最有研究价值的部分不是“215 个任务 SOTA”的宣传数字，而是它公开承认 Qwen3-Omni 的两个核心设计在规模扩大后会暴露问题，并给出新的替代方案。显式时间戳和 ARIA 都不是更复杂的神经模块，而是重新设计 token 序列与训练约束。这类改动往往比堆更大 backbone 更值得精读。

![Qwen3.5-Omni upgrade map](/images/blog/qwen35-omni-upgrades.svg "图 1：Qwen3.5-Omni 相对 Qwen3-Omni 的主要升级，从长上下文 backbone、时间建模到语音对齐和 agentic post-training。")

## 1. 从 omni model 到 native omni agent

Qwen3.5-Omni 延续 Thinker-Talker 架构：Thinker 接收文本、图像、音频和带声音的视频，完成理解、推理、文本生成与工具决策；Talker 读取多模态上下文及 Thinker 输出，生成流式语音。不同之处是，论文把目标从被动的 perception-response 模型推进到“原生 omni agent”：模型不仅回答问题，还应自主 WebSearch、发起复杂 FunctionCall、根据音视频指令生成代码，并进行实时流式交互。

这一定位改变了训练重点。一个离线视频问答模型只需在最终答案上正确；agent 则需要在多轮上下文中保持语言、人格和指令一致性，判断何时调用工具，并让工具结果重新进入音视频对话。报告中的 Interaction-Aligned RL、OmniGAIA 工具评测和 Audio-Visual Vibe Coding 都围绕这个转向展开。

模型系列包含 Flash 和 Plus 两个 instruct 版本，二者均支持 256K 输入。技术报告称 Plus 扩展到数千亿参数级，但没有在正文中披露足以精确复现的总参数、激活参数、专家数和层配置。因此，精读时应把“hundreds of billions”视为规模描述，而不能自行补成一个具体参数值。

## 2. Hybrid-Attention MoE：长音视频为什么不能只靠全注意力

Qwen3-Omni 的 Thinker/Talker 已使用 MoE，但注意力仍面临长序列成本。Qwen3.5-Omni 将两者都换成 Qwen3.5 系列的 **Hybrid MoE**：在保留全局注意力能力的同时，引入 Gated Delta Net，也就是 GDN，处理大量局部或递推的序列状态。

对 256K omni 上下文而言，问题不只是 attention FLOPs。音频和视频 token 会让 KV cache 巨大，服务时每生成一个 token 都要频繁读取历史 KV。GDN 用压缩状态承载一部分长程信息，可以减少 KV-cache I/O；少量全局注意力层再负责需要精确 token-to-token 检索的关系。MoE 则让模型拥有更大总容量，但每个 token 只激活部分专家。

这种组合适合 omni 数据的原因在于，音视频同时包含大量冗余连续片段和少数关键事件。连续背景声、稳定镜头和说话人音色适合状态式累积；跨镜头因果、数字细节、工具参数和很早之前的用户约束仍需要精确注意力。Hybrid Attention 的真正价值是给这两类依赖分配不同计算路径。

但论文没有给出完整的层比例、GDN 与 attention 的消融，也没有公开 Plus 的部署并行配置。我们可以确认架构方向和延迟结果，却无法从报告单独判断性能提升中有多少来自 Hybrid Attention、多少来自模型规模和训练数据。

## 3. 256K 上下文：不是把位置长度参数改大

Qwen3.5-Omni 把可处理输入扩展到 256K token，官方给出的代表性上限包括超过 10 小时音频，或 400 秒、720P、1 FPS 的音视频。不同模态的 tokenizer 与动态采样会改变实际时长，因此这些数字是指定处理设置下的容量说明，不是所有输入组合都能同时达到的硬保证。

音频编码器 AuT 也随之升级。它使用 Qwen3-ASR 生成的 4000 万小时 audio-text 数据从零训练，多语言数据超过 20 种；Conv2D 下采样从 8 倍增加到 16 倍，输出速率从 Qwen3-Omni 的 12.5Hz 降到 6.25Hz，即每个音频表示约覆盖 160ms。token rate 减半直接降低长音频上下文成本，但也可能压缩细粒度声学事件，模型必须依靠更大数据和 encoder 表示能力弥补。

预训练仍是三阶段。S1 冻结 Qwen3.5 LLM，分别对齐视觉与 AuT encoder；S2 解冻全部参数，在 32K 长度上训练约 4T token，其中论文按模态统计为文本 0.92T、音频 1.99T、图像 0.95T、视频 0.14T、音视频 0.29T；S3 再把长度从 32768 推到 262144，并增加长音频与长视频比例。和 Qwen3-Omni 一样，分项 token 数不是互斥桶，不能机械相加后质疑总量口径。

![Qwen3.5-Omni long-context pipeline](/images/blog/qwen35-omni-long-context.svg "图 2：低速率 AuT、显式时间戳、Hybrid MoE 与分阶段扩窗共同支撑 256K omni 上下文。")

## 4. 为什么显式时间戳部分替代纯 TMRoPE

这是报告里最值得注意的自我修正。Qwen3-Omni 把真实时间映射为 temporal position ID，使音频与视频共享绝对时间轴。但当视频很长、帧率变化大时，绝对时间 ID 会变得很大而稀疏，模型难以学习长程关系；为了让模型适应各种 FPS，还需要构造覆盖均匀的大量训练样本。

Qwen3.5-Omni 保留 TM-RoPE 的时空位置结构，但在每个视频或音视频 temporal patch 前加入以秒为单位的格式化文本时间戳；纯音频序列则随机间隔插入时间戳。这样，“当前发生在 132.4 秒”不再只藏在旋转角度中，也成为语言模型可直接读取、复制和推理的符号。

显式时间戳有三个优点。第一，它利用 LLM 已经具备的数字和 timecode 处理能力，适合问“第几秒发生了什么”。第二，它不要求位置 ID 在所有帧率上均匀采样，降低数据构造成本。第三，它天然支持结构化 caption、scene segmentation 和工具参数输出。代价是增加 token，并把连续时间量化为文本字符串；时间格式、数值 tokenization 和训练提示可能影响泛化。

这项设计对长视频研究有直接启发：位置编码负责提供底层顺序和局部对齐，显式 timestamp 负责可解释的时间推理；如果再加入 event boundary token，模型可能更容易在长视频里围绕事件而不是均匀帧进行检索。

## 5. ARIA：文本 token 和语音 token 为什么会走散

流式 Talker 同时依赖 Thinker 的文本和自己生成的语音 codec。问题是不同语言的文本 tokenizer 效率不同：同一句话可能产生完全不同数量的文字 token，而语音时长又由发音决定。如果用固定比例交错，英文可能合适，中文、泰语或数字串就可能过快或过慢；如果维护两条独立生成轨，则需要额外同步，容易出现跳词、重复、错误发音和数字歧义。

**ARIA** 全称 Adaptive Rate Interleave Alignment。它把文本 token 与 speech token 放进一个单调交错的单流，并使用样本级全局 speech-to-text token 比例约束任意前缀：在生成过程的任何位置，累计语音/文本 token 比不能超过该样本的目标比例。它不依赖强制对齐工具产生的逐词边界，也不要求所有语言共享固定交错率。

![ARIA alignment](/images/blog/qwen35-omni-aria.svg "图 3：ARIA 根据每个样本的语音/文本 token 比动态交错，避免固定速率在不同语言和数字表达中产生漂移。")

直观地说，文本如果暂时生成得慢，语音轨不会冲到尚未确定的内容前面；文本 token 密集时，系统又能自适应增加语音 token。这个约束简单，却同时作用于训练序列构造和流式调度。报告称它改善自然度、韵律、漏词和数字读法，并允许在任意文本前缀后继续连贯语音。

Qwen3.5-Omni 仍保留 Qwen3-Omni 的多码本 RVQ、轻量 MTP 和因果 ConvNet Code2Wav。ARIA 改的是 Text-Speech 对齐和调度，不是声码器本身。Flash 在单并发下报告的理论首包延迟为音频 235ms、视频 426ms；Plus 为音频 435ms、视频 651ms。Flash 与 Plus 使用不同资源和并行策略，论文也明确提醒这些数值不适合直接横向比较。

## 6. Thinker 后训练：能力融合、音频补课和交互 RL

Qwen3.5-Omni 的 Thinker 后训练比前代更清晰地针对“模态差距”和“交互稳定性”。

**Stage 1 Specialist Distillation。** 团队从 Qwen3.5 base 分别训练文本推理、代码、agent、视觉和音频领域教师，这些教师经过各自的 SFT 与 RL，再生成领域数据蒸馏到统一 omni 模型。它解决的是多能力整合，但教师数据会继承各专家的偏差。

**Stage 2 On-Policy Distillation。** 同一个问题同时有 audio 和 text 条件时，文本输入往往得到更流畅、更完整的回答。作者先让模型在文本条件下生成高质量响应，再把该响应作为音频条件的训练目标，让 audio-conditioned behavior 靠近模型自己更强的 text-conditioned behavior。这个方法比外部教师更贴近当前策略分布，也直接针对“听懂了但回答质量不如文字输入”的差距。

**Stage 3 Interaction-Aligned RL。** 团队构造多轮轨迹，围绕意外语言切换、人格不一致、长对话指令遵循等体验问题设计奖励。它说明 omni post-training 已经从单轮答案正确性转向交互过程质量。不过报告没有公开 reward 组成、权重、人类标注规模和消融，因此外部研究者难以判断每个目标的实际贡献。

Talker 使用四阶段：超过 2000 万小时多语言语音与多模态上下文预训练；高质量子集持续预训练并扩到 64K Talker context；人工多语言偏好对上的 DPO，加规则奖励与 GSPO；最后做轻量 speaker fine-tuning，支持音色克隆和可控表达。

## 7. 实验结果应该怎样读

论文称 Qwen3.5-Omni-Plus 在 215 个音频和音视频理解、推理、交互子任务与 benchmark 上达到 SOTA，并在关键音频任务超过 Gemini-3.1 Pro。这个覆盖面很大，但“215”包含语言级 ASR、翻译子项，不等于 215 个完全独立的数据集。更合理的阅读方式是按能力族看趋势。

文本侧，Plus 与 Qwen3.5-Plus-Instruct 在知识、指令、长上下文、STEM、推理和 agent 上整体接近，论文认为 OPD 与 interaction RL 还改善了指令遵循。视觉侧整体与文本同源的 Qwen3.5-Plus-Instruct 可比，并在部分长短视频任务更强。音视频侧增加了 DailyOmni、WorldSense、AVUT、AV-SpeakerBench、Qualcomm IVD、OmniCloze 和 OmniGAIA 等评测，OmniGAIA 工具使用得分报告为 57.2%。

语音生成方面，RLHF 后的 Plus 在 SEED-TTS test-en 报告 WER 1.26，优于未经过同阶段优化的 0.99/1.26 表格对应设置中的部分基线；模型支持 29 种独立语言的语音生成，若按语言与方言条目统计为 36。语音输入正文表列出 74 种独立语言、113 种语言与方言。这种口径差异应明确保留，不能简单写成“支持 113 种语言生成”。

报告展示了结构化音视频 caption、自动场景切分、时间戳标注、语义打断、音量/语速/情绪控制、voice cloning 和 Audio-Visual Vibe Coding。这些案例说明能力边界，但缺少足够的独立任务成功率和失败类型统计，尤其是 vibe coding 更适合作为新现象而不是成熟 benchmark 结论。

## 8. 局限：规模、时间与行动之间还有距离

第一，报告缺少关键架构细节。Plus 的精确参数规模、专家配置、Hybrid Attention 层比例和大部分训练算力未公开，无法完整复现。

第二，长上下文容量不等于长上下文有效性。支持 10 小时音频或 400 秒视频，只说明能够输入；关键事件召回、跨小时因果和音画冲突推理仍需更严格的 length-controlled 测试。

第三，显式时间戳有明显工程价值，但也可能让模型依赖人工插入的符号。真实流中时间戳漂移、丢帧、异步音视频和不规则采样是否稳健，报告没有充分分析。

第四，ARIA 的动机合理，但报告缺少清晰的多语言逐项消融，例如固定交错、MFA 对齐、双轨方案与 ARIA 在漏词、数字、韵律和延迟上的受控比较。

第五，模型能识别情绪或语气，不代表 agent 的决策会使用这些信号。近期实时语音系统评测已经反复观察到“能描述恐惧或讽刺，却在最终行动中忽略它”的 perception-action gap。原生 omni agent 必须把跨模态证据真正纳入策略，而不只是生成更丰富的 caption。

第六，网页搜索、工具调用和 voice cloning 增加了安全面。提示注入、声音身份冒充、隐私泄露和错误工具操作需要独立的多模态安全训练与评测，不能只依赖文本安全策略。

## 9. 对未来研究的启示

Qwen3.5-Omni 展示了一条清晰路线：长序列结构负责“装得下”，低速率 tokenizer 负责“算得动”，显式时间戳负责“找得到”，ARIA 负责“说得稳”，interaction RL 负责“聊得久”，工具训练负责“做得到”。这六个问题彼此相关，不能只用一个更大的 Transformer 解决。

对 long-video 和 social omni 研究而言，我认为有四个直接方向。

- 把显式 timestamp 与 semantic event boundary 结合，让时间表示从均匀秒数升级为事件坐标。
- 做 audio counterfactual evaluation：静音、错配音轨或交换说话人，检验模型是否真的依赖声音。
- 为 ARIA 类方法设计可解释指标，同时测漏词、数字读法、停顿、情绪、首包延迟和打断恢复。
- 将 agent reward 从最终任务成功扩展到跨模态证据一致性，要求工具调用理由能被对应音频和视频片段验证。

一句话总结：**Qwen3.5-Omni 的进步不只是把 Qwen3-Omni 做大，而是把全模态模型从“共同感知与生成”推进到“长时、实时、可行动”的系统；其中最值得复用的思想，是用显式序列设计解决时间和语音对齐，而不是把所有问题都留给参数规模。**

## 参考资料

- [Qwen3.5-Omni Technical Report](https://arxiv.org/abs/2604.15804)
- [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765)
- [Qwen3-Omni official repository](https://github.com/QwenLM/Qwen3-Omni)
- [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215)
- [Qwen3.5 model announcement](https://qwen.ai/blog?id=qwen3.5)

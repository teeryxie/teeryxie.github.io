# GPT-4o 之后的全双工语音 Agent：时间、状态与交互策略的深度调研

GPT-4o 让大众第一次直观看到语音助手可以快速插话、带情绪地说话、在被打断后继续对话。但“实时语音”与“全双工 spoken agent”不是一回事。前者只要求较快地从用户语音得到回复；后者要求模型在自己输出语音时持续接收用户语音，并在每一个时间片判断：继续说、停下、给一个短促 backchannel、等待更多证据、换话题、修正当前回答，还是把用户的声音当成背景或他人与他人的对话。

这篇文章只研究 GPT-4o 之后的 **full-duplex spoken dialogue** 工作，不再泛泛覆盖所有 omni 模型。我的结论是：该方向已经从“让模型双向流式收发音频”的架构问题，快速演化为一个显式的 agent policy 问题。2024 年解决时间同步与双流表示；2025 年建立 overlap、backchannel 和 interruption 的专门评测；2026 年开始把时机决策、潜在思考、用户流路由、状态预测和强化学习分离出来。

**最重要的观点是：全双工不是“模型可以一边说一边听”的接口特性，而是“模型在不完整、持续变化的社会信号中选择何时行动”的 policy learning 问题。**

![Full-duplex evolution](/images/blog/full-duplex-evolution.svg "图 1：GPT-4o 之后，全双工研究从双流生成走向显式时间策略、状态预测、潜在思考与交互强化学习。")

## 1. 什么才算 full-duplex

为了避免概念泛化，我用一个严格的分层定义。

- **语音 I/O**：系统接收语音、输出语音，但用户说话期间模型不生成，模型讲话期间也不真正听。这仍是 half-duplex。
- **低延迟 turn-taking**：系统在检测到静音后很快回答，可能使用 streaming ASR/TTS，但依然把对话切成完整轮次。
- **可打断系统**：模型讲话时用户可以插话，外部 VAD、关键词或 command module 使其停止。这比 turn-based 更好，但 timing policy 仍可能主要在模型外。
- **full-duplex spoken model**：用户与 agent 有独立、同步的音频流；模型在每个时间片对 agent stream 预测 speech 或 silence，同时条件化于持续到达的 user stream。overlap、backchannel、停顿和抢话被当作训练分布的一部分。
- **full-duplex agent**：除双流生成外，系统还维护会话状态、判断 addressee、规划工具调用、在用户尚未明确提问时决定是否主动发言，并能对自己正在执行的动作进行取消和修正。

因此，“可以 interrupt”不等于“full-duplex”。一个系统若只要听到任何声音就停止，可能在真正插话上反应很快，却会把用户对他人说话、背景电视声和正常的“嗯嗯”都误判为新命令。真正的全双工必须同时学会 **何时 yield** 与 **何时 hold the floor**。

## 2. 延迟不再只有一个数字：stop latency 与 response latency 必须分开

全双工评测中最关键的两个时间指标是：

- **Stop latency**：重叠用户语音开始，到模型停止说话的时间。对于真实 interruption，它应当低。
- **Response latency**：重叠音频结束，到模型下一次发言的时间。对于重新接话，它也应当低。

但这两个指标在不同场景的最优方向不同。Full-Duplex-Bench v1.5 将重叠分成四类：用户 interruption、用户 backchannel、用户与他人说话、背景语音。

- 真正 interruption：应低 stop、低 response，并回答新的意图。
- backchannel：应高 stop，即不要轻易停下；随后低 response，平滑继续原话。
- 用户对他人说话：应高 stop，维持自身语境而不是插入。
- 背景语音：也应高 stop，不能把噪声当成新轮次。

这解释了为什么“最快停止”不是最高分。v1.5 的自动评测里，GPT-4o 在真实 interruption 上表现很强，`Respond=0.78`、stop latency 约 `0.23s`；但同样的高敏感性使它在“用户对他人说话”和背景语音时经常错误让出话轮。Sonic 与 Gemini 对非面向系统的语音过滤更好，却在真实 interruption 上过于保守，常有超过两秒的 stop latency。

![Overlap policy matrix](/images/blog/full-duplex-overlap-policy.svg "图 2：四种 overlap 情况需要不同的 stop 与 response policy；快速让出话轮并不是普遍正确策略。")

这也是为什么 full-duplex 的评价不能只写一个平均 latency。一个均值很低的模型可能不断误停，用户会感觉它“胆小、易受干扰”；一个均值很高的模型又会显得“听不见、占着话不放”。真正要优化的是条件策略：给定当前说话人、语义内容、声学线索和会话状态，选择合适的动作。

## 3. 2024：从 GPT-4o 的目标，到三类开源架构

GPT-4o 的公开延迟是音频输入最低 `232ms`、平均 `320ms`。它定义了产品体验目标，但没有公开可复现的双流架构。开源研究随后出现三条清晰路线。

### 3.1 Moshi：并行双流与“始终监听、始终生成”

Moshi 是最有影响力的路线。它使用 Mimi neural codec，把用户与模型的语音表示为两个并行自回归流；系统 stream 始终生成 speech 或 silence，user stream 始终进入模型。这样不用先强制把连续对话切成轮次，interruption、overlap、interjection 和 silence 都是同一序列建模问题。

Moshi 的 Temporal Transformer 处理时间步，Depth Transformer 在每个时间步内生成多层音频 codebook。Inner Monologue 让 time-aligned text token 作为同帧语音 token 的前缀，保留文本语言模型的语义优势。论文报告理论 `160ms` 架构延迟，低于其引用的人类对话平均约 `230ms` turn gap。

它的真正贡献是把“双方同时说话”从异常情况变为训练常态。不过 Moshi 也暴露了后续方向的核心矛盾：它的互动很自然，但 instruction following、事实问答和复杂推理仍普遍弱于 ASR → 强 LLM → TTS 的级联系统。

### 3.2 SyncLLM：给预训练 LLM 加上 wall-clock

Synchronous LLMs 从另一个角度解决问题：预训练 LLM 本来没有真实时间概念。SyncLLM 将 HuBERT speech unit 切成固定时长 chunk，并在 token 序列中加入同步标记；两个说话人的 chunk 交错排列，模型因此能按真实时钟预测两侧流。

它还提出 latency-tolerant interaction：在当前 user chunk 尚未完全到达时，模型先估计该 chunk 的可能用户响应，再生成下一段 agent audio；收到真实 user chunk 后再替换掉估计内容。实验显示 160–200ms 模拟延迟下表现接近，240ms 开始下降。这种“带预测地等待真实输入”的思路很像控制系统中的短时状态估计。

SyncLLM 的一个重要工程经验是数据配比：论文用 212k 小时合成 spoken dialogue 扩展文本对话，再使用约 2k 小时真实双通道语音做最终全双工适配。它说明真实 full-duplex 数据很少，纯合成数据又会缺少自然 timing，因此需要二者结合。

### 3.3 OmniFlatten / Freeze-Omni：把全双工改写成通用 GPT 序列

OmniFlatten 通过 chunking 和 flattening，将输入语音、输出文本、输出语音按时间交错成一个序列，使用三阶段后训练：模态对齐、half-duplex dialogue、full-duplex dialogue。它的优势是无需改动 GPT backbone，也不依赖昂贵从零预训练。

Freeze-Omni 则选择保护文本 LLM 的既有智能：冻结 Qwen2-7B-Instruct，只训练输入/输出语音接口与 prefix KV 适配，用状态预测实现 duplex。其价值在于证明稀缺 spoken QA 数据下，冻结 LLM 可以减少 catastrophic forgetting；它也非常诚实地公开了实际链路的延迟，不只报理想首 token。

这三类路线形成了早期 full-duplex 的基本设计空间：原生双流 audio LM、时间同步的 text LLM、以及最小化修改的 adapter/flatten 方案。

## 4. 2025：从“能双工”转向“能不能正确处理重叠”

2025 年最大的变化不是又出现一个更快模型，而是社区开始承认“full-duplex”的核心难题是行为质量。Full-Duplex-Bench v1 将 pause handling、turn-taking、backchanneling 和 interruption 变成可复现自动评测；FD-Bench、FLEXI 和 Full-Duplex-Bench v1.5 再把真实 overlap 情景、紧急情况、人机多轮交互和商业 API 统一纳入。

### 4.1 四轴交互能力

Full-Duplex-Bench 的四轴可以视为 full-duplex 最小行为集。

- **Pause handling**：用户句中停顿时模型保持安静。错误响应会让系统显得急躁。
- **Turn-taking**：用户真正让出话轮时，模型及时接话。过慢会产生尴尬空白。
- **Backchanneling**：用户说话时给出短的“嗯”“我明白”，但不能夺走话轮。
- **User interruption**：用户插话时停止旧输出，理解新的意图并恢复对话。

这四个能力不是互相独立。模型若为降低 interruption stop latency 而提高任意语音的敏感度，通常会损害 pause 和 backchannel；若大量输出 silence 避免抢话，又会损害及时 turn-taking。用 token-level SFT 最大似然训练，很难直接表达这种多目标、条件化的偏好。

### 4.2 SALM-Duplex：持续 user stream 与独立 agent stream

SALM-Duplex 用 continuous user input 与 codec agent output 建模两个流，并通过 channel fusion 直接处理同时到达的输入；agent 与 user 使用不同架构，允许针对 agent voice 微调 codec。它用 NanoCodec 的 4 个独立 codebook，并行预测，每秒 12.5 帧，码率约 0.6kbps。

在 UltraChat / Impatient 上，SALM-Duplex 的 barge-in success 为 `83.0% / 94.5%`，Moshi 为 `56.0% / 55.1%`；barge-in latency 为 `0.52s / 0.69s`，Moshi 为 `0.63s / 0.81s`。这些是明确的交互行为指标，而不是只测文本回答。

但论文也指出自己的 first response latency 受数据构造影响：训练时在用户与 agent 之间固定加入 `0.64s` silence，因此 `0.72s / 0.92s` 首响应并不能作为架构下限。这个提醒很重要：full-duplex 训练数据的 timing prior 会直接决定模型“礼貌地等多久”。

## 5. 2026：full-duplex 被重新表述为状态、路由、思考和 policy 的问题

2026 年的工作开始不满足于让模型直接预测下一段音频，而是明确拆分内部机制。

### 5.1 SoulX-Duplug 与 S-MARC：语义 VAD 不是普通 VAD

传统 VAD 只能判断“现在有没有人声”。但全双工 agent 真正需要判断的是：这是用户真的要夺回话轮吗？是一个短促 backchannel 吗？是在自言自语、对旁人说话，还是环境中的人声？

SoulX-Duplug 是一个可插拔 streaming state prediction 模块，它联合 streaming ASR，让文字语义参与状态判断，可理解为 semantic VAD。S-MARC 则把对话建模成从高层 communicative function 到低层行为的 causal hierarchy：用户的意图、模型的回应策略、最终的 speech / silence / backchannel action 存在因果与时间依赖。

两者共同说明：full-duplex 不能只靠声学 energy 或 silence threshold。系统需要一个状态机，但状态机必须由多模态语义和历史上下文驱动。

### 5.2 FLAIR 与 Chronological Thinking：听的时候不应该只预测 PAD

早期双工模型在用户长篇讲话时通常不停预测 silence 或 PAD。这保证了同步，却浪费了“用户还在说”的计算窗口。Chronological Thinking 因而提出在 listening phase 中进行 on-the-fly thought；FLAIR 更进一步使用 latent embedding 递归传递，在听的同时持续累积内部推理，不生成会破坏时间同步的显式 CoT token。

FLAIR 的关键约束是因果性：每一步 latent state 只依赖当前及过去的音频，因此不引入额外等待。论文用 ELBO 风格目标在无显式 thought annotation 下训练。这条路线非常适合 spoken agent：用户可能随时停下，模型必须能立即把已经积累的内部状态转换成答案，而不是先终止一段冗长外显推理。

这也与我的观点一致。full-duplex 不是“生成时仍然接收输入”，而是 **perceive → update belief → anticipate → act** 在同一时间轴上运行。

### 5.3 用户流如何进入 LLM：channel fusion 与 cross-attention 的根本权衡

How Should LLMs Listen While Speaking? 将一个很容易被忽视的架构问题单独拎出来：当模型正在自回归生成自己的回答时，新到达的用户语音应该直接写进主上下文，还是作为外部 memory 由 cross-attention 读取？

**Channel fusion** 直接把 user stream 融入 LLM input。好处是 semantic grounding 强，在 spoken QA 上稳定更好；坏处是如果模型没有及时 stop，重叠用户语音会污染正在延续的 agent sequence，导致回答语义混杂。

**Cross-attention routing** 保留 agent generation context，把 user stream 放在外部 memory。它在 QA 上较弱，但当漏掉 interruption 时，仍更可能保持原回答的连贯性。

这不是一个可被单一 benchmark 决定的优劣问题。面向高风险任务，抗 context corruption 可能更重要；面向问答助手，强语义融合可能更有价值。未来系统很可能需要动态路由：常规发言融合，检测到重叠、歧义或高风险中断时转为隔离通道。

![Duplex routing tradeoff](/images/blog/full-duplex-routing.svg "图 3：用户流直接融合会提高语义 grounding，但在漏掉 interruption 时可能污染生成上下文；cross-attention 更稳健但语义整合较弱。")

## 6. 从 SFT 到 interaction RL：把“何时说”与“说什么”解耦

监督训练优化的是“参考对话的下一个 token 像不像”，但它不直接奖励正确的 floor control。2026 年的 RL 工作正试图改变这一点。

Multi-Faceted Interactivity Alignment 选取人类双通道对话中的短片段，分别为 pause、turn、backchannel 和 interruption 构造 reward，再以 GRPO 优化 Moshi 和 PersonaPlex。它还加入 LLM judge 的内容质量 reward，防止单纯优化 delay 导致回答语义退化。这个细节很重要：如果 reward 只奖励“快”，模型最优策略可能是抢话或输出无意义 filler。

DuplexPO 的论点更直接：自然对话 dynamics 与 instruction/reasoning 的冲突并非必然，真正的问题是把 `when to speak` 和 `what to say` 绑在同一个 token policy 中。它在 dynamics-critical window 内用 factorized reward 优化 `BOS` 和 `EOS` 决策，惩罚漏掉该启动的事件、无根据启动、过长 backchannel、用户接管后不停止、过早或过晚停止，并保留语义能力评测。

这为 full-duplex agent 提供了一个很有前景的训练范式：

- 用大规模语音/文本数据学习内容与语音生成。
- 用真实双通道对话或高质量模拟对话学习基本行为。
- 在边界事件附近单独优化 timing policy。
- 用内容、工具、事实和安全 reward 保留 agent intelligence。

这比把所有目标塞进一套端到端 SFT 更符合问题结构。

## 7. 评测进化：从“回答对不对”到“在什么时机做了什么动作”

![Full-duplex evaluation stack](/images/blog/full-duplex-evaluation.svg "图 4：full-duplex 评测需要同时覆盖内容、时机、行为、语音表现和多轮状态，而非只看转录后的答案。")

目前的评测可以分为五层。

**内容层。** Spoken QA、instruction following、事实问答、ASR 后的 LLM judge。它回答“模型说的内容是否有帮助”。

**声学层。** WER、UTMOS、speaker similarity、prosody。它回答“说得是否清楚自然”。

**局部 timing 层。** stop latency、response latency、FTO、pause duration、backchannel timing。它回答“是否及时”。

**行为层。** interruption 时 respond、backchannel 时 resume、talking-to-others/background 时 hold。它回答“是否做了正确动作”。

**多轮自然度层。** TurnNat 用一个在自然双人对话上训练的 causal turn-taking predictor，计算实际 future activity 的 NLL；高 NLL 表示该 timing 在自然人类对话中更不典型。它还用 TailNLL 关注少数极不自然的边界事件，而不让全局均值掩盖严重抢话或长沉默。

这套分层能避免一个常见误判：某模型在 interruption 上 stop 很快，但它对 backchannel 和背景声也停止。若只看平均 stop latency，它会显得优秀；若加入行为条件和 TurnNat，自然度差异才会出现。

## 8. 数据瓶颈：为什么 full-duplex 比普通语音模型更缺数据

训练 ASR 或 TTS 时，单声道语音就足够。训练 full-duplex model 则需要每个说话人独立音轨，才能知道某个短声音是 backchannel、真正夺回话轮，还是两人重叠。大多数网络播客和公开视频是混音单声道，speaker timing 已经不可逆地混合。

早期研究依赖 Fisher、CallHome 等电话双通道语料，规模只有数千小时。SyncLLM 因此使用海量 text dialogue 合成语音，再以少量真实双通道数据适配；Moshi 使用大规模 speech data 加多流训练；不同论文还会模拟 interruption、插入 silence、TTS 生成对话。这些手段可扩规模，却容易把“过于规则的等待时间”“单一音色”“合成 backchannel”学进 policy。

DuplexChat 是 2026 年一个值得关注的数据工作。它从公开 podcast RSS feed 抓取音频，经过清洗、diarization、双人对话切分、speech separation 和 restoration，构造 speaker-separated corpus：英文 `282,634` 小时、日文 `132,723` 小时，总计约 `415k` 小时。它远超传统双通道语料规模，但也带来新问题：分离误差、podcast 风格与日常对话差异、语言文化 timing 偏差、版权与数据治理。

数据规模提升不会自动带来更自然的全双工。模型仍要知道同一段 overlap 的社会含义。日语中的 aizuchi 频率和重叠比例就可能和英语明显不同，任何只在英语数据上优化的 timing policy 都不应被假定为全球通用。

## 9. 对 GPT-4o 之后工作的一张技术地图

我认为当前全双工工作可以按六个相互依赖的模块理解。

- **Representation**：speech codec、低 token rate、text/audio interleave、双流或多流序列。Moshi、SALM-Duplex 属于这里。
- **Synchronization**：wall-clock chunk、时间 token、延迟补偿、预测当前 user chunk。SyncLLM、OmniFlatten 属于这里。
- **State estimation**：semantic VAD、speaker/addressee state、intent-to-action hierarchy。SoulX-Duplug、S-MARC 属于这里。
- **Concurrent cognition**：边听边更新 latent belief，提前形成可修正计划。Chronological Thinking、FLAIR 属于这里。
- **Routing and memory**：user stream 是融合到主序列还是外部 memory；怎样避免 overlap 污染 current generation。channel fusion 与 cross-attention 对比属于这里。
- **Policy alignment and evaluation**：何时 start、stop、backchannel、wait、resume；怎样评估局部自然度与多轮交互。Full-Duplex-Bench、TurnNat、DuplexPO、Multi-Faceted RL 属于这里。

这张地图也解释了为什么一个端到端模型很难“一次训练好”。语义内容、音色、时间、社会信号和工具行动的 reward 形式完全不同。一个好的 full-duplex agent 最可能是共享感知和记忆、但在 timing policy、content planner 和 speech renderer 上有明确职责划分的系统。

## 10. 仍未解决的关键问题

第一，**addressee grounding**。模型需要区分用户说给它、说给旁人、对电话说、对宠物说，还是电视背景声。v1.5 的结果已显示，这是商业系统与开源模型都不稳定的弱点。

第二，**语义与时机的联合不确定性**。用户还没说完时，模型可能已经有高概率答案，但也可能被后半句完全反转。何时进行 latent planning，何时给 backchannel，何时保持沉默，需要 calibrated uncertainty。

第三，**长期对话状态**。多数 full-duplex benchmark 只测几十秒或有限多轮。真实 assistant 要处理数小时会话、不断变化的任务和环境，同时控制 KV cache、摘要、隐私和遗忘。

第四，**工具与语音动作的原子性**。用户在模型发起支付、控制设备或检索时打断，系统必须决定取消、确认、继续还是询问澄清。语音 stop token 与 tool cancellation token 应属于同一 agent state machine。

第五，**多文化和多语言对话规范**。停顿时长、backchannel 密度、重叠容忍度、礼貌表达和 speaker role 在语言和文化间差异很大。只用英语电话语料训练的 agent 可能在其他场景显得冒犯或冷漠。

第六，**安全与用户控制**。全双工模型越主动，越可能打断、诱导或泄露。用户应能调节 interruption sensitivity、backchannel 密度、主动提醒权限和语音人格；高风险场景应默认更保守。

## 11. 我的结论：全双工是 agentic interaction 的训练场

GPT-4o 之后最深刻的变化，是研究者逐步发现自然对话不等价于更快 TTS。Moshi 证明双流模型能将 overlap 当作自然数据；SyncLLM 证明 LLM 可以被同步到现实时间；SALM-Duplex 证明 user / agent 流可以被独立建模；2025 年 benchmark 证明“停得快”不是总正确；2026 年的 latent reasoning、semantic state prediction、stream routing 和 RL 则证明全双工问题需要显式 policy。

所以，全双工 spoken model 最终应被看成一个实时 agent：它不断从声音中估计环境和意图，在有限时间内更新 belief，选择 speak / wait / stop / listen / tool-use 等动作，并对动作后果负责。自然回复和可打断是它最容易被看见的表面，真正决定能力上限的，是它能否把时间、状态、内容与行动统一起来。

## 参考资料

- [GPT-4o System Card](https://openai.com/index/gpt-4o-system-card/)
- [Moshi](https://arxiv.org/abs/2410.00037)
- [Synchronous LLMs as Full-Duplex Dialogue Agents](https://arxiv.org/abs/2409.15594)
- [OmniFlatten](https://arxiv.org/abs/2410.17799)
- [Freeze-Omni](https://arxiv.org/abs/2411.00774)
- [Full-Duplex-Bench](https://arxiv.org/abs/2503.04721)
- [SALM-Duplex](https://arxiv.org/abs/2505.15670)
- [FD-Bench](https://arxiv.org/abs/2507.19040)
- [Full-Duplex-Bench v1.5](https://arxiv.org/abs/2507.23159)
- [FLEXI](https://arxiv.org/abs/2509.22243)
- [Chronological Thinking](https://arxiv.org/abs/2510.05150)
- [PersonaPlex](https://arxiv.org/abs/2602.06053)
- [S-MARC](https://arxiv.org/abs/2602.11065)
- [SoulX-Duplug](https://arxiv.org/abs/2603.14877)
- [FLAIR: The Silent Thought](https://arxiv.org/abs/2603.17837)
- [User-Stream Routing in Full-Duplex Spoken Dialogue](https://arxiv.org/abs/2605.10199)
- [Synchronization and Turn-Taking in Full-Duplex Speech Dialogue Models](https://arxiv.org/abs/2605.20356)
- [Multi-Faceted Interactivity Alignment](https://arxiv.org/abs/2606.11167)
- [TurnNat](https://arxiv.org/abs/2607.01345)
- [DuplexChat](https://arxiv.org/abs/2607.04941)
- [DuplexPO](https://arxiv.org/abs/2607.07148)

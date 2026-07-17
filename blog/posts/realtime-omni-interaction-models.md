# 实时 Omni 交互模型发展史：从 232ms 语音响应到 Agentic Multimodal Systems

实时语音和音视频交互正在经历一次目标函数的变化。最初的问题是“能不能让机器听完一句话后尽快回答”，随后变成“能不能直接理解声音并生成语音”，再发展为“能不能边听、边看、边说、被打断、主动提醒，并在需要时调用工具完成任务”。

如果只看演示，研究进展似乎可以被概括为响应越来越快、声音越来越自然。但我的观点是：**交互式 omni model 本质上是 agentic 的。自然回复、低延迟、打断和 backchannel 只是外在表现；内部真正需要的是持续感知、状态维护、时机决策、计划、工具使用、记忆更新和安全控制。** 一个只会在用户停顿后迅速朗读答案的模型，最多是低延迟 speech interface，还不是完整的实时 agent。

本文因此按两个优先级整理进展。第一优先级是时效性：用户结束表达后多久能听到第一段有效声音，模型听到打断后多久停止，以及并发和长上下文下是否仍然流畅。第二优先级才是回答内容质量：知识、推理、音视频理解、语音自然度和 benchmark。最后再总结这些系统为什么能快，以及实时 omni 为什么必然走向 agent。

![Realtime omni timeline](/images/blog/realtime-omni-timeline.svg "图 1：实时交互从级联系统、双通道生成模型发展到原生 omni agent 与交互感知 serving。")

## 1. 先定义“实时”：不同论文的延迟数字并不是同一件事

比较实时模型最容易犯的错误，是把论文里的所有毫秒数放进同一张排行榜。实际至少有六种不同指标。

- **Endpoint latency**：用户实际说完，到 VAD 或 turn detector 确认“轮次结束”的时间。太激进会抢话，太保守会增加几百毫秒等待。
- **TTFT**：从模型开始处理到第一个文本 token 出现。它不能代表用户已经听到声音。
- **Audio TTFP / TTFA**：从指定起点到第一段可解码、可播放音频到达。这是最接近体验的模型侧指标。
- **Response-onset latency**：从 turn detection 到形成一段足够送入 TTS 的有效文本或语音前缀。RelayS2S 就采用这种口径，并明确排除了 turn detection、网络和 TTS。
- **Barge-in latency**：用户在模型讲话时开始插话，到模型停止播放的时间。它和首响应延迟是两个不同问题。
- **RTF**：生成一秒音频需要多少秒。RTF 小于 1 只说明可以跟上播放，并不说明第一包来得快。

完整用户延迟更接近：端点判断 + 输入编码 + prefill + 首 token/首 codec + 声码器 + 服务排队 + 网络 + 客户端缓冲。论文常只测其中一段，所以 **160ms 理论架构延迟、234ms 理论首包、700ms 模型延迟和 1.2s 实际部署延迟可以同时成立**。

人类对话的 turn transition 通常只有几百毫秒。Moshi 论文引用的跨十种语言平均值约 230ms。因此，200 到 300ms 是一个重要心理门槛：低于它时系统可以显得“立即接话”，高于约一秒时用户通常能明显感觉到轮次切换。不过，过快也可能意味着模型在用户还没说完时抢答，所以实时系统需要优化的不是单一 latency，而是 latency 与正确发言时机的联合分布。

## 2. 早期阶段：实时语音 Agent 原本就是模块化系统

实时 speech agent 并不是从 GPT-4o 才出现。1990 年代的 Verbmobil 已经研究近实时 speech-to-speech 翻译、增量解析和对话状态；Siri、Alexa 和 Google Assistant 则把 VAD、ASR、NLU、dialog manager、搜索或技能调用、TTS 组织成云端管线。2018 年的 Google Duplex 更进一步，在受限任务里处理停顿、确认、打断和电话预约。

这些系统的重要性不应因为“不是端到端大模型”而被忽视。它们已经包含 agent 的基本骨架：感知用户意图、维护任务状态、决定下一动作、调用外部服务、生成语言并等待反馈。局限是每个模块的信息接口较窄，ASR 转录会丢失语气、笑声、哭泣、背景事件和重叠语音；串行执行也会把每个模块的延迟相加。

换句话说，早期系统 agentic 但不 omni，后来的端到端语音模型 omni 化了，却一度削弱了显式状态、工具和可靠任务执行。今天的研究其实是在重新合并这两条路线。

## 3. 2022 到 2023：生成式 spoken dialogue 的前史

2022 年的 dGSLM，也就是 Generative Spoken Dialogue Language Modeling，是一个重要起点。它在约 2000 小时双通道 Fisher 对话上训练 dual-tower 模型，直接生成两个说话人的语音、笑声和停顿。它证明了模型可以学习自然对话的重叠和 turn-taking，而不必先把一切转成文字。

但 dGSLM 不是在线实时系统，也没有强文本 LLM 的知识和推理能力，主要建模语义语音单元而非完整声学细节。Moshi 后来把它称为 full-duplex 的 proof of concept：对话动力学有了，智能和可部署流式推理还没有。

2023 年 SpeechGPT、AudioPaLM 等工作把离散语音 token 接入 LLM，使模型能理解和生成语音。问题是很多系统采用 Speech → Text reasoning → Speech 的 Chain-of-Modality：先生成完整文本，再开始生成音频。它们拥有更强语义能力，却不适合实时交互，因为用户必须等待整段内部文本完成。

这一阶段确定了后续所有模型都要面对的三角矛盾：**语义质量、声学自然度、低延迟**。纯 audio LM 容易实时和自然，却推理弱；文本 LLM 推理强，却会被中间文本和 TTS 拖慢；完整声学 codec 信息丰富，却产生极长 token 序列。

## 4. 2024：GPT-4o 与 Moshi 把实时交互变成独立赛道

2024 年 5 月发布的 GPT-4o 是第一个大规模公开展示原生文本、视觉、音频实时交互的通用模型。OpenAI 官方报告音频输入的响应时间最低 232ms、平均 320ms，接近人类轮次转换。它能感知语气、直接输出带情绪的声音，并在演示中支持视觉输入和打断。由于架构、训练数据和完整延迟分解未公开，它更像确立产品目标，而不是提供可复现路线。

2024 年下半年，开源研究沿两条方向追赶。

第一条是低延迟半双工或流式 speech-to-speech。Mini-Omni 用 text-instructed speech generation 和 batch-parallel codec generation，让小模型边“思考”边输出语音；LLaMA-Omni 在 Llama-3.1-8B-Instruct 后接 speech encoder、adapter 和因果 streaming speech decoder，在单张 L40 上报告最低约 236ms response latency；GLM-4-Voice 使用 12.5Hz、175bps 单码本 tokenizer 和 streaming-thought 模板交错文本与语音，但首个声码器 block 本身约 0.8 秒。

第二条是真正 full-duplex。Moshi 把用户语音和模型语音表示为独立并行 token stream，模型始终监听，也始终预测语音或 silence。它用 Mimi codec、Temporal/Depth Transformer 和 Inner Monologue，让时间对齐的文本先于同一帧音频 token 出现，兼顾语言质量和流式生成。论文给出 160ms 理论架构延迟，并称其是首个真正在线、全双工的 conversational LLM。

Moshi 的关键突破不是比 GPT-4o 少几十毫秒，而是取消了“轮次必须先切好”这一前提。重叠说话、短促应答、插话和沉默都成为双流序列的一部分。不过，它的知识与指令质量仍弱于大型级联 LLM，后续研究反复观察到端到端 duplex 模型的共同问题：**行为自然，但内容不够强。**

同期的 SyncLLM 和 OmniFlatten 尝试把两个说话人的 speech/text chunk 按时间交错并压成单一 GPT 序列。OmniFlatten 通过 modality alignment、half-duplex learning、full-duplex learning 三阶段，不改 backbone 架构也能学习 overlap 和 turn-taking。Freeze-Omni 则冻结 Qwen2-7B-Instruct，用输入 adapter 和 speech decoder 保留文本智能，并通过状态预测实现打断。

Freeze-Omni 的数字非常有教育意义：从模型判定被打断到首 PCM chunk，统计延迟平均 745ms；语音端点状态另需约 160 到 320ms；加上约 200 到 300ms 网络后，论文估计真实部署约 1.2 秒。它比只报告理论首包的模型“看起来慢”，但测量链条更接近真实体验。

![Latency comparison](/images/blog/realtime-omni-latency.svg "图 2：公开延迟数字按测量口径分层展示；不同起点和终点的指标不能直接组成单一排行榜。")

## 5. 2025：从纯语音实时转向视觉、语音、推理与打断统一

2025 年开始，研究重点从“能不能直接说话”扩展到“能不能在视觉和音频流里保持理解质量”。VITA-1.5 用三阶段训练整合 video、image、text、audio，并报告 A800 上模型端约 700ms、实际 demo 约 1.5 秒；它也明确把个性化和数小时长期记忆列为未解决问题。

Qwen2.5-Omni 给出一套影响很大的 omni 架构：block-wise audio/vision encoder 支持流式 prefill，TMRoPE 对齐音视频时间，Thinker 负责文本理解和推理，Talker 独立生成语音。Talker 用双轨自回归语音 token、sliding-window DiT 和流式声码器降低首包。报告没有给出一个可直接和 GPT-4o 比较的完整端到端毫秒数，但在内容质量上报告 SEED-TTS WER 为中文 1.42%、英文 2.33%、hard 6.54%，并在 OmniBench、AV-Odyssey 等音视频评测取得强结果。

MiniCPM-o 2.6、Baichuan-Audio、Step-Audio、Amazon Nova Sonic、Gemini Live 和 OpenAI Realtime API 则代表产品和开源两侧的扩展。Nova Sonic 通过双向 HTTP/2 stream 同时收发音频，支持 interruption、300K context、tool use 和 agentic RAG；Google Live API 与 OpenAI Realtime API 也把 function calling、会话状态和实时音频放进统一接口。商业系统通常强调体验但不公布可复现的统一首包测量，因此本文不为它们虚构毫秒排名。

Full-duplex 研究也开始建立独立评测。SALM-Duplex 用连续用户流、独立 agent codec 流和 channel fusion 直接建模并发语音。其 UltraChat / Impatient 测试中，barge-in 成功率为 83.0% / 94.5%，停止延迟为 0.52s / 0.69s；Moshi 对应为 56.0% / 55.1% 和 0.63s / 0.81s。SALM 的首轮响应为 0.72s / 0.92s，但作者明确指出训练数据固定插入了 0.64 秒沉默，因此这不是纯架构下限。

Full-Duplex-Bench、FD-Bench 和 FLEXI 开始测 overlap handling、打断、抢话、backchannel 和多轮交互。它们推动社区认识到：单轮 spoken QA 分数无法评价实时交互，模型必须同时避免“该停不停”和“不该停却误停”。

## 6. 2025 下半年到 2026：速度不再只是模型问题

Qwen3-Omni 把 AuT 音频 encoder、MoE Thinker-Talker、多码本 RVQ、MTP 和因果 ConvNet Code2Wav 组合起来。每个 Talker token 对应约 80ms 音频，当前帧残差码本由轻量 MTP 补全，Code2Wav 不再等待 diffusion block。论文报告冷启动理论首包为音频 234ms、视频 547ms；在 36 个 audio / audio-visual benchmark 中，32 个达到开源 SOTA、22 个达到总体 SOTA。更重要的是，它用同规模受控实验论证联合多模态训练没有造成系统性文本和视觉退化。

Qwen3.5-Omni 进一步把 Thinker/Talker 换成 Hybrid-Attention MoE，将上下文扩到 256K，并用 6.25Hz AuT 降低长音频 token 数。Flash 报告音频/视频理论首包 235/426ms，Plus 为 435/651ms。ARIA 用样本自适应 speech-to-text token 比把文本与语音组织成单流交错，减少跨语言漏词、数字误读和节奏漂移。它还加入 specialist distillation、audio-to-text on-policy distillation 和 interaction-aligned RL，把语言切换、人格一致性和长对话指令遵循直接放进后训练。

到这里，模型侧的单请求首包已经稳定进入约 200 到 400ms 区间，新的瓶颈变成并发、长会话和交互调度。LiveServe 的研究说明，普通吞吐优先调度会让模型生成大量用户尚未听到、随后因打断而被丢弃的音频，还会错误淘汰下一轮马上要用的 KV cache。

LiveServe 根据首包、播放 buffer、underrun 风险和 barge-in 暴露来分优先级，并在用户讲话期间预取下一轮 KV。Qwen3-Omni、并发 8 的 ShareGPT audio workload 中，P90 audio TTFP 从 1.38s 降到 0.84s，P95 从 1.45s 降到 0.92s；在另一组 barge-in 0.5 的实验中，P90 从 1.54s 降到 0.91s。注意这并不否定模型论文的 234ms：前者是有并发、长上下文与 KV offload 的服务尾延迟，后者是特定资源下的理论冷启动组件延迟。

## 7. 2026 的新方向：何时说、说什么、想多久开始解耦

ROMA 将连续音视频切成一秒 multimodal unit，用 chunked TMRoPE 对齐，并在 LM head 旁增加两层 MLP speak head。每收到一个 unit，speak head 决定现在是否需要响应；内容生成与发言时机因此分开。ROMA 的平均每 unit 处理延迟约 0.3697s，但它的核心贡献不是首音频，而是同时支持 reactive QA、proactive alert 和 online narration。

这一步非常关键。传统模型等待问题，然后回答；实时 agent 则必须在用户没有显式提问时持续监控，例如“看到锅开始冒烟时提醒我”或“每当实验状态变化就报告”。这已经是 policy learning，而不是普通 next-token prediction。

RelayS2S 从另一侧处理速度与质量冲突。它在 turn detection 后并行运行两个路径：快速 duplex S2S 先生成约五个可用词的 substantive prefix，慢速 ASR → 大 LLM 则生成高质量后续；轻量 verifier 判断是否提交快路径前缀。其 P90 onset metric 为 81ms，接近纯 S2S 的 71ms，同时平均质量保留级联系统的 99%。但该 81ms 从“模型已经决定说话”开始，排除了 turn detection、网络和 TTS，只能说明 speculative handoff 有效，不能写成 81ms 端到端语音响应。

同一时期的 LTS-VoiceAgent、DDTSR、S-MARC、SoulX-Duplug、Chronological Thinking、Silent Thought 和 full-duplex RL 工作都在拆分内部状态：一条路径持续听和更新 world state，一条路径进行潜在推理，一条路径决定 turn-taking，一条路径负责可播放语音。研究问题从“如何更快生成第一个 token”转向“模型能否在说话时继续听，在听时提前想，并在证据变化时修正动作”。

TurnNat、SPEARBench 和新的 full-duplex reward model 也开始评价 timing naturalness、语音流畅度与行为合理性，而不仅是答案文本。2026 年的交互强化学习研究进一步将 overlap、backchannel、hold/shift turn 和打断恢复作为 policy action 来训练。

## 8. 延迟优先的模型比较：我会怎样读这些数字

以下数字都来自论文或官方报告，但测量边界不同。

- **Moshi，2024**：160ms 理论架构延迟；双音频流、始终监听、真正 full-duplex。最接近“模型自身反应下限”，不包含完整网络体验。
- **GPT-4o，2024**：官方最低 232ms、平均 320ms，从音频输入到音频响应的产品级描述；细节不可复现。
- **LLaMA-Omni，2024**：最低约 236ms，在单张 L40 上测从 speech instruction 到 speech response 开始；chunk 越大，质量更高但延迟增加。
- **Freeze-Omni，2024**：模型统计链平均 745ms；计入 endpoint 和网络后估计实际约 1.2s。口径更完整。
- **VITA-1.5，2025**：A800 模型端约 700ms，实际 demo 约 1.5s，包含视觉与语音能力。
- **SALM-Duplex，2025**：首轮 0.72 到 0.92s；barge-in stop 0.52 到 0.69s。它区分了“开始回答”和“听到插话后停下”。
- **Qwen3-Omni，2025**：理论首包音频 234ms、视频 547ms；MoE、多码本和因果 Code2Wav。
- **Qwen3.5-Omni Flash，2026**：理论首包音频 235ms、视频 426ms；Plus 为 435/651ms。规模和部署资源不同，Flash/Plus 不宜只按延迟比较。
- **ROMA，2026**：每个一秒 audio-video unit 平均处理约 369.7ms；这是持续感知吞吐和 timing decision 指标，不是首音频。
- **RelayS2S，2026**：P90 81ms，从 turn decision 到五个可合成词，排除 endpoint、网络和 TTS；展示快慢路径并行的潜力。
- **LiveServe，2026**：并发 8 时 Qwen3-Omni P90 audio TTFP 由 1.38s 降至 0.84s；说明线上尾延迟主要取决于调度、缓存和播放状态，而非单模型参数。

如果要做公平排名，至少要统一硬件、并发、上下文长度、网络位置、VAD、输入时长、首包大小和客户端 buffer。当前公开资料不足以支持一个绝对的“世界最快”结论。更可靠的结论是：**单模型、单请求、受控条件已经接近 200ms；真实并发服务通常仍处于 0.5 到 1.5 秒；打断停止延迟常在 0.5 秒量级；主动发言时机与内容质量远未标准化。**

## 9. 内容质量：实时模型为什么经常“反应快但想得浅”

端到端 speech-to-speech 模型每秒要生成多组声学 token，计算预算大量用于音色、韵律和背景声；文本 LLM 则可以用极低 token rate 表达同样语义。为追求实时，模型还不能等待完整用户输入或长 chain-of-thought。这会产生结构性的质量差距。

不同工作用三种方法弥补。

第一是保留文本语义通道。Moshi 的 Inner Monologue、GLM-4-Voice 的 streaming thought、Mini-Omni 的 text-instructed generation、Qwen Thinker-Talker 和 ARIA 都让文本 token 先于或伴随语音 token，为声学生成提供更稳定的语义骨架。

第二是保护或蒸馏强 LLM。Freeze-Omni 冻结 backbone；Qwen3-Omni 使用 SFT、off/on-policy distillation 和 GSPO；Qwen3.5-Omni 用 specialist teacher 和 audio-conditioned OPD，把文本输入下更强的回答蒸馏到语音输入。

第三是快慢路径并行。RelayS2S 让小型 duplex model 负责前两秒的及时性，大 LLM 负责剩余内容；这和 speculative decoding 的思想类似，只不过“draft”是用户已经能听到的语音前缀，错误提交的代价更高，因此必须有 verifier。

Benchmark 上，Qwen2.5/3/3.5-Omni 展示了强音频、视频与语音生成分数；Qwen3-Omni 报告 36 个音频/音视频 benchmark 中 32 个开源 SOTA，Qwen3.5-Omni 则覆盖 215 个子任务与语言项。RelayS2S 报告保留 99% 级联回答平均质量。SALM-Duplex 在 turn-taking 和 UTMOS 上优于 Moshi。这些结果说明质量差距在缩小，但数据集、judge、提示词和语音转录评分口径差异很大。

更值得警惕的是 2026 年 Real-Time Voice AI Hears but Does Not Listen 的结果：四个领先生产实时语音系统在哭泣、恐惧和讽刺等场景里，经常能在被直接询问时识别声音状态，却在最终决策中仍按字面文本行动。也就是说，**perception 可能正确，policy 却没有使用该信息。** 这正是“omni 感知”与“agentic 行动”之间的缺口。

## 10. 这些模型为什么能这么快：八个共同技术

![Realtime techniques](/images/blog/realtime-omni-techniques.svg "图 3：实时模型的低延迟来自表示、计算图、交互策略和 serving 四层共同设计。")

**1. 降低音频 token rate。** Moshi 的 Mimi、GLM-4-Voice 的 12.5Hz tokenizer、Qwen3 的 12.5Hz RVQ、Qwen3.5 的 6.25Hz AuT 都在减少序列长度。低码率使长音频可进入 LLM，也降低每秒解码步数。

**2. 语义与声学分层。** 主 token 或文本表示内容，残差 codebook 表示音色与细节。MTP、Depth Transformer 或并行 codebook head 在一个时间步内补全声学层，避免大型主干逐个生成所有码本。

**3. 流式 encoder 与 chunked prefill。** 音频和视频 encoder 按 block 输出，不等待完整输入；Thinker 在用户说话时预填上下文，Talker 可以异步接收高层表示。计算被移动到“用户正在说”的时间里。

**4. 文本与语音并行或交错。** Inner Monologue、streaming thought、dual-track generation 和 ARIA 避免先生成完整文本再 TTS。语义 token 稍微领先，语音随即播放。

**5. 因果、可首帧解码的声码器。** Qwen3 用多码本 + MTP + causal ConvNet 替代需要未来 block 的 DiT；LLaMA-Omni 使用因果 streaming decoder；这些设计直接缩短第一段波形的等待。

**6. 让 timing 成为显式 action。** Full-duplex 双流预测 silence/speech，Freeze-Omni 预测状态，ROMA 增加 speak head。模型不再依赖外部 VAD 的单一“用户说完了”事件，而是持续决定 hold、speak、stop、backchannel 或 listen。

**7. 推测式快慢路径。** RelayS2S、LTS-VoiceAgent 和 DDTSR 用轻量前景模型先提供安全前缀或 discourse marker，后台大模型继续推理。它们用并行换取 perceived latency，而不是要求大模型瞬间完成所有思考。

**8. 交互感知 serving。** KV prefetch、playback-aware scheduling、buffer urgency、barge-in cancellation 和异步工具调用决定线上尾延迟。实时系统不能只最大化 tokens/s，因为提前生成 60 秒用户还没听的语音，在用户打断后全是浪费。

这些方法的共性是：**把原本串行的链路并行化，把未来必需的计算提前到用户讲话阶段，把声学细节交给轻量模块，把“何时行动”从内容生成中拆出来。**

## 11. 为什么交互式 Omni Model 本质上是 Agent

![Agentic realtime loop](/images/blog/realtime-omni-agent-loop.svg "图 4：自然语音只是执行器；实时 omni agent 的核心是持续状态估计、时机决策、规划、工具和反馈闭环。")

普通 multimodal QA 的计算形式是 input → answer。实时 omni interaction 则是一个持续运行的 partially observable process：环境不断产生声音、画面和用户动作，模型只能看到到当前时刻为止的 prefix，并且每一刻都要决定是否更新记忆、是否打断自己、是否询问澄清、是否调用工具、是否等待更多证据。

因此它至少需要以下 agentic 能力。

- **持续感知与状态估计**：区分用户、旁人、设备声和模型自身回声；维护谁在说、任务进行到哪、画面发生了什么。
- **时机 policy**：决定 listen、hold、backchannel、speak、stop、resume 和 proactive alert，而不是只输出下一个词。
- **计划与工具调用**：在对话继续的同时查询网页、数据库、日历或控制设备，并处理工具超时和失败。
- **异步执行**：语音前台可以先确认“我来查一下”，后台 agent 继续检索；结果到达后再自然接续，而不是让用户面对沉默。
- **长期与工作记忆**：保留多轮约束、用户偏好、未完成子任务和视觉环境变化，决定哪些 KV、事件或摘要应长期保存。
- **自我监控与修正**：用户一句“不是这个”可能要求取消工具、停止语音、回滚计划并重新理解指代。
- **多模态安全**：声音克隆、摄像头隐私、屏幕敏感信息和工具权限必须在 action 发生前实时检查。

从这个角度看，“回复自然”和“支持打断”只是 agent policy 的两个可观察动作。更强的系统应能在用户还没问时发现风险，在证据不足时保持沉默，在工具执行中汇报进度，在环境变化后修正计划，并能解释行动依赖了哪段音频或视频。

这也解释了为什么只提高 benchmark QA 分数不够。实时 agent 的目标函数至少应包含：首响应延迟、停止延迟、误打断率、抢话率、恰当 backchannel、任务成功率、工具正确率、跨模态证据使用、长期状态一致性、语音自然度和安全违规率。任何单一指标都可能被投机，例如用“嗯”“好的”快速抢到低延迟，却没有提供有效内容。

## 12. 下一阶段最值得研究什么

第一，**thinking while listening**。模型应在用户表达过程中形成可更新的 latent plan，但不能因为早期假设而抢答。流式 latent reasoning、prefix uncertainty 和 reversible planning 会成为关键。

第二，**event-driven omni memory**。音视频流不能全部保留，应围绕说话人切换、语义边界、异常事件和工具状态构建可检索记忆。固定一秒 chunk 只是起点。

第三，**interaction RL**。训练 action 不应只有 speech token，还应包括 wait、backchannel、interrupt、resume、tool call 和 ask-for-clarification，并用人类对话节奏与任务结果联合奖励。

第四，**fast path 与 slow cognition 的安全交接**。快速前缀不能承诺后台无法完成的事情；大模型接管时也不能语义断裂。需要 verifier、uncertainty calibration 和可撤销输出设计。

第五，**audio-visual evidence to action**。模型不仅要描述“这个人声音在发抖”，还要在转账、医疗、驾驶或社交决策中合理使用该证据，并避免把口音、年龄和情绪识别偏差放大成行动偏差。

第六，**端到端服务指标**。研究应公开从麦克风到扬声器的 P50/P90/P99，包含 VAD、网络、队列、工具和客户端；同时报告并发、上下文和 barge-in 条件。只有这样，234ms 与 1.38s 才能被正确解释。

第七，**主动音视频 agent**。ROMA 已经把 when-to-speak 从 what-to-say 中分离，未来还需要 when-to-look、where-to-attend、what-to-store 和 when-to-act。真正的 omni agent 会控制自己的感知预算，而不是被动吞下所有帧。

## 13. 总结

实时交互模型已经走过三个阶段。第一阶段是 ASR → NLU/LLM → TTS 的模块化 agent，任务可靠但信息损失和串行延迟明显。第二阶段是 GPT-4o、Moshi、LLaMA-Omni、Qwen-Omni 等端到端模型，用低码率 token、流式 encoder、语义/声学分层和 full-duplex 序列把受控首包压到约 200 到 400ms。第三阶段正在发生：ROMA、RelayS2S、LiveServe、interaction RL 和新型 full-duplex benchmark 开始直接建模何时说、如何边说边听、如何并行思考、如何调用工具以及如何在并发服务中保持体验。

所以，我对这个方向的核心判断是：**Omni 是感知和表达的统一，Agentic 是状态、时机和行动的统一。只有二者结合，实时模型才会从“会听会说的界面”变成“能够持续参与世界的智能体”。** 未来最重要的突破未必是把 235ms 再压到 180ms，而是让模型在这 235ms 内做出正确的行动选择，并在接下来的数分钟、数小时里保持一致、可控和可靠。

## 参考资料

- [GPT-4o announcement](https://openai.com/index/hello-gpt-4o/)
- [GPT-4o System Card](https://openai.com/index/gpt-4o-system-card/)
- [OpenAI Realtime API](https://openai.com/index/introducing-the-realtime-api/)
- [Generative Spoken Dialogue Language Modeling](https://arxiv.org/abs/2203.16502)
- [SpeechGPT](https://arxiv.org/abs/2305.11000)
- [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037)
- [Mini-Omni](https://arxiv.org/abs/2408.16725)
- [LLaMA-Omni](https://arxiv.org/abs/2409.06666)
- [Synchronous LLMs as Full-Duplex Dialogue Agents](https://arxiv.org/abs/2409.15594)
- [OmniFlatten](https://arxiv.org/abs/2410.17799)
- [Freeze-Omni](https://arxiv.org/abs/2411.00774)
- [GLM-4-Voice](https://arxiv.org/abs/2412.02612)
- [VITA-1.5](https://arxiv.org/abs/2501.01957)
- [Amazon Nova Sonic](https://aws.amazon.com/blogs/aws/introducing-amazon-nova-sonic-human-like-voice-conversations-for-generative-ai-applications/)
- [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215)
- [SALM-Duplex](https://arxiv.org/abs/2505.15670)
- [Full-Duplex-Bench](https://arxiv.org/abs/2503.04721)
- [FD-Bench](https://arxiv.org/abs/2507.19040)
- [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765)
- [Qwen3.5-Omni Technical Report](https://arxiv.org/abs/2604.15804)
- [ROMA](https://arxiv.org/abs/2601.10323)
- [PersonaPlex](https://arxiv.org/abs/2602.06053)
- [RelayS2S](https://arxiv.org/abs/2603.23346)
- [Real-Time Voice AI Hears but Does Not Listen](https://arxiv.org/abs/2606.26083)
- [LiveServe](https://arxiv.org/abs/2606.22983)
- [A Survey of Full-Duplex Spoken Dialogue Systems](https://arxiv.org/abs/2606.19453)
- [TurnNat](https://arxiv.org/abs/2607.01345)
- [Decoupling Conversational Dynamics through Reinforcement Learning](https://arxiv.org/abs/2607.07148)

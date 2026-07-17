# 实时人机交互中的心智理论：模型、评测与 Agent 缺口

当一个实时语音或音视频助手在你犹豫时继续等待、在你只是说“嗯”时不抢话、在你指着屏幕说“把那个发给他”时知道“那个”和“他”分别指向什么、在你改变主意后取消刚才的操作，它表面上表现的是自然对话，内部却在做一件更困难的事：持续估计人的心理状态。

这通常被称为 **Theory of Mind（ToM，心智理论）**：根据可观察到的语言、声音、动作、视线、环境和互动历史，推断他人当下能看到什么、知道什么、相信什么、想做什么、在说给谁听，以及下一步可能怎样行动。对实时 AI 而言，它不是一个“读心”功能，也不是一道 false-belief 选择题，而是一种必须在不完整证据下、随时间更新、并影响行动的状态估计能力。

本文研究大模型时代人机交互中的 ToM，重点放在实时语音、视频和 agent 场景。核心结论是：**静态 ToM benchmark 的高分最多证明模型能在给定故事里产生正确答案；它不能证明模型会在真实互动中维护用户信念、识别共同注意、在恰当时机行动，并在证据变化时修正自己。** 后者是一个部分可观测、带风险与时间约束的 agent 问题。

![Real-time ToM loop](/images/blog/realtime-tom-loop.svg "图 1：实时 ToM 不是单次问答，而是感知、信念更新、交互策略、行动与反馈构成的持续闭环。")

## 1. 先把概念讲清：ToM 不是情绪识别，也不是提示词技巧

心理学中的 ToM 指理解他人拥有与自己不同、且可能与事实不同的心理状态。经典 false-belief 任务中，Sally 看见物体放在 A 处后离开，Anne 把它移到 B 处；问题不是“物体实际上在哪”，而是“Sally 会去哪里找”。回答 A 需要显式区分 world state 与 Sally 的 belief state。

在大模型研究中，这个概念常被扩展得过宽。为了保持讨论可检验，我将实时人机 ToM 分成五层。

- **Perspective taking**：区分用户看到、听到或知道的内容，与模型自己拥有的上下文不同。例如用户看不到后台工具错误，模型不应假定用户知道。
- **Belief tracking**：维护用户对世界的事实判断，以及该判断何时已经过时或错误。它包含一阶 belief，也包含“用户认为助手知道什么”的二阶 belief。
- **Goal and intention inference**：从不完整表达、手势和任务轨迹推断用户希望达成的目标，而不是只执行字面命令。
- **Common ground and reference**：维护双方已经共同确认的信息，解决“这个”“刚才那个”“发给他”等指代，并在不确定时主动澄清。
- **Interaction policy**：将前四层状态用于时机与行动选择，包括 wait、backchannel、ask、speak、correct、cancel、tool-use 和 escalation。

情绪识别只提供其中一部分观测。例如声音发抖、停顿变长或脸部悲伤可以是用户情绪的线索，但它不能单独说明用户希望被安慰、希望安静，还是只是在演示台词。同样，模型能在提示中说出“他以为物体在 A”，并不意味着它会在一段带噪声的多轮语音里记住该人的错误 belief，并据此改变工具操作。

## 2. 为什么实时互动使 ToM 变得更难

离线故事题把人、事件、提问和答案一次性给全。实时交互则有四个本质变化。

第一，**证据是逐步到达的**。用户说到一半时，模型可能已经猜到意图，但后半句可能完全反转前半句。视频中一个人转头、伸手或和旁人说话，也只有在后续帧到来后才有明确含义。模型需要维护概率分布，而不是太早锁死一个解释。

第二，**信念会因为 AI 的行动而改变**。助手说“我已经提交了”会改变用户的 belief；工具失败但模型没有说明，会制造错误共同知识；助手在用户讲话时插嘴，会改变用户对系统是否可靠、是否在听的判断。因此模型不是旁观者，而是社会环境的干预者。

第三，**时间本身是行动的一部分**。面对一句“等一下，我想想”，最好的动作不是立即回答，也不是无限静默。它需要判断这是句中 pause、保留话轮的信号、寻求反馈，还是确实让出话轮。全双工语音 agent 的 interruption、backchannel 和 addressee 判断，都是在线 ToM 的行为外显。

第四，**观测与隐私都不完整**。摄像头、麦克风和屏幕提供丰富社会信号，但系统不应该因为能观察就无限保存、推断或公开用户状态。高质量 ToM 必须伴随 uncertainty、权限边界与可撤销的行动，而不是对人格、情绪或意图做确定性标签。

![Mental-state layers](/images/blog/realtime-tom-layers.svg "图 2：从感知线索到 belief、共同语境和动作策略，实时 ToM 需要分层状态表示与不确定性控制。")

## 3. 从语言题到多模态交互：研究脉络

### 3.1 早期文本 ToM：证明语言模型会做什么，也暴露它不会做什么

2022 年的 [Neural Theory-of-Mind?](https://arxiv.org/abs/2210.13312) 直接检验语言模型在经典 ToM 问题上的表现，并提醒一个至今仍重要的事实：模型可能因为语料中的语言模式而答对，却未必有稳健、可组合的心理状态表征。此后工作将评测从简单一阶 false belief 扩展到多角色、嵌套信念、讽刺、欺骗、错误信息和对话场景。

[ToMBench](https://arxiv.org/abs/2402.15052) 以多种 ToM 任务比较模型的能力与一致性；[HI-ToM](https://arxiv.org/abs/2310.16755) 专注高阶嵌套推理，例如“甲认为乙以为丙知道什么”；[MindGames](https://arxiv.org/abs/2305.03353) 借助动态认识逻辑控制故事状态；[FANToM](https://arxiv.org/abs/2310.15421) 则把多方交互、知情状态和问答形式放在同一压力测试中。

它们的贡献是把“模型会不会答经典题”改造成更系统的诊断。但文本题仍有三个局限。

- 故事通常提供了干净、完整、无歧义的事件记录，真实互动没有。
- 问题直接告诉模型要推断谁的心理状态；真实用户通常不会明确发问。
- 输出答案本身不改变故事；真实 agent 的话语和工具操作会反过来改变人的 belief。

因此，文本 ToM 更适合作为必要但远不充分的能力检查。

### 3.2 视频与音频 ToM：把心理状态放回可见的社会线索

[MMToM-QA](https://arxiv.org/abs/2401.08743) 将 ToM 问答扩展到视频、语音和文本，要求模型利用多模态线索理解角色的知识、意图和情绪。随后 [Theory of Mind's Eye](https://arxiv.org/abs/2406.13763) 讨论视频 LLM 对社会线索的理解，[MOMENTS](https://arxiv.org/abs/2507.04415) 提供更全面的多模态 ToM 评测。它们强调一个事实：从人脸、视线、语气、动作和场景中推断心理状态，不能简单等价于把字幕交给强文本 LLM。

但多模态 benchmark 也很容易被捷径攻破。若答案可由单帧表情、字幕中的显式台词或常识先验推断，模型不必真正对齐音视频时间，也不必维护“谁在什么时候看到了什么”。可靠的测试应包含：去除某一模态后的反事实对照、时间顺序扰动、角色交换、只给 prefix 的流式设定，以及同一情境下不同 belief 的最小对。

### 3.3 从“理解别人”到“与人协作”

真实交互要求模型在行动中用 ToM。[MindCraft](https://arxiv.org/abs/2109.06275) 将 situated collaborative dialogue 中的心理状态建模带入共同任务：参与者看到的环境、持有的知识和下一步计划不完全相同。它比故事题更接近助手要解决的问题，例如模型应知道用户为何问这个、用户是否已经看到某个工具结果、以及此刻是解释还是执行。

[SOTOPIA](https://arxiv.org/abs/2310.11667) 通过社会场景模拟来评估语言 agent 的社交智能，包括目标、关系、约束和多轮互动；后续的 [SOTOPIA-TOM](https://arxiv.org/abs/2605.02307) 聚焦多 agent 对话中的信息管理与 ToM。此类环境开始测量“策略后果”，但它们依然以文本模拟为主，语音时机、画面共同注意、工具延迟和人类真实行为分布尚未被完整覆盖。

## 4. 评测地图：每类 benchmark 到底测到了什么

要避免把不同指标混成一个“social intelligence 分数”，可以按输入、状态、动作与后果四个维度审视。

**经典故事与 false belief。** ToMi、ToMBench 在完整文本故事后提问，适合测一阶 belief 以及事实与 belief 的区分；它们没有持续输入，也没有行动后果。

**嵌套与多方文本 ToM。** HI-ToM、MindGames、FANToM 使用多角色对话和嵌套问题，能够诊断高阶 belief 与 knowledge attribution；但问题通常显式指出了推断对象，感知与时机较简单。

**社交语用与谈判。** NegotiationToM、FANToM 涉及隐含意图、欺骗、承诺和他人目标；多数仍是离线文本，缺少真实语音与动态环境。

**多模态社会理解。** MMToM-QA、MOMENTS、ToM's Eye 使用视频、音频和字幕，覆盖视线、情绪、角色意图与事件时序；它们多为看完再答，不能直接证明在线 prefix 能力。

**Situated 协作。** MindCraft、MuMA-ToM 把环境、对话和共同任务放在一起，能测 perspective、共同知识与计划；环境规模和真人行为多样性仍有限。

**交互式 social agent。** SOTOPIA、SOTOPIA-TOM 在多轮模拟中测信息管理、社会目标和策略结果；主要风险是模拟 agent 的行为分布可能不同于真人。

**实时 spoken interaction。** Full-Duplex-Bench、TurnNat 处理双向音频、overlap 和打断，能测 floor control、addressee 与时机；它们很少直接标注 belief 和 common ground。

表格中最后一行很关键。全双工评测看似不是 ToM benchmark，却已经在测 ToM 的一个实际切面：用户说话时，模型能否判断这段声音是在真正夺回话轮、给 backchannel、对旁人说话，还是背景噪声。若不能区分，模型就无法正确决定 stop、hold、resume 或 respond。

![Benchmark map](/images/blog/realtime-tom-benchmark-map.svg "图 3：现有基准分别覆盖 belief 正确性、社会线索、交互行为与任务后果，但尚少有一个基准同时覆盖在线、多模态、可行动与安全。")

## 5. 当前模型都在怎样做 ToM

### 5.1 纯提示与通用 LLM：最容易展示，也最不可靠

最简单的路线是把故事或对话放入 GPT、Qwen、Claude、Gemini 等通用模型，要求其“逐步思考角色知道什么”。它往往在短、清晰、单轮任务上有效，特别是加入角色表、时间线和显式 belief prompt 后。

它的问题不只是 hallucination，而是状态表示通常不受结构约束。模型可以在同一段对话的不同问题上给出相互冲突的 belief；对事实顺序、姓名和表述方式敏感；在多轮中新证据到来后无法可靠撤销旧结论。对实时系统而言，这种不稳定会变成行为错误，例如错误提醒用户、重复解释已知内容，或在用户只是自言自语时执行工具。

### 5.2 显式 belief tracker：把“谁知道什么”从隐状态拉到外部

[Minding Language Models' (Lack of) Theory of Mind](https://arxiv.org/abs/2306.00924) 提出了可插拔的 multi-character belief tracker，核心思想是不要只依赖 LLM 隐式记忆，而要维护每个角色的事实、观察、belief 和提问时刻。后续的结构化状态、知识图、事件日志和 PDDL-style belief reasoning 也遵循这一原则。

对于实时 agent，一个实用状态可写成：

```text
belief_state[agent] = {
  world: what the agent believes is true,
  user: what the user likely believes,
  common_ground: mutually confirmed facts,
  attention: current addressee / referents / scene,
  uncertainty: confidence and missing evidence,
  commitments: pending tool actions and promises
}
```

这种设计牺牲了一些端到端简洁性，却带来三个好处：可审计、可增量更新、可在行动前进行约束检查。它也符合实际工程边界：语言模型擅长从嘈杂观测中提出假设，状态机或数据库更适合保存可验证事实、权限、工具请求与取消条件。

### 5.3 多模态 LLM：从“描述表情”到“把线索用于行动”还有距离

视觉语言模型、audio-language model 和 omni model 可以把语气、停顿、眼神、场景和语言放到同一上下文中。Qwen2.5/3/3.5-Omni、GPT-4o、Gemini Live、Nova Sonic 等产品或技术报告已展示实时音视频理解、语音交互和工具调用；SocialOmni 则专注音视频社会互动评测。

然而，模型“感知到”不等于“策略使用了”。[Real-Time Voice AI Hears but Does Not Listen](https://arxiv.org/abs/2606.26083) 的核心警告正是：系统可能在被直接追问时识别哭泣、恐惧或讽刺，却仍在最终决策里只按文字字面执行。评估需要测 counterfactual action：当文本不变、语调或视觉线索改变时，模型是否以合理且不过度自信的方式改变下一动作。

### 5.4 实时 spoken agent：ToM 以时机 policy 的形式出现

刚发布的 [Full-Duplex Spoken Agents survey](https://teeryxie.github.io/blog/full-duplex-spoken-agents-post-gpt4o/) 讨论过 Moshi、SyncLLM、SALM-Duplex、semantic VAD、FLAIR、DuplexPO 等路线。它们共同说明，系统必须持续处理用户 stream，并选择 speech、silence、backchannel、stop 或 resume。

这可以被理解为极短时间尺度上的 ToM：模型估计“用户此刻是否希望我让出话轮”“这段声音是对我说的吗”“用户是否仍认为我在执行前一项任务”。当前研究已经有 stop latency、response latency、barge-in success、hold-floor 与 TurnNat 等指标，但尚缺少将这些行为与显式 belief state、reference resolution、工具承诺结合起来的端到端基准。

### 5.5 代表系统现状：它们各自覆盖了哪一段能力链

**通用闭源模型与实时 API。** GPT-4o / Realtime API、Gemini Live、Amazon Nova Sonic 等系统把语音或音视频流、会话状态和 function calling 放入同一实时接口。它们最接近真实人机体验，但完整模型结构、ToM 专项训练数据和统一交互评测通常不公开。因此不能仅凭演示中的自然语气或一次正确打断，推断其拥有稳定的 belief tracking。

**开源或开放报告的 omni model。** Qwen2.5-Omni、Qwen3-Omni、Qwen3.5-Omni、MiniCPM-o 等模型提供音频、视频和文本联合理解，其中部分支持流式输入、直接语音生成和工具式 agent。它们为研究者提供了可测的感知基础，但主流技术报告仍以 ASR、audio QA、video QA、speech generation 和通用 reasoning 为主，ToM 常被分散在社会理解样本里，缺少长期共同语境和行动后果。

**原生 full-duplex 模型。** Moshi、SyncLLM、SALM-Duplex、OmniFlatten、PersonaPlex 以及后续 state-prediction / RL 工作，最直接地建模双方同时说话与话轮状态。它们能够学习 interruption、silence 和 backchannel，却不天然等于理解复杂信念。模型可能在 300ms 内正确停下，但仍不知道用户为何打断、是否改变目标，或前一项工具任务应取消还是保留。

**显式 ToM 与 belief-tracking 模型。** multi-character belief tracker、动态认识逻辑、结构化事件记忆和规划器把角色状态显式化，更容易审计一致性与错误来源。它们在干净文本上较强，却尚未普遍接入低延迟音视频前端。一个关键研究机会是将这类 symbolic / structured state 与 omni encoder、流式 LLM 结合，而不是二选一。

**Social-agent 模型。** SOTOPIA-π、SOTOPIA-RL、Werewolf 或 negotiation agent 通过角色扮演、自博弈和社会 reward 学习策略。这些模型开始优化关系、目标达成和对他人信息的利用，但 reward model 可能偏好表面礼貌、冗长解释或迎合。它们还需要真人校准、跨文化测试以及反操纵约束。

当前没有一个公开模型同时在静态 belief consistency、实时音视频 prefix、全双工 timing、长期人机任务、工具可撤销性和安全上被系统评测。这不是排行榜缺了一列，而是研究对象尚未真正统一。

## 6. 如何把实时 ToM 建成一个可工作的 Agent

我认为不应期待一个单一 next-token model 自然学会所有社会状态。更合理的架构是共享感知、分层记忆、显式策略与可撤销行动。

- **Perception layer**：流式 ASR、speaker diarization、视线/手势/场景事件、语气与重叠语音。输出不是人格判断，而是带时间戳和置信度的证据。
- **Mental-state estimator**：融合当前证据与历史，更新用户 knowledge、goal、attention、affect 和对 agent 的 expectation；每个状态都保留 uncertain / unknown。
- **Common-ground memory**：存储已确认的指代、约束、授权、待办与工具结果。只有明确确认的信息进入高置信共同语境。
- **Interaction policy**：根据风险、置信度和轮次状态选择 wait、acknowledge、ask clarification、answer、stop、resume 或 proactive alert。
- **Task planner and tools**：把自然语言目标变为可取消、可验证、可回滚的 action graph；工具返回后必须更新 user-facing belief，而不是只更新内部日志。
- **Safety governor**：在涉及医疗、支付、隐私、情绪危机或第三方信息时提高澄清门槛，限制基于敏感推断的自动行动。

这个结构不是为了把系统拆得复杂，而是为了避免两类常见错误：一是把“模型猜到用户可能想什么”直接当作授权；二是把“模型曾经听到过什么”误当作双方都确认过的共同知识。

## 7. 延迟与 ToM 的共同优化：快不代表懂，慢也不一定安全

实时系统存在一个典型张力。太快响应可能抢在用户补充前锁定错误 intent；太慢又会让用户以为系统没听懂。ToM 不是把延迟无限压小，而是根据状态选择正确时机。

可以把一个动作写成：

```text
action_t = policy(observation_prefix, belief_state, common_ground, risk, latency_budget)
```

当用户说“把这个发给王老师，呃，等等”，低风险聊天机器人可以轻量 backchannel；有实际发送权限的 agent 应等待、保留草稿、明确指出尚未发送。这里的关键并非模型是否识别了“等等”，而是系统把它解释为用户对当前 action 的 cancel signal，并让这个信号在语音播放、工具队列与持久化状态中原子生效。

同理，full-duplex 的 low stop latency 只有在正确识别 interruption 时才有价值。对背景谈话快速停下会破坏任务；对真实紧急插话继续朗读则造成风险。因此研究应报告条件化的 stop / hold / resume 正确率、错误动作成本和端到端 P50/P90/P99，而不是只宣传某一个“首包毫秒数”。

## 8. 现有评测的六个结构性缺口

### 8.1 题目泄漏与语言捷径

许多 false-belief 样本的写法高度模板化，模型可能记住“离开后物体被移动”的叙事模式。应使用角色、物体、事件顺序和问法的组合泛化，并加入对抗性事实问题来确认模型没有混淆 reality 与 belief。

### 8.2 答案正确不代表内部状态一致

同一故事中，对同一角色先问 belief 再问 action，模型应给出一致结果；新增证据后，应只修改受影响的 belief。评测应有 consistency graph，而不只是逐题 accuracy。

### 8.3 非因果视频评测高估了实时能力

看完整段视频再回答，允许模型利用未来帧。实时 agent 只能使用当前 prefix。应报告 time-to-correct-belief、早期错误动作率和新证据到来后的 belief revision latency。

### 8.4 模拟交互不等于人类交互

agent-against-agent 环境易于规模化，却可能形成共享提示词、文风和策略的闭环。应在真人参与、未见口音、自然打断、环境噪声和跨文化礼貌规范下复测。

### 8.5 缺少行动后果与安全代价

若模型把“用户看起来焦虑”直接解释为“替用户取消订单”，即使它的状态推断有一定合理性，行动仍可能越权。benchmark 必须区分 prediction accuracy、appropriate clarification、authorized action 和 harmful overreach。

### 8.6 忽略模型对用户 belief 的影响

系统的回复会成为用户的新证据。好的评测应检查模型是否正确表达不确定性、是否让用户误以为工具已执行、是否在纠错后修复共同语境，以及用户是否能预测系统下一步会做什么。

## 9. 一个更完整的实时 ToM benchmark 应怎样设计

我建议将未来评测做成连续 episode，而非一组孤立题。每个 episode 同时提供同步音频、视频、屏幕或环境状态、工具事件和用户目标，并隐藏一部分角色私有信息。评价分为五层。

- **State accuracy**：在关键时刻询问 user belief、agent belief、共同知识、referent 和 addressee；使用最小对与 counterfactual 检验。
- **Temporal calibration**：只给不同长度 prefix，测模型何时形成正确但不过早的判断；报告 belief revision latency 与错误承诺率。
- **Interaction behavior**：测 pause、backchannel、interruption、talking-to-others、纠正、澄清和恢复任务的条件策略。
- **Task consequence**：用户目标是否完成，工具是否正确调用/取消，模型是否避免把猜测升级为不可逆操作。
- **Human outcome and safety**：真人对被理解、被尊重、可控性的评价；敏感属性、情绪危机、第三方隐私和操纵风险的专项红队。

一个有价值的分数不应把这些层简单相加。对于医疗、金融或未成年人场景，应让高风险错误具有更高权重；对无法确定的心理状态，适当提问或保持沉默应优于自信猜错。

## 10. 研究方向：从“会答 ToM 题”到“可靠地与人共同行动”

第一，**在线 belief revision**。需要针对流式输入设计数据与目标：何时建立假设、何时标记 uncertain、何时因新证据撤销、何时向用户显式确认。

第二，**共同注意与指代 grounding**。语音中的“这个”、视频中的手势、屏幕中的光标、任务历史中的“刚才那份”必须统一到时间对齐的实体与事件记忆，而不是依赖单次语言猜测。

第三，**multi-party and addressee modeling**。家庭、会议、车内和公共环境都不只有一个用户。系统必须识别谁在对它说话、谁有权限、谁只是背景，而不能把所有声音当作命令。

第四，**ToM-aware interaction RL**。reward 应同时涵盖准确推断、适当不确定性、恰当 timing、任务完成与用户控制；只奖励“让用户满意”会诱导迎合、过度承诺或操纵。

第五，**可解释的状态与隐私保护**。用户应能看到系统正在依据什么共同语境行动，能撤销记忆与授权。状态日志应最小化保存、支持生命周期管理，不能把短暂情绪或旁人谈话变成永久画像。

第六，**跨文化与跨语言有效性**。停顿、视线、礼貌、直接拒绝和 backchannel 的社会意义并不通用。只在英语、单一人群或实验室视频上得到的 ToM 分数，不能直接推广为普遍的人机理解能力。

## 11. 总结

大模型在文本 ToM、视频社会理解和交互模拟上已经积累了大量工作，但这些方向目前仍相对分离。文本 benchmark 擅长测 belief 逻辑，多模态 benchmark 擅长测社会线索，full-duplex benchmark 擅长测时机与话轮，social-agent 环境开始测策略结果。真正的实时人机交互需要把它们连接起来。

我的判断是：**实时 ToM 的目标不是让模型宣称“我理解你的感受”，而是让模型在证据有限时维护可修正的用户状态，在不确定时尊重地澄清，在时间正确的窗口内行动，并永远不把推断替代授权。** 这正是 omni 与 agentic 系统的交叉点，也是下一代实时交互模型最难、也最值得做的能力。

## 参考资料

- [Neural Theory-of-Mind? On the Limits of Social Intelligence in Large LMs](https://arxiv.org/abs/2210.13312)
- [Minding Language Models' (Lack of) Theory of Mind](https://arxiv.org/abs/2306.00924)
- [MindGames: Dynamic Epistemic Logic for ToM Evaluation](https://arxiv.org/abs/2305.03353)
- [MindCraft: Theory of Mind Modeling for Situated Dialogue](https://arxiv.org/abs/2109.06275)
- [FANToM: A Benchmark for Stress-testing Machine Theory of Mind in Interactions](https://arxiv.org/abs/2310.15421)
- [HI-ToM: Higher-Order Theory of Mind Benchmark](https://arxiv.org/abs/2310.16755)
- [SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents](https://arxiv.org/abs/2310.11667)
- [ToMBench: Benchmarking Theory of Mind in Large Language Models](https://arxiv.org/abs/2402.15052)
- [MMToM-QA: Multimodal Theory of Mind Question Answering](https://arxiv.org/abs/2401.08743)
- [Theory of Mind's Eye: Reading Minds with Multimodal Video LLMs](https://arxiv.org/abs/2406.13763)
- [MuMA-ToM: Multi-modal Multi-Agent Theory of Mind](https://arxiv.org/abs/2408.12574)
- [Position: Theory of Mind Benchmarks are Broken for Large Language Models](https://arxiv.org/abs/2412.19726)
- [A Survey of Theory of Mind in Large Language Models](https://arxiv.org/abs/2502.06470)
- [SOTOPIA-TOM: Evaluating Information Management in Multi-Agent Interaction](https://arxiv.org/abs/2605.02307)
- [NegotiationToM: Stress-testing Theory of Mind in Negotiation](https://arxiv.org/abs/2404.13627)
- [Full-Duplex-Bench](https://arxiv.org/abs/2503.04721)
- [TurnNat: Evaluating Turn-Taking Naturalness](https://arxiv.org/abs/2607.01345)
- [Real-Time Voice AI Hears but Does Not Listen](https://arxiv.org/abs/2606.26083)
- [MOMENTS: A Comprehensive Multimodal Benchmark for Theory of Mind](https://arxiv.org/abs/2507.04415)

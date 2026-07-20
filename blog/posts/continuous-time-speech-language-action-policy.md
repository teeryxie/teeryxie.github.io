# 交互式 Omni Model 的本体：连续时间 Speech-Language-Action Agent Policy

今天的实时语音模型已经能做到快速接话、自然停顿、附和、被用户打断后停止，以及用不同音色和情绪说话。如果只看产品演示，很容易把研究目标理解成“让 Speech-to-Speech 模型更像真人”。但这些现象只是外在行为，不能解释一个交互式系统为什么在复杂任务里仍会失败。

真正的问题是：用户表达在不断变化，语音和视频只提供部分观测，工具调用有延迟和副作用，模型自己的回答会改变用户认知，而且用户随时可能纠正、撤销或增加约束。模型必须持续更新对用户目标的判断，在信息不足时保留多个假设，决定继续等、澄清、回答还是提前检索；它还要一边说话一边监听新证据，跟踪工具执行、长期记忆和用户实际听到的播放位置，并在延迟收益、错误风险和任务价值之间做选择。

因此，我对交互式 Omni Model 的定义是：

**它不是带语音输入输出的聊天模型，而是运行在连续时间、部分可观测环境中的 Agent Policy。自然接话、倾听、打断和回复只是该 policy 的可观察动作。**

更完整的研究问题可以写成：如何训练一个以音频、视频、工具事件、记忆读取和播放状态为观测，以语音、对话控制、工具调用、工具撤销和记忆操作为动作的流式策略模型？

![Continuous-time policy loop](/images/blog/continuous-agent-policy-loop.svg "图 1：交互式 omni model 持续接收人、环境、工具和播放事件，在统一状态上并行决定语音、控制、工具和记忆动作。")

## 1. 不是消灭模块，而是消灭推理关键路径上的串行同步点

传统级联语音 Agent 的延迟通常来自一组同步 barrier：

```text
等待用户说完
→ 等待 ASR final
→ 等待 LLM 完整规划
→ 等待工具返回
→ 等待 TTS 首包
→ 开始播放
```

问题不只是组件多，而是每一阶段必须等上一阶段“最终完成”。用户说话期间，大模型没有开始建立计划；工具执行期间，前台只能沉默；用户在 TTS 播放时补充信息，旧 action 仍可能继续执行。

搜索、数据库、CRM、日历、支付系统和持久化记忆不可能全部塞进模型参数。工具需要独立权限、事务、日志和扩缩容。因此合理目标不是物理上做成一个不可拆分的程序，而是：

**Policy monolithic, Execution modular。**

这里的 monolithic 指感知、belief update、轮次控制、计划更新、工具调度和回复生成共享同一个因果流式状态，不再由多个互不知情的阶段串行做决定。Execution 仍然模块化：搜索和数据库异步返回事件，写操作保留权限与事务边界，长期记忆由外部存储负责。模型在等待事件时仍可以继续听、继续想、给出简短反馈或修正计划。

[Asynchronous Tool Usage for Real-Time Agents](https://arxiv.org/abs/2410.21620) 已用事件驱动有限状态机展示异步工具与语音前端如何并行；[VoxMind](https://arxiv.org/abs/2604.15710) 则让辅助 Agent 异步管理大量工具，使主 spoken model 不必同步遍历工具集。它们说明 modular execution 可以保留，但事件必须进入实时 policy，而不是成为阻塞调用。

## 2. 从 Speech-to-Speech 走向 Speech-Language-Action

最接近这一方向的公开架构是 [DuplexSLA](https://arxiv.org/abs/2605.20755)。它将用户音频、助手音频与结构化 action 放到共享 `160 ms` chunk 时钟上。用户音频以 `80 ms` stride 提供因果特征；助手每个 chunk 生成一个文本锚点和四个 `40 ms` 音频 token；独立 action channel 每个 chunk 最多生成 10 个文本 token，用于延迟转录、短 planning、turn-taking 标签与 JSON 工具调用。

同一个约 7B backbone 因而可以在持续输出助手语音时，另外发出 `interrupt`、`backchannel`、planning fragment 或工具调用。action 不需要占用语音 token 的位置，也不必等待助手停止说话。

可以把目标模型抽象成：

```text
Observation o_t
  user audio / video
  tool result events
  memory read results
  playback position
  previous actions
          │
          ▼
  Causal streaming state h_t
  intent · belief · plan · dialogue · tool state
          │
     ┌────┼──────────┬───────────┐
     ▼    ▼          ▼           ▼
  Speech  Control    Action      Memory
  tokens  policy     channel     policy
```

每个时间片的状态更新与策略输出可以写成：

```text
h_t = f_theta(h_(t-1), o_t, a_(t-1))

(control_t, action_t, speech_t, latent_t) = pi_theta(h_t)
```

这里不要求所有输出使用同一 token rate。音频可能以 12.5Hz 或更高频率生成；control head 每 80–160ms 做一次 listen / start / continue / backchannel / yield / abort 决策；action channel 只在出现语义事件时输出；belief 和 plan 大部分时间可以保留为不可见 latent state。

关键设计是：**共享统一隐状态，使用不同速率、不同职责的输出通道。**

DuplexSLA 目前应被视为重要架构参考，而不是立即可复现基线。截至本文撰写时，其仓库已发布技术报告和 demo，README 仍明确标记 inference code、checkpoint、deployment recipe 与 DuplexSLA-Bench harness/data 为待发布。

## 3. 为什么必须有独立 Action Channel

假设助手正在说：“我正在帮您查询最近的订单……”用户插入一句：“查耳机那一单。”理想系统不需要先停止语音、生成完整转录、再把 JSON 插入同一个输出序列。它可以继续自然地给出前台反馈，同时 action channel 立即提交只读查询：

```text
speech:  "我先帮您查一下。"
control: CONTINUE
action:  CALL search_orders(item="headphones")
```

更困难的情况是用户改口：

```text
t = 0.80s
user:    "帮我查一下昨天买的耳机"
belief:  intent=search_order, date=yesterday, confidence=0.62
control: LISTEN
action:  CALL_SAFE_READ(search_orders, date=yesterday)

t = 1.44s
speech:  "我先帮您查一下。"
action:  WAIT_TOOL(search_1)

t = 2.08s
user:    "不对，是前天。"
belief:  date=day_before_yesterday, old_hypothesis=invalid
control: YIELD_AND_REVISE
action:  CANCEL(search_1)
action:  CALL_SAFE_READ(search_orders, date=day_before_yesterday)
speech:  stop old clause, then repair naturally
```

这说明“被打断”和“成为 Agent”不是两个独立能力。它们共享同一个判定：**新的用户证据是否使当前 belief、plan、speech commitment 或 external action 失效？**

只训练 `INTERRUPT` 标签最多让模型停下声音。真正的 Agent 还必须学习 slot revision、plan invalidation、tool cancellation、memory correction、speech repair 和 execution-state reconciliation。

独立 action channel 还有三个工程优势。第一，结构化 JSON 不会破坏连续音频生成。第二，每个 action 有明确语义触发时间，可测 earliest legal action latency。第三，action 可以进入独立 grammar 和权限检查，不必信任任意自然语言 token 直接驱动业务系统。

## 4. 级联 Agent 不应被丢弃，而应成为训练 Teacher

现有级联 Agent 虽然推理慢，却常常拥有更强 ASR、更强文本推理、成熟工具系统、RAG、长期记忆、显式 workflow 与可解释日志。它很适合离线生成专家轨迹。

传统蒸馏通常只保留：

```text
完整用户问题 → 最终答案
```

连续时间 policy 需要的 supervision 是：

```json
{
  "timestamp_ms": 1440,
  "observed_prefix": "帮我取消昨天",
  "belief": {
    "intent": "cancel_order",
    "date": "yesterday",
    "order_id": null,
    "confidence": 0.62
  },
  "control": "LISTEN",
  "candidate_action": null,
  "earliest_legal_action": "need_order_identifier",
  "response_plan": "wait_or_clarify"
}
```

Teacher 可以读取完整 transcript、未来用户纠正和最终工具结果，构造高质量 posterior trajectory；student 在训练和推理时必须保持严格因果，只能看当前时间之前的音频和事件。未来信息用于生成 target，不得泄漏到 student observation。

### 4.1 显式轨迹蒸馏

Teacher 输出 partial intent、slot belief、候选 action、最早合法执行时间、是否可撤销、等待或澄清理由、计划修订和 memory operation。Student 以多任务分类、结构化 next-token prediction 或 compact planning token 学习。

这种监督适合可审计的客服流程，也能直接计算 action precision、slot consistency 和 rollback correctness。缺点是人工 schema 有上限，Teacher 的自然语言 rationale 也可能冗长或不稳定。

### 4.2 Hindsight latent distillation

[FLAIR / The Silent Thought](https://arxiv.org/abs/2603.17837) 让监听阶段的 latent embedding 递归传递，在不输出外显 CoT 的情况下持续形成内部状态。其 global-aware expert 可利用完整对话提供 posterior latent target，再蒸馏给严格因果的 student；推理时丢弃 expert。

这条路线给出一个重要训练原则：**训练阶段可以使用强级联、未来信息和重计算，推理关键路径只保留已经蒸馏的 causal policy。**

显式轨迹和 latent distillation 不需要二选一。业务 action、权限和 commit 条件应显式监督；难以穷举的语义假设、对话节奏和潜在计划可以由 latent target 承担。

![Teacher-student trajectory](/images/blog/continuous-agent-distillation.svg "图 2：离线 Teacher 可观察完整对话和工具结果以生成后验轨迹，在线 Student 只能使用因果前缀学习连续 belief、control 和 action。")

## 5. 不能等用户说完才推理：训练 think while listening

实时 Agent 的大部分可感知延迟并不来自 vocoder，而来自“用户说完以后模型才开始理解”。如果一个 8 秒请求直到第 8 秒才开始建立计划，那么即使 TTS 首包只有 200ms，工具仍会迟到数秒。

[Can Speech LLMs Think while Listening?](https://arxiv.org/abs/2510.07497) 发现文本空间 reasoning fine-tuning 可使多流 Speech LLM 在一组语音推理任务上的平均准确率提升 `2.4×`。该工作使用 question completeness 判断何时开始推理，再用基于 rejection sampling 的 DPO 推动准确率-延迟 Pareto frontier，报告在不损失准确率时降低约 `70%` 推理延迟。

[SHANKS](https://arxiv.org/abs/2510.06917) 将语音切成连续 chunk，用户仍在说话时生成不可见 reasoning。数学讲解场景中，它比无 reasoning 的 interruption baseline 提高 `37.1%` 错误打断准确率；工具对话中，`56.9%` 的必要 API 调用在用户说完之前完成。

[Stream RAG](https://arxiv.org/abs/2510.02044) 进一步训练 spoken model 在用户讲话期间预测检索 query。AudioCRAG 上，准确率从 `11.1%` 提升到 `34.2%`，工具延迟降低 `20%`。这说明提前 action 不只改善速度，也可以通过更早 grounding 改善答案。

但不应在每个音频 chunk 后生成数百 token 的显式 CoT。它会与音频和 action 竞争计算，partial input 容易诱发错误假设，新证据到来后长 reasoning 难以撤销，而且生成链无法及时抢占。

更合适的是三档状态更新：

- **高频 latent belief**：每 80–160ms 更新意图分布、slot 置信度、用户是否可能继续、当前计划是否失效、哪些工具参数已具备。
- **低频 action token**：只在事件发生时输出 `CALL_TOOL`、`CANCEL_TOOL`、`ASK_CLARIFY`、`MEMORY_WRITE`、`PREPARE_COMMIT`、`COMMIT`。
- **更低频 compact plan**：输出 `need_order_id`、`safe_to_prefetch`、`await_confirmation`、`tool_result_conflict` 等短锚点，而非长自然语言独白。

## 6. 训练数据的核心不是“对话”，而是时间对齐的 Agent 轨迹

普通客服文本日志缺少连续时间 policy 最关键的信息：谁何时开始说、重叠发生在哪、停顿是否意味着结束、工具最早何时可调用、用户何时纠正参数、助手哪些内容已经真正播放、哪些只在 server buffer 中生成。

完整样本至少需要：

- 用户独立音频轨道与时间戳；
- 助手生成音频轨道；
- 助手实际播放进度与被取消区间；
- 用户增量 transcript 或语义特征；
- 助手文本锚点和 speech token；
- listen / start / backchannel / yield / abort 控制标签；
- belief 与 plan snapshot；
- tool call、result、timeout、cancel、commit 时间线；
- memory read / write / correction；
- action 可撤销性、权限和确认要求。

中文 full-duplex 数据基础正在改善。[BayLing-Duplex](https://arxiv.org/abs/2606.14528) 从 GLM-4-Voice 初始化，只加入少量特殊状态 token，将 listen、speak 和 stop 转为普通 next-token prediction。论文使用 `400K` 全双工对话做 SFT，再进行轻量 DPO；在 InstructS2S-Eval 上报告 `92%` turn-taking success 与 `100%` interruption success，同时保持通用问答能力。

BayLing-Duplex 解决的主要是全双工 dynamics，不是完整 agentic tool policy。它证明成熟 turn-based SpeechLM 可以用相对可控的数据规模改造成 native duplex，但 action、rollback 和业务状态仍需新增轨迹。

[SmoothConv / DuplexConv](https://github.com/qualialabsAI/SmoothConv-DuplexConv) 则提供约 2100 小时自然中文多通道对话。SmoothConv 为约 100 小时专家毫秒级标注，DuplexConv 为约 2000 小时 LLM 辅助标注，覆盖 tutoring、social chat、overlap、backchannel、interrupt、pause 与 turn transition。它们适合学习自然交互 prior，却仍需要和自建客服 action timeline 配对。

DuplexSLA 的规模说明从零训练通用 backbone 成本很高：技术报告披露 continued pretraining 约 50 万小时音频加 192 万文本样本，其中 duplex dialogue 约 32 万小时；post-training 约 5 万小时，其中 interaction control 约 3.6 万小时、tool-call 约 1.4 万小时。对大多数团队，更现实的路线是从 BayLing-Duplex、GLM-4-Voice、Moshi 或 PersonaPlex 初始化，只学习新 action/control adapter 与领域轨迹。

## 7. 六阶段训练方案

### 7.1 保留原模型智力

从强文本或 SpeechLM 初始化，不从零训练。冻结 speech tokenizer、speech decoder 与 vocoder，先训练 backbone LoRA、action embedding、action decoder 和 control head。持续混入文本 instruction、spoken QA 和原始语音生成数据，防止 full-duplex 数据将模型训练成“反应自然但内容变弱”。

[Moshi](https://github.com/kyutai-labs/moshi) 已提供完整推理栈和独立 fine-tuning repository。其 Mimi codec 使用 12.5Hz 表示，理论延迟 160ms，L4 上实际整体延迟可低至约 200ms，适合英文原型。中文则可从 BayLing-Duplex 或 GLM-4-Voice 衍生。

### 7.2 学习连续双流与自然 dynamics

使用用户音频、助手历史音频、助手目标音频、对齐文本、静音与 overlap 区间训练 simultaneous listen-and-speak。目标是学会自然 silence、开始和停止、backchannel、用户抢话与背景声音过滤。

silence 占比通常极高，均匀交叉熵容易让模型通过“永远沉默”取得低损失。需要降低 silence 权重、提高稀有 control state 权重，并按事件窗口采样。普通长静音用于校准，真正影响策略的边界事件需要过采样。

### 7.3 加入独立 Action Channel

先限定领域，不立即做万能 Agent。客服原型可只提供 10–20 个 API、5–10 种 memory operation 和 8–12 个 control action，例如订单查询、物流、取消、退款、地址修改、创建工单、身份验证和转人工。

每个 action 应标注语义触发时间、最早合法执行时间、最晚有用时间、参数来源、依赖关系、是否只读、是否可撤销、是否需要确认。action decoding 使用 schema / grammar constraint，权限校验放在执行层。

### 7.4 蒸馏边听边想的 belief 与 plan

让强级联 Teacher 在每个关键 chunk 输出 compact state，或由 global-aware expert 提供 latent target。训练 student 判断信息是否充分、当前意图是否稳定、什么可以提前读、什么必须等待、最新证据是否推翻旧 action。

这里本质上是 optimal stopping：现在执行得到的延迟收益，是否大于错误 action 的期望代价。不是“越早调用越好”，而是“在最早合法且风险可接受的时刻调用”。

### 7.5 在 critical windows 做 interaction RL

不要一开始对完整十分钟通话做 RL，长序列的时间信用分配会淹没关键动作。优先采样用户结束前后、句中停顿、backchannel、barge-in、参数纠正、工具最早可调用、结果返回和 irreversible commit 前的短窗口。

[DuplexPO](https://arxiv.org/abs/2607.07148) 正是将 when to speak 与 what to say 解耦，在高影响窗口上用 factorized reward 和 GRPO-style objective 优化 turn initiation、backchannel、yielding 和 participation，同时保留推理与指令能力。这一思想可扩展到 action timing：单独奖励 earliest legal tool call、correct cancellation 和 safe commit。

### 7.6 用交互模拟与 DAgger 纠正 exposure bias

让 student 自己 rollout，模拟用户主动制造中途改口、长停顿、backchannel、强打断、多请求、参数纠正、工具失败、超时、改变主意、背景人声和播放滞后。Student 进入错误状态后，由强 Teacher 对当前轨迹重新标注，再聚合回训练集。

纯 teacher forcing 只教模型在正确历史上行动；真实部署最需要的是进入错误历史后恢复。DAgger-style data aggregation 能把“失败后的修复轨迹”变成训练分布。

![Six-stage training](/images/blog/continuous-agent-training-stages.svg "图 3：从保留基础智力、学习双流 dynamics，到 action channel、因果蒸馏、局部 RL 和交互式数据聚合的六阶段路线。")

## 8. 多目标损失：内容、时机与回滚必须分开监督

可以使用以下概念性组合：

```text
L = lambda_audio    * L_audio
  + lambda_text     * L_text
  + lambda_control  * L_control
  + lambda_action   * L_action
  + lambda_belief   * L_belief
  + lambda_latent   * L_latent
  + lambda_rollback * L_rollback
```

`L_audio` 负责语音 token、韵律和流畅度；`L_text` 保留原语言能力；`L_control` 学习 LISTEN / START / CONTINUE / BACKCHANNEL / YIELD / ABORT，并需要类别重加权；`L_action` 学习 grammar-constrained 工具和记忆操作；`L_belief` 监督 intent、slot、identity verification、action readiness 和 confirmation requirement；`L_latent` 将 posterior expert state 蒸馏给 causal student。

`L_rollback` 是现有 spoken model 经常缺少的一项。它专门训练：删除用户未听到的助手尾部承诺、取消依赖旧参数的工具、保留仍有效的上下文、修正错误 memory，并在恢复对话时不引用已经生成但没有播放的内容。

## 9. Playback-aware rollback：模型历史不等于用户历史

流式系统至少有三条不同时间线：模型已生成的位置、server 已发送的位置、扬声器已播放的位置。用户打断时，模型可能已经生成后续 3 秒音频，但用户只听到前 500ms。

如果模型把全部 generated text 当作 common ground，就会产生典型错误：用户没听过某项确认，模型却说“正如我刚才解释的”；未播放的承诺进入长期记忆；已经在 buffer 中取消的语句继续影响后续计划。

因此 observation 必须包含 playback event：

```text
generated_until = 8.4s
sent_until      = 7.2s
played_until    = 6.1s
interrupt_at    = 6.3s
```

rollback policy 应将历史拆为：用户已听到的 committed speech、尚在播放的 revocable speech、从未播放的 discarded speech。只有 committed speech 默认进入共同语境；其余内容在被打断后应撤销或重新表达。

这也是 serving 和 model policy 必须共享状态的原因。只在前端停止播放器、却不通知模型实际播放位置，无法实现语义级中断恢复。

## 10. 对不可逆工具使用两阶段提交

只读搜索、缓存和检索可以在 partial intent 稳定后提前执行；退款、支付、删除、取消订单和发送消息具有副作用，不能为了低延迟直接 commit。

更合理的是：

```text
safe read:
  CALL_EARLY → USE_OR_CANCEL_RESULT

side-effect action:
  PREPARE → USER_CONFIRMATION → COMMIT
                         └────→ ABORT
```

例如用户说“取消我刚买的那个”，模型可以 `PREPARE_CANCEL(recent_order)` 并读取订单详情，再口头确认具体商品。用户确认后才 `COMMIT_CANCEL`；用户说“等一下”则 `ABORT_PREPARED_ACTION`。

[Speculative Interaction Agents](https://arxiv.org/abs/2605.13360) 使用类似思想：只读工具可提前执行，有副作用的写操作等待 commit point；未执行的 tool call 可 modify/remove，依赖被取消结果的子 action 也级联取消。论文报告云端模型获得 `1.3–1.7×` 加速，小型 Qwen/Llama 模型获得 `1.6–2.2×` 加速，同时承认存在轻微准确率代价。

两阶段提交使“提前思考”和“提前造成现实后果”解耦，是 agentic realtime system 必须具备的安全边界。

## 11. 强化学习不能只奖励快

概念性 reward 可以写成：

```text
R = R_task + R_tool + R_semantic + R_safety
  - alpha * response_delay
  - beta  * yield_delay
  - gamma * false_interrupt
  - delta * premature_action
  - eta   * user_effort
  - mu    * compute_cost
```

延迟不应只从“用户说完”开始计算，而应从 `t_ready` 开始：模型已经获得足够语义信息、允许发起该 action 的最早时间。若用户说“帮我查明天下午两点北京到上海的高铁，最好靠窗”，听到时间与路线后即可提前搜索；“最好靠窗”作为后续过滤。但购票必须等待完整约束、身份和确认。

early action metric 也不能只算越早越高。需要同时报告 early tool-call precision、无效 speculative call 成本、撤销正确率和不可逆误执行率。否则模型可能通过对每个 partial prefix 都乱发 API 来获得低平均 latency。

## 12. Benchmark 应从局部交互走向完整 Agent trajectory

[Full-Duplex-Bench v3](https://arxiv.org/abs/2604.04847) 已开始测试真实人类音频中的五类 disfluency 与四个领域的 chained API。六种系统中，GPT-Realtime 的 Pass@1 为 `0.600`；Gemini Live 3.1 延迟最快，为 `4.25s`，但 turn-take rate 只有 `78.0%`；Whisper → GPT-4o → TTS 级联保持完美 turn-take，却有 `10.12s` 最高延迟。自我修正和困难多步 reasoning 是所有系统的共同弱点。

该结果说明评测方向正在从“能否停止说话”转向“disfluency 下是否仍能正确使用工具”。但连续时间 Agent 还需增加以下指标：

- 相对于 `t_ready` 的 action trigger latency；
- response、yield、cancel 的 P50/P95/P99；
- early tool-call precision 与无效查询成本；
- plan invalidation 和 slot revision 正确率；
- speculative action cancellation success；
- playback-aware common-ground consistency；
- user-unheard content reference rate；
- irreversible action false-commit rate；
- tool timeout / failure 后 workflow recovery；
- 多轮 task success 与 belief consistency。

此外必须按 action 风险分层。搜索提前 500ms 和退款提前 500ms 不是同一收益；benchmark 应给副作用、可逆性与错误成本显式权重。

![Risk-latency frontier](/images/blog/continuous-agent-risk-latency.svg "图 4：实时 Agent 的目标不是无条件最早行动，而是在可逆性、置信度和任务风险约束下逼近 earliest legal action。")

## 13. 一条现实可做的中文研究路线

直接复现 DuplexSLA 的 50 万小时 CPT 和 5 万小时 post-training 对多数研究团队不现实。更可行的最小系统是：

- 以 BayLing-Duplex / GLM-4-Voice 为中文基线，英文可使用 Moshi / PersonaPlex；
- 用 SmoothConv / DuplexConv 稳定自然 full-duplex dynamics；
- 限定 2–3 个客服领域和 10–20 个工具；
- 新增轻量 action/control adapter，不改 tokenizer 和 vocoder；
- 用成熟级联 Agent 生成约十万级、时间对齐的因果轨迹；
- 先做 SFT 与 latent distillation，再做 critical-window DPO/GRPO；
- 最后通过 user simulator、延迟注入与 DAgger 训练 rollback 和异常恢复。

[PersonaPlex](https://arxiv.org/abs/2602.06053) 提供一个领域适配规模参考：它从 Moshi 初始化，使用约 1840 小时客服互动和 410 小时通用 QA，共约 14.5 万段对话；论文实验配置在 8 张 A100 上训练约 6 小时。这个数字不能直接代表新增 action channel 的成本，但说明从成熟 duplex base 出发，角色和领域适配不一定需要预训练级预算。

实验可先只实现订单查询、物流、取消、退款、地址修改、身份验证、工单与转人工。每个业务都设计 partial request、改口、ambiguous slot、overlap、tool dependency、delayed result、cancellation 和 confirmation。

最小论文比较至少包含：同步级联 Agent、异步级联 Agent、full-duplex base + 外部工具 controller、共享 backbone + action channel，以及加入 latent distillation / critical-window RL / playback rollback 的完整模型。这样才能知道收益来自模型架构、异步系统，还是数据与训练。

## 14. 最有价值的研究贡献是什么

“给全双工语音模型加 function calling”本身不够强。更有价值的课题是：

**将 belief update、action scheduling、conversation control 和 speech generation 放到同一连续时间策略中，通过离线级联 Teacher 蒸馏与局部时序 RL，使模型在 partial observation 下提前行动、持续修正并安全回滚。**

可以将其概括为 Continuous-Time Speech-Language-Action Agent，并提出六个可检验假设：

- 共享 backbone + 独立 audio/control/action channel，优于串行 speech-agent pipeline；
- earliest-legal-action trajectory distillation 能让只读工具更早启动；
- latent belief update 比 chunk-level 长 CoT 具有更好的准确率-延迟-算力权衡；
- critical-window RL 改善交互和 action timing，同时保留基础模型智力；
- playback-aware rollback 降低对用户未听内容的错误引用和承诺；
- prepare/commit protocol 在降低 perceived latency 的同时抑制不可逆误执行。

这里最根本的转变是：不再把打断、工具调用、记忆和语音生成看成四个附加功能，而把它们视为同一 policy 在不同通道上的动作。模型不是“先理解，再规划，再行动，再说话”；它在同一个持续时间轴上不断感知、更新、行动、表达和撤销。

## 15. 结论

交互式 Omni Model 的上限不由声音多自然决定，而由它是否具备 Agentic state 和 policy 决定。自然回复、停顿、附和、倾听与打断只是最终表现；内在本质是持续估计用户目标、维护不确定 belief、异步安排工具、跟踪执行和播放状态，并在新证据到来时撤销旧计划。

因此，下一代模型不应只是 Speech-to-Speech，而应是 Speech-Language-Action。它可以保留模块化工具与存储，但必须消除推理关键路径上的同步 barrier；它可以使用强级联 Agent，但主要将其作为离线 Teacher；它可以提前思考和检索，但必须用可撤销 action、两阶段提交和 playback-aware rollback 约束现实后果。

一句话总结：**不要把现有 Agent pipeline 原样接到实时语音模型后面，而要用它生成连续时间专家轨迹，再把“理解、规划、行动、说话、停止与回滚”蒸馏进一个共享状态、多通道输出的流式 policy。**

## 参考资料

- [DuplexSLA](https://arxiv.org/abs/2605.20755)
- [DuplexSLA repository](https://github.com/hyzhang24/DuplexSLA)
- [FLAIR / The Silent Thought](https://arxiv.org/abs/2603.17837)
- [Can Speech LLMs Think while Listening?](https://arxiv.org/abs/2510.07497)
- [SHANKS](https://arxiv.org/abs/2510.06917)
- [BayLing-Duplex](https://arxiv.org/abs/2606.14528)
- [SmoothConv / DuplexConv](https://github.com/qualialabsAI/SmoothConv-DuplexConv)
- [Moshi](https://github.com/kyutai-labs/moshi)
- [DuplexPO](https://arxiv.org/abs/2607.07148)
- [PersonaPlex](https://arxiv.org/abs/2602.06053)
- [Stream RAG](https://arxiv.org/abs/2510.02044)
- [Full-Duplex-Bench v3](https://arxiv.org/abs/2604.04847)
- [Speculative Interaction Agents](https://arxiv.org/abs/2605.13360)
- [Asynchronous Tool Usage for Real-Time Agents](https://arxiv.org/abs/2410.21620)
- [VoxMind](https://arxiv.org/abs/2604.15710)

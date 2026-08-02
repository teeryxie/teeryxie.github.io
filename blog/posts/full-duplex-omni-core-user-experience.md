# 全双工 Omni 模型的核心与本质：不是同时听说，而是连续共同决策

> 如果只把全双工理解为“双向音频流可以同时工作”，我们会得到一个反应很快、声音很自然，却经常抢话、误停、答非所问、无法恢复任务的语音模型。真正值得研究的是：模型如何在连续、部分可观测、随时可能被纠正的交互中，与用户共同管理注意力、话权、目标和行动。

全双工 Omni 模型正在成为实时交互研究的核心方向。最直观的目标是：模型可以一边说话一边继续听，用户不必等待它说完，系统也不必等待一个僵硬的 VAD 终点才开始回答。GPT-4o、Moshi 以及之后大量 full-duplex spoken model 让这种体验从演示走向可研究的系统。

但当“同时收发音频”已经能够实现，真正的问题反而变得更清楚：为什么许多全双工模型仍然不好用？

- 用户停顿半秒思考，模型就急着抢答；
- 用户只说“嗯”，模型误以为被打断并停止；
- 用户真的说“等一下，不是这个”，模型却继续讲两秒；
- 电视里有人说话，助手把它当作新指令；
- 模型发出自然的“我明白”，后续回答却没有利用用户的情绪或上下文；
- 用户打断复杂任务后，系统停下了声音，却丢失了原任务和工具状态；
- 面向陪伴、教学、面试和紧急协助时，模型仍使用同一套“永远礼让”的交互策略。

这些失败表明：**全双工的本质不是通信链路，而是交互控制。** 更准确地说，它是一个连续时间中的 joint-action policy：用户和模型共同推进一个对话或任务，双方持续发出不完整证据，模型必须决定何时观察、等待、附和、接话、让出话权、澄清、行动、撤销和修复。

本文不再重复完整模型时间线，而是集中回答两个问题：

1. 全双工 Omni 模型真正的核心能力是什么？
2. 为了提升用户交互体验，研究资源应该优先投入在哪里？

![Full-duplex essence](/images/blog/full-duplex-ux-essence.svg "图 1：双向音频流只是基础设施；共同状态、话权策略、任务行动和修复能力才决定用户体验。")

## 1. 第一性原理：为什么人类交互不是两个音频流

人类对话表面上是声音交替或重叠，内部却是一个持续协调过程。说话者在组织内容，听者在预测话轮边界、判断意图、给出反馈并准备下一步动作。一个短暂停顿可能是换气、寻找词语、强调，也可能是真正结束；一个“嗯”可能表示继续听、同意、质疑，甚至准备夺回话权。

经典 conversation analysis 将 turn-taking 视为参与者共同组织的秩序，而不是固定轮询协议。Sacks、Schegloff 与 Jefferson 在 1974 年描述了人类话轮转换的系统性；Stivers 等人在跨语言研究中观察到，不同文化的具体时长存在差异，但人类普遍会预测并协调话轮边界。Clark 的共同基础理论则强调，交流不是发送句子，而是参与者不断建立“双方都认为已被理解”的 common ground。

因此，完整的交互状态至少包括：

```text
谁正在说？
这句话说给谁？
用户是否还在构造当前意图？
用户是在附和、犹豫、纠正，还是夺回话权？
哪些内容已经被双方确认？
模型当前说到哪里，用户实际听到哪里？
当前任务、工具和计划是否仍有效？
此刻沉默、短反馈、完整回答或执行动作，哪个成本最低？
```

如果模型只学习 `audio_in → audio_out`，这些变量只能被隐式压进下一 token 预测。它可能在平均数据上产生自然重叠，却很难保证在高风险、多人、嘈杂和长任务中做出一致决策。

我的操作性定义是：

> **全双工 Omni 模型是在连续时间中，对多模态部分观测进行共同状态估计，并选择对话控制、内容表达和现实行动的策略模型。**

“同时听说”只是它具备实时观测和实时动作的必要条件，不是最终能力。

## 2. 一个更准确的系统抽象：Sense、Believe、Coordinate、Act、Repair

可以把全双工交互拆成五个持续并行的过程。

### 2.1 Sense：持续感知，而不是等待用户说完

系统持续接收用户语音、面部、视线、手势、屏幕操作、环境声和工具事件。关键不是有没有人声，而是这些信号在当前任务中意味着什么。

### 2.2 Believe：维护不确定状态，而不是过早得出单一结论

用户说“帮我订明天……”时，模型可以判断任务可能与日程、交通或餐厅有关，但不应立刻固化意图。好的系统维护多个假设和置信度，随着新证据更新。

### 2.3 Coordinate：管理话权和共同注意

模型决定继续听、给出 backchannel、开始说、继续持有话权、让出话权或请求澄清。在带视频的 Omni 场景中，还要判断双方是否在关注同一个人、物体或屏幕区域。

### 2.4 Act：把语言与任务行动统一起来

回答、搜索、调用工具、修改 artifact 和写入记忆都是动作。全双工意味着这些动作可以异步发生：模型说“我先查一下”的同时开始检索，也能在用户改口时取消旧查询。

### 2.5 Repair：发现共同状态不一致后恢复

用户说“不是，我指左边那个”，不是一次普通新输入，而是对共同状态的修复。模型必须撤销旧指代、停止依赖旧指代的动作、确认新对象，并从正确位置继续任务。

![Continuous joint action loop](/images/blog/full-duplex-ux-joint-action-loop.svg "图 2：高质量全双工体验来自感知、状态估计、交互协调、任务行动和修复的持续闭环。")

这五个过程可以共享一个流式 backbone，也可以由多个模块协作。用户并不关心内部是否“端到端”；用户关心的是同步点是否阻塞、状态是否一致、错误是否可恢复。因此，架构判断应服从交互目标，而不是把 monolithic 当作信仰。

## 3. 用户体验的第一优先：不是更快开口，而是正确地听

很多实时系统首先优化 TTFT，但用户对延迟的感知不是单一数字。一个 250ms 开口却打断用户的模型，比 600ms 后正确接话更令人烦躁。一个立即停止但误把“嗯”当作新指令的模型，也不比半双工更自然。

真正的第一优先级是 **interaction precision**：

- 用户没说完时不要开始完整回答；
- backchannel 不应触发停止；
- 真正纠正和紧急抢话必须及时响应；
- 用户对旁人说话时不要介入；
- 电视、回声和扬声器泄漏不能改变任务状态；
- 不确定时用低成本澄清，而不是武断执行。

[Semantic-Aware Interruption Detection](https://arxiv.org/abs/2603.24144) 将当前系统概括为两个极端：VAD 方案 trigger-happy，容易把附和误判为打断；较稳健的端到端模型又可能停止太慢。该工作构建的 SID-Bench 来自真实人类对话，包含约 10 小时音频和 3,700 个实例，显式区分 genuine interruption、backchannel 与 ambiguous event。这比“有声音就停”更接近产品需求。

[FastTurn](https://arxiv.org/abs/2604.01897) 则同时使用流式 CTC 语义与声学特征，目标是在 partial observation 下平衡早决策和语义完整性。它代表一个重要方向：不必等待完整 ASR，也不能只看音量；系统需要尽早获得“足够语义”，同时保留不确定性。

[IRAF](https://arxiv.org/abs/2606.06559) 进一步关注现实声学干扰。全双工系统播放自己的声音时，麦克风会收到回声、漏音和旁人语音；如果这些信号直接污染 user stream，LLM 会把自己的话或干扰者内容当作用户状态。声学 echo cancellation、speaker attribution 与语义 addressee detection 因此不是低层工程细节，而是交互正确性的地基。

### 3.1 研究上应该优先测什么

```text
False interruption rate
Missed interruption rate
Premature response rate
Addressee error rate
Background-speech contamination
Semantic completeness at action time
Clarification precision and user effort
```

平均 stop latency 不能单独代表体验。应该按事件条件报告：真实抢话希望低 stop latency；backchannel 和背景语音则希望模型继续说。相同的“停止”动作，在不同语境中可能完全相反。

## 4. 第二优先：让用户能够低成本纠正，而不是追求零错误幻觉

真实交互一定会出现误听、指代错误、目标变化和工具失败。产品体验的上限不由“模型永不出错”决定，而由 **repair cost** 决定。

假设用户说：

```text
“把右边这张图放到第一页……不对，是第二页，右上角。”
```

系统需要完成：

1. 立即停止依赖旧目标的编辑；
2. 保留“右边这张图”这一仍然有效的实体绑定；
3. 将 page=1 supersede 为 page=2；
4. 新增 region=top-right；
5. 如果旧工具已经执行，恢复或生成可逆补丁；
6. 用一句简短确认建立新的 common ground；
7. 继续任务，而不是从头询问全部信息。

这要求 interruption handling 不只是停音频，而是 plan invalidation、tool cancellation、slot revision、playback rollback 和 artifact recovery。

[IHBench](https://arxiv.org/abs/2606.19595) 关注被中断后能不能正确续接工作流。它揭示了一个容易被忽略的事实：模型可能在声学上成功停止，却在任务层面不知道应该恢复旧回答、接受新目标，还是丢弃旧计划。用户体验真正关心的不是“停没停”，而是“停下之后事情还在不在”。

### 4.1 Repair 应成为一等动作

建议将交互动作显式表示为：

```text
HOLD       保持话权
YIELD      让出话权
BACKCH     短反馈但不夺话
ABORT      停止当前语音或动作
REVISE     更新部分状态
RESUME     从有效状态继续
CLARIFY    请求最小必要信息
CONFIRM    高风险动作前确认
```

如果系统只有 `speak / silence`，修复只能靠下一段自然语言临时解释，难以保证工具和 memory 同步。

## 5. 第三优先：从话轮自然度走向“任务连续性”

全双工模型最容易展示的是闲聊，但最有产品价值的是在任务中减少协调成本。例如：

- 用户一边描述行程，Agent 一边预取航班；
- 用户在看屏幕时说“这个字段不对”，Agent 绑定当前 UI 区域；
- 面试中候选人停顿思考，系统不催促，但必要时进行中性追问；
- 教学中学生出现明确概念错误，系统适时打断，而不是等完整错误推导结束；
- 客服中用户改口，系统撤销旧业务动作并保留已验证身份。

这说明全双工 Omni 的目标不是让语音更像人，而是让 **human-agent joint work** 更顺畅。

可以把交互效用写成一个产品目标：

```text
Interaction Utility
= Task Success
+ Shared Understanding
+ User Control
- Waiting Cost
- Coordination Cost
- Repair Cost
- Unsafe Action Risk
- Cognitive Load
```

更自然的 backchannel 可能提升 shared understanding，也可能增加 cognitive load；更积极的抢话可能提升教学纠错，却会破坏心理咨询中的表达安全。不存在一种全局最优的“像人”策略。

## 6. 第四优先：策略必须可控，不能只有一种“礼貌人格”

全双工系统常见的默认策略是 always-yield：检测到用户声音就停止。它在通用助手中看起来礼貌，却不适合所有角色。

- **心理支持**：应保守倾听，减少主动打断；
- **教学纠错**：学生出现关键错误时可以及时介入；
- **紧急辅助**：识别危险信号后应主动打断；
- **访谈和面试**：应允许完整叙述，但在答非所问时结构化追问；
- **任务协作**：用户只是口头附和时，Agent 应继续当前操作说明；
- **多人会议**：系统多数时候应保持旁观，只有被点名或满足触发条件才发言。

[F-Actor](https://arxiv.org/abs/2601.11329) 将 backchannel、interruption、话题、声音和主动发起作为可指令控制的行为，而不只被动处理用户打断。[Instruct-FD](https://arxiv.org/abs/2607.20460) 更直接评估模型能否遵循 turn-management instruction。其六个系统对比中，最佳模型的 instruction adherence 也只有 `64.4%`，主动 backchannel 和主动 interruption 尤其困难。

这说明“模型会打断”与“模型能按产品策略打断”是两种能力。训练集中出现过某类行为，不等于部署时能稳定控制它。

[PersonaKit](https://arxiv.org/abs/2605.06007) 则把 Yield、Resume、Bridge、Override 等策略做成可配置对象，并提供实时用户测试平台。它的试验规模很小，不能作为总体体验结论，但产品思路值得重视：turn-taking policy 应像声音和 prompt 一样可配置、可记录、可 A/B test，而不是藏在不可解释的模型权重里。

### 6.1 产品应暴露哪些控制项

```text
interruptibility       用户多容易打断模型
initiative             模型多主动发起或介入
backchannel density    附和频率
floor persistence      模型维持话权的倾向
silence tolerance      对用户停顿的容忍时间
clarification policy   不确定时何时询问
repair style           简短改口或完整解释
risk threshold         何时必须显式确认
```

这些参数不一定直接暴露给普通用户，但产品角色、场景和无障碍设置必须能够控制它们。

![UX priority stack](/images/blog/full-duplex-ux-priority-stack.svg "图 3：全双工体验应先保证听对、可修复和任务连续，再优化策略个性、表现力与拟人感。")

## 7. 第五优先：Omni 的独特价值是共同注意，不是给语音模型再加摄像头

纯语音 full duplex 主要协调话权；Omni 还必须协调“我们正在看什么”。视频、视线、手势、屏幕指针和空间位置使 turn-taking 变成 joint-attention policy。

用户说“这个不对”时，系统需要知道：

- 用户看向屏幕还是现实物体；
- 指向的是哪个区域；
- 该对象是否在上一帧移动；
- “不对”是在纠正对象、内容还是 Agent 的动作；
- 模型是否应该继续说、停下看、还是立即撤销操作。

[Moshi-Face](https://arxiv.org/abs/2606.21970) 将用户音频与面部输入、模型语音与面部动作放入全双工生成框架，代表从 audio duplex 向 embodied duplex 扩展的一步。但研究不能停在“能生成同步表情”。真正的 Omni 价值是使用目光、点头、困惑表情、手势和场景变化改善 belief update 与 action timing。

例如用户皱眉不一定意味着否定，也可能是看不清屏幕。系统应该把非语言信号作为概率证据，选择“暂停并确认”而不是直接推断心理状态。

[Real-Time Voice AI Hears but Does Not Listen](https://arxiv.org/abs/2606.26083) 的核心警示正是：模型可能感知到非语言线索，却没有在最终回复 policy 中正确利用它。感知 benchmark 高分不等于交互行为变好。

## 8. 第六优先：延迟要按用户感知拆解，而不是只报一个 TTFT

对全双工产品，至少有七种延迟：

| 延迟 | 起点与终点 | 用户体验含义 |
| --- | --- | --- |
| Perception latency | 信号发生到系统识别 | 是否及时注意到用户 |
| Stop latency | 用户真实抢话到助手静音 | 是否尊重用户夺回话权 |
| Backchannel latency | 适合反馈的语义点到短反馈 | 是否显得在认真听 |
| Turn-start latency | 用户完成意图到助手开口 | 是否有尴尬等待 |
| First-useful latency | 信息足够到第一条有用内容 | 是否真的开始解决问题 |
| Action latency | 参数足够到工具启动 | 是否利用用户讲话时间并行做事 |
| Recovery latency | 用户纠正到状态恢复一致 | 错误后多久重新可用 |

用户不总是要求立即得到完整答案。对于复杂任务，及时的 grounded acknowledgement、可见进度和允许继续补充，能够降低 perceived latency；但空洞 filler 不能伪装成进度。

[FLEXI](https://arxiv.org/abs/2509.22243) 同时关注延迟、质量和交互有效性，并加入 emergency 场景。论文提出 `<150ms` 和 `<400ms` 的交互等级划分，这是其 benchmark 的操作性定义，不应被当成所有产品的通用人类阈值。更可靠的做法是固定客户端、网络、音频设备和事件定义，报告 P50/P95/P99，并与任务成功和误触发联合解释。

### 8.1 最快并不总是最好

在用户仍可能补充关键信息时，等待 300ms 可能降低一次错误澄清；对紧急“停”指令，多等 300ms 又可能不可接受。延迟目标必须条件化于：

```text
semantic completeness
action reversibility
user urgency
confidence
interaction role
current floor state
```

## 9. 第七优先：自然度不是声音像人，而是时机符合情境

高音质、情绪和笑声会增强 social presence，但也可能掩盖模型没有理解。用户更容易把流畅表达误认为可靠推理。

[TurnNat](https://arxiv.org/abs/2607.01345) 尝试用自然双人对话训练的因果活动预测器，估计一个话轮边界之后双方语音活动的可能性，再用 NLL 衡量 timing atypicality。它的重要意义是将不同 timing failure 放到统一自然度框架中，而不是分别人为设定“停顿必须 300ms”之类的硬阈值。

[HumDial 2026 Full-Duplex Interaction](https://arxiv.org/abs/2604.21406) 则将 interruption 与 rejection 拆成多个场景：follow-up、correction、emergency、backchannel、第三方语音、pause 等。这里的 `rejection` 不是拒绝用户，而是拒绝对无效或非面向系统的语音做出错误响应。

这说明自然度是条件分布：在相同 500ms 重叠下，用户纠正时应该停止，用户附和时应该继续，用户对旁人说话时应该忽略。单一 pause threshold 永远无法覆盖这些语义。

## 10. 内容智能与交互动态必须解耦评估

当前全双工模型经常面临一个表面 trade-off：互动自然的模型内容较弱，强 reasoning 级联系统又显得迟钝。这不代表二者理论上不可兼得，而可能是训练目标和推理路径绑定造成的。

[DuplexPO](https://arxiv.org/abs/2607.07148) 的核心观点是将 `when to speak` 与 `what to say` 解耦，在 dynamics-critical window 优化开始、停止、附和和让出话权，同时保护 reasoning 和 instruction-following。[Multi-Faceted Interactivity Alignment](https://arxiv.org/abs/2606.11167) 也将 pause、turn-taking、backchannel 和 interruption 分开构造奖励，并额外加入内容质量约束，避免只优化 timing 后语义退化。

产品架构也可以解耦：

```text
fast interaction path:
  streaming perception
  floor control
  short acknowledgement
  interruption handling
  state repair

slow cognition path:
  deep reasoning
  retrieval
  planning
  tool execution

merge policy:
  validate freshness
  cancel stale work
  decide when and how to present result
```

关键不是所有模块是否位于同一个模型，而是二者是否共享一致状态、能否异步工作、旧结果是否可取消。`Policy monolithic, execution and reasoning modular` 往往比“所有能力强塞进一个串行 token 流”更符合实时产品。

## 11. 目前的评测为什么仍不足以代表用户体验

现有 benchmark 已经覆盖 pause、turn、backchannel、interruption、emergency、第三方语音和自然度，但仍有五个缺口。

### 11.1 局部片段多，完整任务少

模型在 10 秒片段里正确停止，不代表它能在 20 分钟客服任务中维护订单、工具和授权。

### 11.2 系统输出多，用户适应少

用户会因模型行为改变说话方式：缩短句子、避免停顿、提高音量、重复唤醒词。短期成功可能来自用户迁就，而不是系统自然。

### 11.3 平均分多，严重失败少

一次紧急中断漏检、错误发送或持续抢话，可能比 100 次自然 backchannel 更重要。应报告 tail risk 和 failure slice。

### 11.4 语音多，共同注意少

视频、视线、屏幕和多人空间中的 addressee、reference 与 privacy 尚缺统一协议。

### 11.5 自然度多，控制感少

用户是否知道模型在听谁、是否能暂停、能否改变打断策略、错误后要重复多少信息，这些产品指标常被忽略。

## 12. 一个面向用户体验的评测框架

我建议把评测分成六层。

### Layer 1：感知与指向

speaker、addressee、声学事件、情绪线索、视觉对象、视线和指针是否识别正确。

### Layer 2：交互控制

wait、start、hold、yield、backchannel、interrupt、resume 的条件精度和时延。

### Layer 3：共同状态

用户和模型已确认什么、用户实际听到什么、指代绑定什么、纠正后哪些状态失效。

### Layer 4：内容与任务

回答质量、工具正确率、task success、异步结果是否过期、被中断后能否继续。

### Layer 5：体验与行为

用户认知负担、重复次数、修复时间、控制感、信任校准、长期使用意愿，以及用户是否开始迁就系统。

### Layer 6：安全与边界

紧急中断、不可逆动作确认、旁人隐私、公共空间录音、情绪误判和未成年人场景。

![Full-duplex UX evaluation](/images/blog/full-duplex-ux-evaluation.svg "图 4：全双工评测应从感知、控制、共同状态一直追踪到任务、用户行为和安全后果。")

关键指标不应只有模型分数，还应包括：

```text
User repetition count
Clarification turns
Repair completion time
Task abandonment rate
False-interruption frustration
User adaptation index
Perceived control
Trust calibration
Long-session state consistency
Critical failure rate
```

## 13. 不要一上来训练：先做五个 Toy Experiment

### 实验 A：停顿还是结束

构造相同句子，仅改变停顿位置、音高延续和后续词。固定模型，只比较 VAD、声学 turn detector、流式语义 detector 和融合策略。测 premature response、真实结束等待和用户重复成本。

### 实验 B：附和还是真打断

使用相同“嗯”“对”“等一下”，改变其语义上下文和说话重音。测 false stop、missed interruption 和恢复时延。根因若是 addressee/intent，而非音频质量，换更大 codec 不应显著解决。

### 实验 C：停止声音还是恢复任务

让用户在工具执行、解释和确认三个阶段分别打断。比较仅停止播放、保存 turn state、完整 task/tool/playback state 三种系统。测任务连续性、重复信息量和 stale action rate。

### 实验 D：策略是否适配角色

同一段用户行为分别放入心理支持、教学和紧急辅助场景。比较固定 always-yield、规则策略和 instruction-conditioned policy。测策略遵循、主观适配和严重误介入。

### 实验 E：音频线索还是 Omni 共同注意

用户说“这个不对”，分别提供无视频、当前帧、带时间戳视线/指针和完整 artifact state。测澄清率、错误对象动作和修复成本，验证体验收益是否来自共同注意而非更自然语音。

这些实验都应固定 backbone、语音、网络和内容能力，只改变一个交互变量。Toy 环境稳定后，再扩展到噪声、多人、多语言、长会话和真实用户。

## 14. 我认为最值得专注的七个研究方向

### 14.1 Semantic addressee and interruption belief

从“检测到声音”走向“谁在什么语境下试图改变共同任务”。联合声学、streaming semantics、speaker、gaze 和历史状态。

### 14.2 Repair-first interaction policy

把 revise、resume、rollback 和 clarify 作为显式动作，优化错误后的恢复成本，而不是只优化首次正确率。

### 14.3 Task-aware duplex agent

让对话控制与工具、memory、artifact 和授权状态连接。打断必须影响计划，不只是音频播放器。

### 14.4 Controllable role-conditioned dynamics

让 backchannel、initiative、interruptibility、silence tolerance 和 risk threshold 可配置，并验证 instruction adherence。

### 14.5 Multimodal common ground

研究 gaze、gesture、screen pointer、facial feedback 和空间场景如何改善 referent binding、理解确认与行动时机。

### 14.6 Perceived-latency orchestration

同时优化 stop、first-useful、action 和 recovery latency；允许快路径反馈、慢路径推理和异步工具协作，但禁止伪造进度。

### 14.7 Longitudinal UX and user adaptation

研究用户使用数周后是否被迫改变表达方式、是否产生错误信任、个性化策略是否漂移，以及记忆和隐私边界是否仍可控。

![Research focus roadmap](/images/blog/full-duplex-ux-research-roadmap.svg "图 5：研究路线从听对与修复出发，推进到任务 Agent、策略控制、共同注意和长期体验。")

## 15. 产品落地的优先级：什么先做，什么后做

如果要做一个真正可用的全双工 Omni 产品，我会按以下顺序投入。

**P0：声学稳定与对象边界。** AEC、噪声、speaker、addressee、播放位置和客户端事件必须可靠。否则上层模型收到的是错误世界。

**P1：turn policy precision。** 先解决不该说时闭嘴、该停时及时停、附和不误停。不要先追求丰富笑声和情绪。

**P2：repair 与状态一致性。** 用户纠正后，语音、计划、工具、memory 和 artifact 一起更新。

**P3：任务连续性。** 加入异步工具、取消、过期检查、两阶段提交和跨设备状态。

**P4：角色与用户控制。** 为不同业务设置 interaction contract，让用户能调节或关闭主动打断和持续监听。

**P5：Omni common ground。** 用视频、视线、手势和屏幕提高共同注意，而不是只生成表情。

**P6：表现力与人格。** 在理解、修复和安全可靠后，再优化音色、笑声、风格和高拟人感。

这个顺序可能不如展示一个会笑、会抢话的 demo 吸引眼球，但更接近用户长期愿意使用的产品。

## 16. 最终判断

全双工 Omni 的最低定义是双方流可以同时工作；工程核心是低延迟流式感知与生成；研究核心是连续时间中的 state estimation 和 policy；产品核心则是 **共同完成事情时的协调质量**。

自然停顿、附和、打断、眼神和表情都只是外在行为。它们之所以有价值，是因为帮助用户判断模型是否在听、是否理解、谁拥有话权、任务是否仍受控制。脱离这些目标，拟人化行为反而会增加干扰和错误信任。

因此，我对下一阶段的结论是：

> **不要把主要资源继续投入“让模型更像人在说话”；应该优先让它更准确地知道何时听、听谁、理解到哪一步、什么时候不该行动，以及共同状态出错后如何低成本修复。**

真正优秀的全双工 Omni Model 不是最爱说话、最快说话或最像人的模型，而是能在持续变化的交互中保持注意、尊重话权、支持纠正、继续任务，并让用户始终拥有控制感的协作 Agent。

## 参考资料

1. [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037)
2. [Full-Duplex-Bench v1.5](https://arxiv.org/abs/2507.23159)
3. [FLEXI: Benchmarking Full-duplex Human-LLM Speech Interaction](https://arxiv.org/abs/2509.22243)
4. [F-Actor: Controllable Conversational Behaviour in Full-Duplex Models](https://arxiv.org/abs/2601.11329)
5. [Semantic-Aware Interruption Detection in Spoken Dialogue Systems](https://arxiv.org/abs/2603.24144)
6. [FastTurn: Unifying Acoustic and Streaming Semantic Cues](https://arxiv.org/abs/2604.01897)
7. [Full-Duplex Interaction from the ICASSP 2026 HumDial Challenge](https://arxiv.org/abs/2604.21406)
8. [PersonaKit: User Testing Diverse Roles in Full-Duplex Dialogue](https://arxiv.org/abs/2605.06007)
9. [Integrating Facial Generation into Full-Duplex Spoken Dialogue Systems](https://arxiv.org/abs/2606.21970)
10. [Real-Time Voice AI Hears but Does Not Listen](https://arxiv.org/abs/2606.26083)
11. [Multi-Faceted Interactivity Alignment in Full-Duplex Speech Models](https://arxiv.org/abs/2606.11167)
12. [IRAF: Noise-Robust End-to-End Full-Duplex Spoken Dialogue](https://arxiv.org/abs/2606.06559)
13. [TurnNat: Automatic Evaluation of Turn-Taking Naturalness](https://arxiv.org/abs/2607.01345)
14. [DuplexPO: Decoupling Conversational Dynamics through RL](https://arxiv.org/abs/2607.07148)
15. [Instruct-FD: Can Full-Duplex Systems Follow Turn-Taking Instructions?](https://arxiv.org/abs/2607.20460)
16. [IHBench: Interruption Handling in Real-Time Voice Agents](https://arxiv.org/abs/2606.19595)
17. [Sacks, Schegloff, Jefferson: A Simplest Systematics for Turn-Taking](https://doi.org/10.2307/412243)
18. [Stivers et al.: Universals and Cultural Variation in Turn-Taking](https://pubmed.ncbi.nlm.nih.gov/19805143/)

文中的架构优先级、体验框架和 Toy Experiment 是基于现有证据提出的研究判断，不代表相关论文已经完成整套产品验证。论文中的数字仅限其各自实验协议；跨模型比较必须统一客户端、网络、音频设备、事件定义和测试日期。

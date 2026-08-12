# 自然人机交互到底需要什么：Omni、全双工与 Agentic 架构的充分必要关系

“边听、边想、边看、边说、边写”听起来像是在描述一个端到端 Omni 全双工大模型。但如果把教培、电话销售、银行客服和会议助手放在一起，会发现它们需要的不是同一种模型：教培中的板书可以静默并行，电话销售更依赖语音节奏，银行业务的关键在事务治理，而被动会议纪要甚至不需要模型开口。

真正的问题不应该是“是否必须训练 Omni 全双工模型”，而应该是：**任务依赖哪些观测，哪些输入必须在输出期间保持开放，新证据多快影响当前行为，系统是否要改变外部世界，以及错误能否撤销。**

本文基于 DuplexOmni、ROMA、FireRedChat、DuplexSLA、Full-Duplex-Bench 系列与多人语音 Agent 等工作，给出一个任务条件化的关系框架。结论先说：自然交互是一种系统性质；Omni、全双工、Reasoning、Agentic、Memory 和 Governance 分别解决不同子问题，没有任何一个单独构成充分条件。

![Capability map](/images/blog/natural-interaction-capability-map.svg "图 1：自然交互由感知、时间、交互、认知、行动与治理六个正交维度共同决定。Omni、全双工和 Agentic 只覆盖其中一部分。")

## 1. 一页结论

- **单体 Omni 几乎从来不是逻辑必要条件。** 必要的是系统覆盖任务不可替代的模态；模块化感知也可以做到。
- **系统级双工是自然语音产品的常见必要条件。** 用户必须能在助手输出时继续输入、打断或纠正，但这可由 VAD、EoT、对话控制器和可取消 TTS 实现。
- **语义全双工只在更强条件下必要。** 输出期间的新证据必须在当前输出结束前改变当前计划或后续表达，而不是简单“停下后重答”。
- **Agentic 在现实任务中通常必要。** CRM、账户、日历、邮件、检索和业务 API 不在模型参数里；需要读取当前状态或产生副作用时必须有工具运行时。
- **Agentic 仍不充分。** 没有确认、幂等、取消、回滚和审计，越快调用工具反而越危险。
- **全双工不应是系统总开关。** 对话通道可全双工，查询可异步并行，转账、下单等提交阶段必须事务化、显式确认。
- **最稳健的默认方案是混合架构。** 快交互层持续听看和控制时机；慢认知层推理；Agent Runtime 执行；Commit Governor 管权限与副作用。
- **评测也必须分层。** 自然度、打断、工具正确率、任务成功率、恢复成本与安全违规不能压成一个平均分。

一句话概括：**你要研究的不是“Omni 全双工模型”，而是基于共享时间轴、可中断输出、持续状态、异步推理、可撤销行动和明确治理的 interaction-native Agent system。**

## 2. 六个正交维度

自然交互至少包含六个问题：

| 维度 | 核心问题 | 典型实现 |
| --- | --- | --- |
| Perception | 系统能感知什么 | Audio、Video、Screen、OCR、Gesture |
| Temporality | 输入输出能否同时持续 | Streaming、Duplex、Barge-in |
| Interaction | 何时听、等、说、停、附和 | Turn policy、EoT、Speak head |
| Cognition | 如何理解、推理和建模用户 | LLM/VLM、belief state、planner |
| Action | 如何读取和改变外部世界 | Tool use、workflow、Agent runtime |
| Governance | 什么能执行，错了怎么办 | Confirmation、ACL、rollback、audit |

Omni 主要影响第一维；full duplex 主要影响第二维，并为第三维提供可达性；Agentic 主要影响第五维。一个模型即使三者都有，也可能回应错对象、抢话、误用参数或未经授权转账，所以仍然不充分。

## 3. 先区分系统级与模型级

### 3.1 系统级多模态不等于单体 Omni

设任务 `T` 所需的不可替代模态集合为 `M_T`。如果学生的笔迹决定错误类型，视觉或屏幕输入属于 `M_T`；纯电话销售中，视频通常不属于 `M_T`。

系统必须覆盖 `M_T`，但实现可以是：

```text
ASR + speaker model + vision/OCR + shared state + LLM
```

也可以是：

```text
Audio + Video + Text -> unified Omni model
```

因此：

```text
任务需要多模态系统  ⇏  必须使用单体 Omni 模型
```

单体 Omni 的优势是细粒度音画对齐、统一 hidden state 和较少接口损失；代价是训练复杂、升级耦合、故障难定位。只有跨模态实时联合建模带来的收益明显高于这些代价时，它才是合理选择，而不是功能上的必选项。

### 3.2 系统级双工不等于语义全双工

**系统级双工**要求助手播放语音时，麦克风、摄像头和事件流仍开启，用户可触发停止或让出话权。FireRedChat 展示了模块化路线：个性化 VAD、语义 EoT、Dialogue Manager 与级联/半级联后端也能形成可插拔全双工系统。

**语义全双工**更强：模型在自己输出时仍理解新输入是背景声、附和、抢话还是条件修正，并在线改变尚未完成的计划。

```text
系统级：听到用户 -> 停止 TTS -> 等用户说完 -> 重新回答
语义级：理解修正 -> 取消旧计划 -> 直接续接新的有效回答
```

前者已经能显著消除 IVR 感；只有当“新证据必须在当前输出结束前因果改变当前行为”时，后者才接近必要。

![Duplex levels](/images/blog/natural-interaction-duplex-levels.svg "图 2：I/O 双工、行为双工和语义双工是逐级增强的能力，不应被一个 full-duplex 标签混为一谈。")

## 4. 三层双工定义

### Level 1：I/O 双工

输入和输出通道并发。助手说话时仍接收麦克风、视频和屏幕事件。核心是流式协议、AEC、缓存、调度与播放位置，不要求模型原生全双工。

### Level 2：行为双工

系统能够判断 `WAIT / BACKCHANNEL / SPEAK / YIELD / INTERRUPT / IGNORE`，并控制语音、板书或 UI。ROMA 把 `when to speak` 与 `what to say` 分离：连续音视频被组织为时间对齐单元，独立 speak head 判断是否响应。这说明自然性的瓶颈首先是时机 policy，而不只是内容生成。

### Level 3：语义或认知双工

系统输出时持续更新意图、参数、用户状态和计划；新信息到来后能修正后续措辞、板书与工具草案。DuplexSLA 将 speech、language 与 action 放在同步时间轴上，使工具动作可与语音并行；但对不可逆动作，能生成 action token 不代表可以绕过业务确认。

2026 年全双工系统综述进一步指出，在其审计的系统中，持续的并发双向长语音仍是未被稳定覆盖的状态。大多数产品真正需要的是选择性重叠、可靠打断和状态修复，而不是双方一直同时长篇说话。

## 5. Agentic 何时必要

当任务只需回答稳定知识，工具不是逻辑必要条件；当任务涉及下列任何事项，Agent Runtime 基本必要：

- 查询 CRM、账户余额、物流、实时价格；
- 创建订单、修改预约、发送邮件；
- 生成并保存会议任务；
- 搜索、数据库、计算器或企业知识库；
- 持续任务和外部事件订阅。

但“Agentic 必要”不等于“端到端模型直接提交工具必要”。可选实现包括 action channel、外部 planner、确定性 workflow 和多 Agent 编排。

Full-Duplex-Bench-v3 的结果说明了为什么要拆分交互与行动：在其六系统实验中，GPT-Realtime 的 Pass@1 为 `0.600`；Gemini Live 3.1 的延迟最快，为 `4.25 s`，但 turn-take rate 只有 `78.0%`；Whisper→GPT-4o→TTS 级联保持完美 turn-take，却有 `10.12 s` 的最高延迟。各系统在自我修正和困难多步任务上都持续失败。**快、会接话和会正确做事是三条不同轴。**

## 6. 为什么 Omni + 全双工 + Agentic 仍不充分

即使系统具备统一 Omni 模型、语义双工和工具调用，它仍可能：

- 把旁人谈话当成用户命令；
- 在学生思考停顿时错误抢话；
- 在会议中回应错误的人；
- 用户尚未说完就提交旧参数；
- 工具完成后假装可以撤销；
- 引用被打断、用户从未听到的内容；
- 自然流畅地完成错误任务。

更接近工程充分条件的能力集合是：

```text
N_T ≈ P_T ∧ S ∧ C_T ∧ R_T ∧ A_T ∧ E_T ∧ G_T
```

其中 `P_T` 是任务感知覆盖，`S` 是持续流式处理，`C_T` 是交互控制，`R_T` 是任务推理，`A_T` 是必要行动，`E_T` 是错误恢复，`G_T` 是权限与治理。这个表达不是数学定理，而是一份可测试的系统 contract：缺少任一项，都能构造出自然交互失败案例。

![Necessary relation](/images/blog/natural-interaction-relation.svg "图 3：单体 Omni、语义全双工和 Agentic 都是条件化实现手段；任务覆盖、流式状态、交互控制、恢复与治理共同构成工程能力集合。")

## 7. 双工必须按通道和阶段定义

银行客服是最清晰的例子：

| 通道或阶段 | 推荐并发方式 |
| --- | --- |
| 用户讲话与系统感知 | 始终流式 |
| 助手播报与用户插话 | 系统级全双工 |
| 屏幕说明或文字摘要 | 可并行 |
| 查询账户与规则 | 后台异步 |
| 修改参数草案 | 可更新、可取消 |
| 转账、下单、账户变更 | 参数冻结、显式确认、事务提交 |

所以正确说法不是“银行 Agent 是全双工”，而是：**对话层全双工，认知层异步，事务提交层半双工。**

同理，教培中的板书可以在学生讲话时继续，但语音不必一直重叠；销售检索可以提前，订单提交必须等待确认；会议摘要可以持续写，主动发言需要角色与 floor 权限。

## 8. 三个时钟的混合架构

DuplexOmni 给出了很有启发性的分层：实时 interaction layer 持续接收音频、视频和历史并流式输出；可插拔 thinking layer 可以是 LLM 或 tool agent，按需在后台计算，结果被前台逐步接收并融入后续回复。这说明低延迟互动与深度推理不必被迫共享一个生成时钟。

我建议进一步加上事务治理层：

```text
Audio / Video / Screen / Text / Events
                    ↓
       Streaming perception and alignment
                    ↓
       Shared timeline + world/interaction state
          ├─ Fast interaction clock
          │  listen / wait / speak / yield / annotate
          ├─ Cognitive clock
          │  reason / model user / plan / retrieve
          ├─ Task clock
          │  CRM / bank API / calendar / email
          └─ Commit governor
             permission / confirm / idempotency / rollback / audit
                    ↓
       Speech / text / board / UI / tool action
```

![Three clocks](/images/blog/natural-interaction-three-clocks.svg "图 4：快交互、中认知、慢事务三个时钟通过共享状态和事件总线协作，避免慢推理阻塞自然互动，也避免低延迟策略直接越权提交。")

### 快时钟：几十到数百毫秒

负责语音活动、addressee、backchannel、停止播放和让出话权。目标是低延迟与低误触发，不适合做长链推理。

### 中时钟：数百毫秒到数秒

负责意图、异议、学生理解状态、会议议题和答案计划。它可流式更新，但不应阻塞快时钟维持自然交互。

### 慢时钟：事务生命周期

负责外部查询、订单、账户、邮件和持久记忆。它强调一致性、权限、审计和恢复，而非每 100 ms 决策。

## 9. 教培：输入常开，输出按需

教培“边听、边想、边看、边写”不要求持续语音重叠。必要能力通常是：

- 音频、屏幕或笔迹持续输入；
- 学生步骤、概念掌握与错误假设的共享状态；
- 板书作为独立、可修改的 action stream；
- 学生插话时语音立即停止；
- 后台推理不阻塞简短反馈；
- 在高价值时刻才介入。

推荐非对称或条件式双工：用户输入常开，语音按需，板书可并行，所有输出可取消。只有口语陪练、跟读纠音、实验安全干预和长篇讲解中的实时 backchannel，才强烈需要语音语义双工。

最重要的指标不是“重叠时长”，而是错误发现延迟、无必要打断率、学生重复信息量、修正后的状态一致性和学习收益。

## 10. 电话销售：语音双工 + Agentic

电话销售没有视觉任务时不需要 Omni；但系统级双工基本必要，因为用户会随时打断、犹豫、附和和修改条件。语义双工价值较高：用户说“等等，我只考虑随时能取的”，系统应取消旧产品计划并立即切换流动性约束。

推荐：

```text
speech-native interaction model
+ strong reasoning model
+ CRM / product knowledge agent
+ controlled order workflow
```

销售层可以积极做 backchannel、异议识别和策略调整；正式订单必须参数冻结并确认。评测应同时看转化、用户反感、误打断、事实正确率、合规话术与取消成功率，不能只看“像不像真人”。

## 11. 银行客服：可控双工 + 确定性事务

电话银行默认不需要视觉 Omni；视频开户、证件核验或共享屏幕排障才需要多模态系统。自然对话层需要系统级双工，但模型级语义双工可按收益选择。

真正不可缺的是 Policy & Commit Governor：

```text
PREPARE -> VERIFY IDENTITY -> FREEZE PARAMETERS
        -> USER CONFIRMATION -> COMMIT -> RECEIPT
```

查询可以提前并行，转账不能因为听到半句就提交。端到端语音模型可提出 action proposal，但不应拥有绕过权限、确认和幂等检查的最终能力。

## 12. 会议助手：先确定角色

“会议助手”至少有三种完全不同的产品。

### 被动记录者

持续听、做说话人分离、看 PPT/白板、写纪要。它需要多模态流式感知、长上下文和记忆，但不需要语音全双工。

### 私人 Copilot

在私人屏幕提示证据、待办或问题。文字输出与会议音频天然并行，也通常不需要开口；Agentic 搜索和记忆比语音双工更重要。

### 主动参与者或主持人

需要追问、总结、控场和推进议程。此时必须判断是否应该参与、回应谁以及以什么角色发言。正式论文《Adaptive Turn-Taking for Real-time Multi-Party Voice Agents》中的 ModeratorLM 正是角色条件化多人 turn-taking：论文报告相对非角色条件基线，turn-taking precision 提升超过 40%，recall 提升超过 70%，并减少误插话。这里的核心不是简单双人全双工，而是 role、addressee、group floor 和议程状态。

![Scenario matrix](/images/blog/natural-interaction-scenario-matrix.svg "图 5：教培、电话销售、银行客服与会议助手需要不同能力组合。勾选的是场景默认需求，不是所有产品形态的绝对规则。")

## 13. 一套直接可用的选择规则

按顺序回答六个问题：

1. **非音频模态是否包含不可替代信息？** 是，则需要多模态系统；跨模态细粒度时序很关键时，再评估单体 Omni。
2. **助手输出时用户是否必须保持输入权？** 是，则需要系统级双工和可取消输出。
3. **新输入是否必须在当前输出完成前改变当前计划？** 是，则需要语义双工或等价的在线状态更新内核。
4. **任务是否读取实时外部状态或产生副作用？** 是，则需要 Agent Runtime。
5. **状态是否跨回合、跨设备或跨任务保持？** 是，则需要有来源、过期和撤销语义的 Memory。
6. **动作是否涉及资金、账户、隐私、订单或对外通信？** 是，则需要独立治理与事务提交层。

若第 2 项为否，不要为了“先进”训练全双工；若第 1 项为否，不要为了“Omni”引入视频；若第 4 项为否，不要把普通问答过度 Agent 化。

## 14. 应该怎样做最小研究原型

第一版不需要训练包办全部能力的模型。可以依次验证：

### 实验 A：可中断输出是否足够

对比半双工、barge-in 重启和语义修正三种策略。若用户满意度与任务成功率在后两者之间没有显著差异，暂时不需要昂贵的模型级双工。

### 实验 B：视觉是否真的改变决策

对教培或会议构造音画冲突、屏幕错配和指向消融。若移除视觉不影响结果，当前任务或数据没有证明 Omni 必要。

### 实验 C：并发工具的收益与风险

比较 turn-end 后调用、earliest-ready 调用与 prepare/commit。报告端到端延迟、错误预调用、取消成功率和不可逆误执行，不只报告平均速度。

### 实验 D：三时钟消融

分别移除快速交互层、后台思考层和 Commit Governor。观察自然度、任务正确率和安全违规是否在不同层独立下降，从而验证分层架构而非凭直觉堆模块。

## 15. 评测合同

一个合格 benchmark 至少分开报告：

| 类别 | 指标示例 |
| --- | --- |
| 感知 | ASR、speaker/addressee、视觉 grounding |
| 时机 | start/stop latency、false barge-in、backchannel precision |
| 认知 | intent、constraint revision、task success |
| 行动 | tool arguments、Pass@1、dependency order |
| 恢复 | cancel success、rollback、unheard-content reference |
| 治理 | unauthorized commit、confirmation coverage、audit completeness |
| 体验 | interruption cost、user effort、subjective naturalness |

Full-Duplex-Bench 从 pause、turn-taking、backchannel、interruption 扩展到 v1.5 的 overlap handling，再到 v3 的真实 disfluency 与 chained API，正说明评测对象正在从“会不会同时说听”转向“在混乱输入中能否正确完成任务”。

## 16. 最终判断

自然人机交互没有一个脱离任务的单模型充分条件。更准确的关系是：

- Omni 是跨模态信息融合的实现路线；
- 全双工是输入输出并发和在线修正的能力层级；
- Agentic 是读取现实状态和执行任务的运行时；
- Memory 维持跨时间状态；
- Governance 决定哪些动作能安全落地；
- Interaction policy 决定何时使用上述能力。

对多数产品，合理默认不是“一个 Omni 全双工模型包办一切”，而是：

```text
条件式多模态感知
+ 系统级双工与可中断输出
+ 按需升级的语义双工内核
+ 异步强推理
+ Agent Runtime
+ 独立事务治理
```

教培优先持续观察、学生状态和并行板书；电话销售优先语音双工、异议状态和 CRM；银行客服优先可控双工与事务安全；会议助手先定义是记录者、私人 Copilot 还是主动参与者。

因此，最值得研究的统一问题不是“如何让模型永远同时听说”，而是：**系统能否在共享时间轴上持续判断，在正确时刻通过正确通道采取最小必要动作，并在用户修正、工具失败或权限不足时低成本恢复。**

## 参考资料

1. [DuplexOmni: Real-Time Listening, Seeing, Thinking, and Speaking for Full-Duplex Interaction](https://arxiv.org/abs/2606.09186), 2026.
2. [ROMA: Real-time Omni-Multimodal Assistant with Interactive Streaming Understanding](https://arxiv.org/abs/2601.10323), 2026.
3. [FireRedChat: A Pluggable, Full-Duplex Voice Interaction System with Cascaded and Semi-Cascaded Implementations](https://arxiv.org/abs/2509.06502), 2025.
4. [DuplexSLA: A Full-Duplex Spoken Language Model with Synchronized Speech, Language, and Action](https://arxiv.org/abs/2605.20755), 2026.
5. [Full-Duplex-Bench](https://arxiv.org/abs/2503.04721), 2025.
6. [Full-Duplex-Bench v1.5](https://arxiv.org/abs/2507.23159), 2025.
7. [Full-Duplex-Bench-v3](https://arxiv.org/abs/2604.04847), 2026.
8. [Adaptive Turn-Taking for Real-time Multi-Party Voice Agents](https://arxiv.org/abs/2606.13544), 2026.
9. [A Survey of Full-Duplex Spoken Dialogue Systems: Architectural Hierarchy, Interaction Ontology, and Decision State Machine](https://arxiv.org/abs/2606.19453), 2026.
10. [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037), 2024.

> 说明：本文的“必要/充分”均相对于明确的任务 contract，而非对所有人机交互做形式逻辑定理。论文数据按各自公开协议引用，不把不同系统的延迟和任务分数做未经控制的直接排名。

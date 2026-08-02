# 全双工 Omni 的心智理论：从 MeetingToM 标签到 Mental World Modeling 轨迹

> 联合精读 [MeetingToM](https://arxiv.org/abs/2607.19235) 与 [Mental World Modeling](https://arxiv.org/abs/2607.27201)。本文关注的不是模型能否猜中一个心理标签，而是标签究竟如何构造、它代表谁的判断、模型在什么时间获得了哪些证据，以及这种心智假设如何进入全双工交互策略。

全双工 Omni 模型可以持续看、听、说，也可以在用户说话时继续更新内部状态。但“持续接收音视频”并不会自动产生社会理解。一个系统可能在 300 ms 内发出自然的“嗯”，却不知道用户是真的困惑、礼貌性附和、不同意，还是只在组织下一句话；它也可能识别出皱眉，却把“没有理解”误判为“理解但反对”。

这正是 Theory of Mind（ToM，心智理论）进入实时交互的原因：系统需要根据不完整、带噪声且不断变化的证据，对他人的注意、知识、信念、目标、意图、态度和情绪维护**可修订的假设**。这些假设不是读心结果，而是一个用于选择下一步交互动作的 belief state。

2026 年 7 月的两篇新论文从两个互补方向推进了这个问题：

- **MeetingToM** 把 ToM 放进自然多人会议，测试个体状态、指向关系和群体共识；
- **Mental World Modeling（MWM）** 把 mental variables 写入 world state，让模型显式模拟候选动作如何同时改变物理世界与心理世界。

我的核心判断是：

> MeetingToM 将 ToM 操作化为多人会议中的局部、可观察、分层社会推断；MWM 将 ToM 从最终答案升级为影响行动与后继状态的过程变量。对全双工 Omni 而言，两者还差最后一步：将离散窗口或单步过程改造成带时间、来源、不确定性和修订关系的在线 mental-belief trajectory。

![Three levels of MeetingToM](/images/blog/tom-meeting-three-levels.svg "图 1：MeetingToM 从个体、二人关系到群体共识逐层提高社会推理粒度。")

## 1. 先把“心智理论”说清楚：模型预测的不是人的真实内心

ToM 常被简化为“推断别人怎么想”。这个说法容易引出两个错误：一是把任何情绪分类都叫 ToM；二是把模型输出当作人的真实私有状态。

对交互模型更严谨的定义应包含三个层次：

1. **Epistemic state**：对方看到了什么、知道什么、相信什么，是否拥有错误信念；
2. **Motivational and affective state**：对方想达成什么、准备做什么、处于何种情绪或态度；
3. **Recursive social model**：A 认为 B 知道什么，A 是否意识到 B 在误解自己，群体成员是否认为其他人已经同意。

全双工场景还多一个时间约束：模型在时间 `t` 只能根据 `t` 以前的音频、视频、对话和行动来更新判断，不能借用未来的解释或结果。于是系统真正维护的是：

```text
P(mental state of person i at time t | observable evidence up to t)
```

而不是：

```text
the true mental state of person i
```

这一区分决定了数据如何标、指标如何解释，也决定了产品能否允许用户纠正系统。

## 2. 两篇论文不在解决同一个层级的问题

| 维度 | MeetingToM | Mental World Modeling |
| --- | --- | --- |
| 场景 | AMI 自然多人会议 | 人工构造的日常决策场景 |
| 输入 | 同步视频、音频、转写与多视角裁切 | 文本、图片或有声视频故事 |
| 核心问题 | 此刻个体、二人关系或群体状态是什么 | 目标看到什么，采取动作后世界如何变化 |
| 标签形式 | 离散分类与参与者 ID | joint state、target observation、每个动作的 successor state、最终动作 |
| 时间结构 | 局部 5 秒或约 50 秒窗口 | 决策时刻的一步 counterfactual transition |
| gold 性质 | 多位观察者按规则达成的一致解释 | 规范化、人工构造的过程标注 |
| 强项 | 真实多人互动与非语言证据 | 可诊断的状态、视角与转移过程 |
| 主要缺口 | dominant label 压平变化，缺少连续轨迹 | 非自然实时轨迹，媒体合成且仅一步转移 |

MeetingToM 更像“在真实会议片段中，观察者能否给出一致的社会解释”；MWM 更像“如果我们把心理状态当作世界模型变量，能否用它预测人的下一步行动”。前者有真实互动，过程监督较弱；后者过程监督完整，生态真实性较弱。

联合阅读的价值就在这里：**真实数据告诉我们证据有多模糊，过程模型告诉我们应该把哪些中间变量显式化。**

## 3. MeetingToM：把多人会议拆成三层 ToM

MeetingToM 基于 AMI Meeting Corpus，包含 `1,800` 个片段、`60` 场会议，三个 task family 各 `600` 个实例。它同步裁切会议全景、个人 close-up、音频和 transcript，并可向模型提供全局视图、个人视图或 `2×2` mosaic。

### 3.1 Task 1：Subject-Level Mental State

第一层对目标参与者做七分类：

| 标签 | 操作性含义 | 容易混淆的边界 |
| --- | --- | --- |
| Active Engagement | 主动发言、推动任务、明显投入 | 不等于只是保持注视 |
| Supportive Endorsement | 明确支持、赞同或鼓励 | 礼貌点头不一定是支持 |
| Focused Listening | 高注意、低外显表达的倾听 | 安静不等于 disengaged |
| Cognitive Conflict | 理解内容，但怀疑、反对或价值冲突 | 与“没听懂”不同 |
| Confused Bewilderment | 对内容理解发生困难 | 与 disagree 不同 |
| Hesitation | 理解后在表达或行动前自我监控 | 与 confusion 不同 |
| Disengaged Withdrawal | 注意和任务参与明显下降 | 短暂看开不必然 disengaged |

标签证据来自多个 AMI-derived cues：dialogue acts、词语和 disfluency、头部与手部动作、gaze/focus、movement/posture 以及 turn-taking timing。

这里需要提出一个分类学问题：七类并不是同一语义轴上的“纯心智状态”。`Focused Listening` 和 `Active Engagement` 更像 interaction state，`Disengaged Withdrawal` 混合注意与行为，`Supportive Endorsement` 带关系态度，`Confused Bewilderment` 才更接近认知状态。这个混合 taxonomy 对会议产品很实用，却不能被解读为完整心理学本体。

### 3.2 Task 2：Dyadic Referential Reasoning

第二层首先识别说话者把话说给谁：

```text
A / B / C / D / Multiple / Unknown
```

证据包括 speaker gaze、pointing、torso orientation，以及“谁发生了即时反应”。然后预测被指向者的 stance：

```text
Support / Oppose / Neutral / Uncertain
```

标注尤其关注 `you` 前后约 `±2s` 的反应。论文还设置了值得产品团队注意的 **politeness trap**：如果语言表面赞同，而身体姿态表现抵触，应按视觉 stance 标注。

这一任务比简单情绪分类更接近 ToM，因为模型必须回答“这句话对谁产生了什么社会效果”。但它仍是第三方解释：身体后仰可能表示反对，也可能只是坐姿调整。标签需要证据，却不能消除行为到心智之间的不可辨识性。

### 3.3 Task 3：Group-Level Consensus

第三层判断会议是否形成共识：

```text
True Consensus
Pseudo Consensus
No Consensus
Uncertain
```

并识别 hidden dissenter：

```text
participant ID / None
```

`Pseudo Consensus` 是论文最有特点的设置：表面语言上形成一致，但某位成员通过 lip press、token nod、arms crossed、lean back 或 gaze away 泄露反对。模型不仅要看发言者，还要同时观察未发言成员。

这里也有最强的伦理边界。会议辅助系统可以说“存在未解决信号，建议再确认”，却不应向管理者断言“B 私下反对”。前者是可行动的不确定性提示，后者把外部观察变成了心理定罪。

## 4. MeetingToM 的标签究竟怎样构造

用户特别关心 trajectory 中如何构建标签。MeetingToM 严格来说并未发布连续 mental trajectory；它构建的是从长会议中挖掘、裁切并人工解释的**事件窗口标签**。

![MeetingToM annotation pipeline](/images/blog/tom-meeting-annotation-pipeline.svg "图 2：GPT-5 负责起草候选问题与选项，最终 gold 由独立人工标注、仲裁和过滤产生。")

### 4.1 第一步：按任务驱动挖掘候选窗口

- Task 1 从目标参与者作为 speaker 或 listener 的互动窗口中采样；
- Task 2 以 second-person pronoun `you` 为锚点，寻找潜在指向事件；
- Task 3 使用 AMI decision timestamps，再结合 proposal-response structure 与 agreement markers 找共识事件。

这不是随机均匀切片。候选挖掘先用规则提高目标现象密度，因此 benchmark 分布不能直接代表真实会议中各种状态的自然发生率。

### 4.2 第二步：GPT-5 起草问题和选项，但不决定 gold

论文使用 GPT-5 生成 candidate questions 与 answer options。关键职责边界是：GPT-5 不应确定正确标签，人工标注者也不能依赖它的 rationale。最终答案必须回到原始视频与 transcript。

这是一种合理的人机分工：LLM 扩展问法和干扰项，人在证据层裁决。但它仍留下可研究的问题：候选问题的表述是否会引导标注者，选项空间是否把连续状态强行离散化，以及 GPT 草拟的 distractor 是否具有稳定难度。

### 4.3 第三步：先看上下文，再看目标窗口

人工 viewing protocol 的顺序是：

1. 先看完整 corner view，理解 topic flow 与 interaction structure；
2. 再聚焦目标窗口，以 close-up 为主、corner 为辅；
3. 按 `visual cues > transcript > inference` 的优先级收集证据；
4. Task 1 每个窗口至少记录两条 supporting cues；
5. 标 dominant state；若窗口内发生变化，取主导状态或末尾最明显状态；
6. 只标 episode-bound state，不推断稳定人格、长期偏好或敏感属性。

附录界面显示，标注者会在约 50 秒上下文中处理 early `0–5s` 与 late `last 5s` 窗口；benchmark 的 Task 1 输入则使用对应 5 秒 close-up。换言之，上下文帮助人工形成标签，但模型最终面对的是短片段分类。这一点可能造成 **annotation-context advantage**：标注者看到了更长背景，模型输入是否拥有同等可见信息必须在具体设置中核对。

### 4.4 第四步：双人独立标注、第三人仲裁与删除

协议是：

```text
task-driven candidate mining
→ GPT-5 drafts question/options
→ two trained annotators independently label
→ third annotator adjudicates disagreements
→ majority vote
→ discard no-majority cases
→ jointly verify retained non-unanimous cases
```

temporal misalignment、missing view、insufficient evidence、unresolved ambiguity，以及依赖 clip 外信息的样本会被删除。

这使保留集更“可判”，也意味着 benchmark 测的是筛选后的可观察 ToM，而不是开放世界中所有难以判定的社交状态。产品里恰恰会大量遇到被 benchmark 丢弃的情况，因此 `Unknown` 和主动澄清能力不能被忽略。

### 4.5 一致性数字告诉了我们什么

每个任务抽取 200 个实例，由三位 annotator 独立标注，报告的 Fleiss' kappa 为：

| 子任务 | κ |
| --- | --- |
| Task 1 mental state | 0.7104 |
| Task 2 addressee | 0.7319 |
| Task 2 attitude | 0.5014 |
| Task 3 consensus | 0.5681 |
| Task 3 dissenter | 0.5729 |

addressee 相对容易达成一致；stance、consensus 和 hidden dissenter 明显更主观。这个差距不只是“标注质量问题”，而是说明：从行为推到隐含态度本身就存在多解性。

### 4.6 gold 的正确解释

作者明确承认无法重新联系参与者取得 self-report，因此 gold 是 systematic third-person inference，不是参与者 private belief 的 definitive truth。论文也警告不应将输出用于真实员工的 hidden intention、attitude 或 performance 决策。

因此最准确的表述不是：

```text
participant B is opposed
```

而是：

```text
under MeetingToM's operational scheme,
independent observers judged the visible evidence as Oppose
```

## 5. MeetingToM 的结果：困难到底在哪里

人类结果为：

| 任务 | 人类结果 |
| --- | --- |
| Task 1 mental state | Acc 86.33 / Macro 75.79 |
| Task 2.1 addressee | 86.00 |
| Task 2.2 stance | 83.33 |
| Task 3.1 consensus | Acc 80.33 / Macro 74.97 |
| Task 3.2 hidden dissenter | 79.67 |

人类也远非 100%，再次说明标签不是直接可测量物。论文评估的 proprietary 与 open-source MLLM 均明显落后，尤其是 Task 1 Macro 与 Task 3 consensus Macro。Task 1 分布偏向 active/attentive，Task 3 中 True Consensus 较多，因此只看 accuracy 会掩盖少数类失败。

对全双工研究最重要的不是追单个总分，而是问模型为何错：

- 没看到关键非语言信号；
- 看到了信号但分不清 confusion 与 conflict；
- 只跟踪 speaker，忽略 listener；
- 不知道 utterance 指向谁；
- 把礼貌语言当作真实态度；
- 无法聚合多人状态形成 group belief；
- 过度自信，拒绝输出 Unknown。

## 6. Mental World Modeling：把心智变量写入世界状态

MWM 的出发点是：只建模物理世界会在“画面看起来正确、人的行动却由错误信念驱动”时预测失败。它把 joint state 写成：

```text
s_t = (s_t^phy, s_t^ment)
```

其中 mental state 不只包含情绪，还包括 identity、beliefs、attention focus、goals、intentions、preferences、values、personality、norms、obligations/prohibitions，以及群体 mental field、角色关系、态度与场景氛围。

对个体 `i`，论文可概括为：

```text
m_t^i = (
  identity,
  beliefs,
  attention,
  goals,
  intentions,
  emotions,
  dispositions,
  norms,
  constraints
)
```

这个 taxonomy 比 MeetingToM 宽得多。它覆盖了预测行动所需的多种变量，但也混合了短时状态与稳定属性。实时 Omni 产品不应仅凭几秒交互推断 personality、values 或长期 preference；这些字段需要更高证据门槛、用户授权和明确的过期策略。

### 6.1 Target observation 不是 global state

MWM 最关键的设计之一是 target-specific observation：

```text
o_t^target = (o_t^{target, phy}, o_t^{target, ment})
```

global state 可以包含信封已被移到抽屉；target observation 必须保留目标人物不知道这件事的事实。mental observation 还可包含 self-observation、first-order ToM，甚至 higher-order ToM。

这把经典 false-belief 问题变成了明确的系统接口：模型不能把 narrator 或摄像头知道的事实直接泄漏给 target。

### 6.2 Action 同时有物理 carrier 与社会意义

```text
a_t^target = (a_t^{target, phy}, a_t^{target, ment})
```

说话、移动、指向、等待、隐藏是 physical carrier；安慰、欺骗、拒绝、道歉、引导注意、维护面子是 mental/social meaning。同一句“没关系”在物理上都是发声，在社会意义上可能是安慰，也可能是结束冲突。

后继状态同时依赖当前物理/心理状态与动作的 carrier/meaning。这正适合全双工：backchannel 不是一个声学 token，而是具有“保持对方话权、表达持续关注”作用的社会动作。

## 7. MENTIS 的六阶段过程轨迹

![Mental world trajectory](/images/blog/mental-world-trajectory.svg "图 3：MENTIS 显式保存 state、target observation、action branches、successor states、value 与最终选择。")

MENTIS 是 training-free、可检查的 baseline。它不是端到端输出选项字母，而是保存六阶段 artifact：

1. **State parsing**：从故事解析当前 joint physical-mental state；
2. **Observation generation**：投影出目标人物在决策时刻能够观察到什么；
3. **Action decomposition**：将六个候选动作拆成 physical 与 mental components；
4. **Branch simulation**：对每个动作模拟 coupled successor state；
5. **Value evaluation**：按 mental consistency、physical plausibility、social appropriateness 评分，并执行 safety/legality veto；
6. **Deterministic decision**：根据分支分数确定最终动作。

轨迹可写为：

```text
scene x
→ current joint state s_t
→ target observation o_t
→ {candidate action a_t^k}_{k=1..6}
→ {successor state s_{t+1}^k}_{k=1..6}
→ {value v_k}_{k=1..6}
→ selected branch
```

这类 process trace 的价值不只是“更可解释”。当答案错时，可以定位是 state parse 错、observation 泄漏、action 语义分解错、transition 不合理，还是 evaluator 偏好错。对全双工 Agent，这种组件诊断比最终 response quality 更有研究价值。

## 8. Menti-Bench 的 process gold 如何构造

Menti-Bench 有 `448` 条记录：`320` 条文本、`100` 条图片、`28` 条有声视频；每条提供六个候选动作，共标注 `2,688` 个 option-level successor states。`78%` 的记录至少包含两个人物，覆盖 interpersonal、object/resource、spatial/perceptual、risk/norm 四类 scene 与五个日常领域。

所有人物和场景均为 fictional；图片与有声视频是合成媒体，并经人工质检。视频最初制作 50 个 story，最终只保留 28 个。

### 8.1 从 source scenario 到 decision record

统一 conversion pipeline 可以概括为：

1. 将故事截断在 decision moment，保证未来尚未发生；
2. 显式写出 objects、agents、positions、occlusions、affordances 与 perceptible signals；
3. 固定 target agent；
4. 将问题改写为 target-action query；
5. 构造六个句法上都合理的 candidate actions；
6. 让选项在 physical feasibility、belief consistency、intention fit、norm compliance 与 social consequence 上产生受控差异；
7. 为每个 option 标注 successor joint state；
8. 确定 uniquely defensible best action；
9. 对 ambiguous record 重新裁决；
10. 平衡 option letter 与长度，并运行 blind options-only probe。

这不是从真实视频自动抽出的自然 trajectory，而是将已有 ToM 与 situated social-reasoning scenario 改写为 state-observation-action-transition 格式的**人工构造 counterfactual decision record**。

### 8.2 `unspecified` 与 `Unknown` 是两种完全不同的空值

这是整篇论文最值得借鉴的 annotation convention：

```text
global state 中 unspecified:
  该字段对当前故事没有被限定；
  模型填写或不填写都不应受罚。

target observation 中 Unknown:
  目标人物在 decision moment 不可能知道；
  如果模型填入真实值，就是 perspective leakage。
```

普通 JSON benchmark 经常把 `null`、`unknown`、`not applicable` 和 `not annotated` 混在一起。MWM 明确赋予它们不同 evaluation semantics，这对在线 ToM 尤其重要：

- `not observed`：系统尚未获得证据；
- `unobservable`：当前传感器无法得知；
- `uncertain`：存在多个可解释假设；
- `not applicable`：该字段对当前实体无意义；
- `superseded`：先前假设被新证据推翻；
- `withheld`：隐私或权限不允许系统持有。

### 8.3 媒体数据的质量控制

Text subset 经过 structural cleaning、minimal question、show-don't-tell rewriting、两轮 adversarial option balancing、独立模型 options-only probe 与 manual re-adjudication。

Image subset 检查 information coverage、temporal order 与 answer leakage；生成图片还经过 structural validation、two-tier cross-model review 与 carrier-solvability probe。有问题则重新生成或手工修正，人物名称必须在 scene anchor 中获得视觉 identity anchor。

Video subset 从 per-shot script 与 dialogue 开始，人工检查视觉和音频是否 faithfully realize annotated scene。论文还披露曾有 26 个 media records 缺失 gold process，之后 back-fill 并做完整 manual re-audit。披露修复过程比只写“经过质检”更有可信度。

### 8.4 当前公开数据能看到什么

截至本文核验时，[MENTIS 代码仓库](https://github.com/mental-world/Mentis) 已公开 Pydantic schema、pipeline 与两个 demo input；[Menti-Bench](https://huggingface.co/datasets/mental-world-model/menti-bench) 发布输入场景、target、question、options 和 answer。仓库 README 明确说明：论文中的 intermediate gold annotations **没有随 benchmark 发布**。

因此不能把论文附录里的 process gold 误写成用户已经可以直接下载的 JSON 字段。官方当前可见 input 形如：

```json
{
  "sample_id": "demo_1",
  "modality": "text",
  "target_agent": "Ben, who is looking for the letter",
  "question": "What will Ben most plausibly do next?",
  "options": [
    {"option_id": "A", "action_description": "Look on the table"}
  ],
  "story": {"text": "..."},
  "answer": "A"
}
```

MENTIS 推理时再生成 `WorldState`、`Observation`、`ActionPlan` 和 `BranchScore`。这与人工 process gold 是两个不同层次。

## 9. 结果诊断：最大的瓶颈不是读状态，而是模拟变化

论文在八个 world-model backbones 上测试，Full MWM 均优于对照。平均消融为：

| 消融 | 平均 F1 变化 |
| --- | --- |
| 移除 mental channel | -12.1 |
| 移除 physical channel | -16.5 |
| physical/mental 独立转移 | -6.4 |

最佳模型的 Full MWM `S6 = 90.7`，人类为 `98.5`。Oracle intervention 更能说明问题：

| 注入的 gold stage | 增益 |
| --- | --- |
| gold state | +2.8 |
| gold observation | +1.7 |
| gold action decomposition | +0.7 |
| gold transition | +3.5 |
| 四阶段全部 gold | +6.3 |

最大的单阶段提升来自 gold transition。也就是说，模型不仅要识别“他很焦虑”，更难的是预测“如果此刻打断、追问、沉默或执行动作，这个人与场景会怎样变化”。

这对全双工 Omni 的启示非常直接：ToM 不应止于 perception head。真正有价值的是 **action-conditioned mental transition**：

```text
如果我现在附和，对方会继续说还是误以为我同意？
如果我现在澄清，会降低误解还是增加打断成本？
如果我直接执行工具，用户之后改口的回滚成本是多少？
如果我保持沉默，对方是否会认为系统没有听见？
```

## 10. 从 snapshot label 到 online belief trajectory

MeetingToM 是 snapshot/window label，MWM 是单步 process gold。全双工系统需要的是第三种结构：连续在线、可修订、与动作相连的 belief trajectory。

![Label versus trajectory](/images/blog/tom-label-vs-trajectory.svg "图 4：短窗标签、单步 counterfactual process 与在线可修订 belief trajectory 的监督目标不同。")

### 10.1 为什么一个 dominant label 不够

考虑 8 秒交互：

```text
0–2s  用户专注倾听
2–4s  因术语陌生而困惑
4–6s  通过上下文形成猜测，开始犹豫提问
6–8s  明确反对模型结论
```

如果整个窗口只标 `Cognitive Conflict`，模型不会学到何时从等待切换到解释；如果只标 `Confused Bewilderment`，又会忽略最终立场。对实时策略来说，**转移时刻和修订路径**比末端类别更重要。

### 10.2 建议的在线 schema

下面是面向研究的建议，不是两篇论文的官方格式：

```json
{
  "time_ms": 8120,
  "target": "participant_B",
  "state_type": "confusion",
  "status": "hypothesis",
  "confidence": 0.64,
  "evidence": [
    {"modality": "gaze", "event": "rapid_shift", "time_ms": 7780},
    {"modality": "speech", "event": "clarification_question", "time_ms": 8050}
  ],
  "perspective": "assistant_inference",
  "valid_from_ms": 7600,
  "valid_to_ms": null,
  "alternatives": {
    "cognitive_conflict": 0.23,
    "hesitation": 0.13
  },
  "action_implication": "slow_down_and_clarify",
  "depends_on": ["event_402", "event_407"],
  "supersedes": null,
  "superseded_by": null
}
```

每个字段都有必要性：

- `target` 防止把一个人的状态传播给整个群体；
- `perspective` 说明这是 assistant inference、用户 self-report 还是第三方 observation；
- `confidence` 和 `alternatives` 保留多假设；
- `evidence` 与 `depends_on` 支持 provenance audit；
- `valid_from/to` 表示状态只在局部有效；
- `supersedes` 支持纠正和回滚；
- `action_implication` 把 ToM 与交互策略连接起来。

### 10.3 标签应分成 observation 与 inference

不要直接标：

```text
user is confused
```

更可靠的两层表示是：

```text
observation:
  user asked “what does that term mean?”
  gaze shifted between diagram and assistant
  speech contained a 1.2s hesitation

inference:
  confusion 0.64
  disagreement 0.23
  hesitation 0.13
```

当用户随后说“我听懂了，我只是不同意前提”，系统可以保留 observation，撤销旧 inference，并将新 self-report 置于更高证据优先级。

## 11. 在线 ToM 如何影响全双工策略

![Online ToM policy](/images/blog/tom-omni-online-policy.svg "图 5：实时多模态证据更新 mental belief distribution，再驱动等待、附和、澄清、行动与修订。")

全双工的外在动作只有几个：听、说、附和、打断、让出话权。但它们背后的决策需要 ToM：

```text
持续观测
→ 更新用户状态分布
→ 估计当前动作的后继状态
→ 比较交互收益、误判风险与延迟
→ 选择低成本动作
→ 观察用户反应
→ 修订先前假设
```

### 11.1 Confusion 与 disagreement 需要不同动作

- confusion 高：降低术语密度、换例子、询问具体卡点；
- disagreement 高：不要重复解释，先询问反对的前提；
- hesitation 高：留出思考时间，避免立即填满沉默；
- disengagement 高但证据弱：用简短确认，不要做人格判断。

静态分类精度相同的两个模型，可能因为策略映射不同而产生完全不同的用户体验。

### 11.2 Backchannel 与 hidden dissent 不应共享规则

点头和“嗯”可能只是 conversational receipt，不代表接受结论。全双工系统应分别维护：

```text
floor signal: please continue
epistemic signal: I understand
stance signal: I agree
commitment signal: you may execute
```

用户说“嗯，你继续”最多支持前两个，不能自动升级为支付、提交或取消订单的授权。

### 11.3 ToM 的价值不是更大胆，而是更会选择低风险动作

模型不确定用户是困惑还是反对时，最佳动作常常不是猜中标签，而是提出最小区分问题：

```text
“你是对这个术语不清楚，还是不同意这个假设？”
```

这个动作同时获得信息并降低用户纠正成本。可交互 ToM 的评价应奖励 **value of information**，而不只是 hidden-state accuracy。

## 12. 五个从根因出发的 Toy Experiments

按照“问题在哪里 → 为什么发生 → 最小实验 → 再扩大规模”的路线，可以先做五个低成本实验。

### 12.1 Confusion vs Disagreement

**问题**：模型把皱眉和否定性语气统一预测为 negative state。

**根因假设**：数据强调 valence，没有区分内容理解与立场。

**Toy**：构造同一句话、相同面部负向信号，但后续分别是“能再解释一遍吗”和“我理解，但这个前提不成立”。让模型在每 500 ms 输出二元分布与证据。

**指标**：首次可区分时刻、最终 Macro-F1、错误动作成本、澄清后修订率。

### 12.2 Backchannel vs Hidden Dissent

**问题**：模型把点头当作同意。

**根因假设**：训练标签将 turn-management cue 与 stance cue 合并。

**Toy**：控制语言完全一致，只改变 gaze、posture 与后续行动；分别预测 `continue_signal` 与 `stance`。

**指标**：两个 head 的校准、伪共识检出率、误把附和当授权的比例。

### 12.3 Early/Late Revision

**问题**：模型一旦形成 mental label 就不愿撤销。

**根因假设**：普通 SFT 只监督最终标签，没有 supersession 轨迹。

**Toy**：前 3 秒支持 confusion，后 3 秒用户 self-report 明确说是 disagreement。比较无 revision supervision、显式 supersession 和 Bayesian filtering 三种方案。

**指标**：revision latency、旧假设残留率、动作恢复时间。

### 12.4 Perspective Leakage

**问题**：模型把摄像头或旁白知道的事实当作目标人物知道。

**根因假设**：global state 与 target observation 使用同一上下文，没有 access mask。

**Toy**：制作最小 false-belief 场景，分别控制目标的视线、遮挡、音频可达性与离场时间。

**指标**：Unknown 保持率、泄漏率、target-conditioned action accuracy。

### 12.5 ToM-Conditioned Turn Policy

**问题**：更高 ToM 分类分数未必提升交互。

**根因假设**：perception 与 policy 解耦，标签没有进入动作选择。

**Toy**：固定同一个感知 backbone，对比 rule policy、label-conditioned policy 和 successor-simulation policy。

**指标**：任务成功率、用户打断次数、澄清轮数、repair cost、主观被理解感，而不只看 state accuracy。

## 13. 怎样构建真正的 ToM trajectory 标签

如果下一步要做 full-duplex Omni 数据，我建议四层标注，而不是一开始训练一个大模型猜心理：

### 13.1 Layer A：可复核 observation events

标注时间对齐的可见/可听事实：说了什么、谁看向谁、何时点头、语速变化、停顿、重叠、工具状态。它们应尽量不带心理解释。

### 13.2 Layer B：perspective access

对每个 participant 标记哪些事件可见、可听、已被告知或不可知。这个层防止 perspective leakage，是 higher-order ToM 的地基。

### 13.3 Layer C：uncertain mental hypotheses

人工不是被迫选唯一真值，而是记录候选分布、证据、置信度与 `Unknown`。若有 self-report，应单独存储 provenance，不能直接覆盖观察者标签而丢失冲突。

### 13.4 Layer D：action and successor effects

标注系统在不同动作下可能造成什么后果：用户继续、澄清、反感、纠正、授权或退出。可以先做少量 counterfactual annotation，再通过真实 A/B interaction 校正。

一个最小轨迹单元应包含：

```text
timestamp
observation event
who could perceive it
mental hypothesis distribution
evidence provenance
policy action
user reaction
state revision
task consequence
```

## 14. 评测不能只问“标签对不对”

### 14.1 Perception metrics

- event detection precision/recall；
- addressee identification；
- multimodal evidence localization；
- temporal alignment error。

### 14.2 Belief metrics

- Macro-F1 与 calibration；
- alternative hypothesis coverage；
- Unknown accuracy；
- perspective leakage rate；
- revision latency；
- stale-belief persistence。

### 14.3 Transition metrics

- successor-state consistency；
- action-conditioned user reaction prediction；
- physical/mental coupling；
- counterfactual ranking；
- intervention gain。

### 14.4 Product metrics

- task success；
- unnecessary interruption；
- clarification efficiency；
- user correction effort；
- perceived understanding；
- unsafe commitment from weak social cues；
- opt-out and correction success。

尤其要报告 conditional metrics。真实打断希望停止快，backchannel 则希望模型不断；用户困惑时澄清有价值，用户明确反对时重复解释反而增加成本。平均数会掩盖策略方向相反的事件。

## 15. 产品边界：ToM 越强，越需要克制

### 15.1 不要把 observer consensus 变成员工画像

MeetingToM 已明确警告不能用于员工 hidden intention、attitude 或 performance 决策。会议产品可以呈现“尚未确认一致”“建议逐人确认”，不应生成“某人消极抵抗”的持久标签。

### 15.2 不要从短片段推断稳定属性

MWM taxonomy 包含 preference、values、personality。研究框架可以容纳这些变量，不代表产品应默认推断或存储。短时 hesitation 不等于缺乏自信，单次退出视线不等于 disengaged personality。

### 15.3 用户自我陈述应能修正模型

当用户说“我不是没听懂，我只是不同意”，系统必须能够：

```text
保留原始 observation
标记旧 confusion hypothesis 为 superseded
提升 user self-report 的来源权重
撤销基于旧假设的解释策略
切换到 premise clarification
```

### 15.4 高风险行动不能由弱 ToM 信号授权

点头、微笑、沉默和“嗯”都不能替代明确的交易授权。ToM 可以决定是否询问，不应绕过确认。

### 15.5 用户需要知道系统在推断什么

产品至少应支持：关闭情绪/态度推断、查看或删除长期推断、纠正错误状态、限制会议分析用途，以及区分瞬时交互状态与长期 memory。

## 16. 两篇论文的局限应怎样理解

### 16.1 MeetingToM

- gold 是第三方行为解释，不是 first-person truth；
- Task 1 的 5 秒 dominant label 压平状态转移；
- taxonomy 混合 mental、interaction 与 behavioral states；
- 候选挖掘和模糊样本过滤改变自然分布；
- 类别不均衡需要 Macro 指标；
- 截至核验时，[公开仓库](https://github.com/oliviaziyi/MeetingToM) 仍标注 Coming Soon，无法检查实际 JSON schema。

### 16.2 Mental World Modeling

- Menti-Bench 只有 448 条，是诊断工具而非大规模 leaderboard；
- media subset 小，且图像与视频为合成；
- action space 固定为六选一；
- 只模拟一步 transition，不能证明长期 belief tracking；
- MENTIS 是 training-free prompting baseline；
- process gold 是规范化人工模型，不是真实人的内部状态；
- intermediate gold 当前未随公开 benchmark 分发。

这些局限不削弱两篇工作的价值，反而指出下一步应该补什么：自然连续轨迹、first-person validation、可修订假设、多步行动、真实交互干预与产品安全边界。

## 17. 面向全双工 Omni 的研究路线

### 17.1 从标签识别转向 belief maintenance

模型持续维护分布，允许 Unknown、替代假设、证据累积与状态过期。目标不是每帧都“给答案”，而是在证据不足时保持正确的不确定性。

### 17.2 从 mental perception 转向 action-conditioned transition

借鉴 MWM，为 `WAIT / BACKCHANNEL / CLARIFY / INTERRUPT / ACT / YIELD` 分别预测后继交互状态。最好的动作不是最像人，而是任务收益高、误判风险低、修复成本小。

### 17.3 从第三方标签转向多来源真值

结合：

- observable behavior；
- independent observer judgments；
- participant self-report；
- post-interaction correction；
- intervention outcome。

这些来源可能冲突，数据应保留冲突而不是强行压成一个 label。

### 17.4 从单人目标转向多人递归视角

多人会议需要维护：谁听到了什么、谁认为谁同意、谁的动作改变了谁的判断。可以先限制到 first-order ToM 和 addressee graph，再逐步测试 second-order belief，避免未经验证地声称无限递归心智建模。

### 17.5 从离线 benchmark 转向 closed-loop interaction

离线片段只能测解释能力。真正的 Omni ToM 必须在行动后接受用户反应：模型澄清是否真的减少误解，沉默是否让用户继续，错误推断是否能被纠正。只有 closed-loop outcome 才能验证 mental model 对产品有用。

## 18. 最终判断

MeetingToM 的贡献，是把 ToM 从单人故事题推进到真实多人会议中的个体、二人和群体推理，并用严格人工流程说明“可观察社会解释”怎样成为 benchmark label。MWM 的贡献，是证明 mental variables 不能只作为答案后的 rationale，而应进入 target observation、candidate action、successor state 和 value evaluation。

但全双工 Omni 的核心目标不应是更自信地猜用户心理。它应该是：

> **在连续时间中，根据可追溯的多模态证据维护关于他人心智状态的概率性、视角化、可修订 belief，并利用这个 belief 选择更合适的等待、附和、澄清、行动与修复策略。**

自然回复、倾听和打断只是外在表现。内在本质是持续回答五个问题：我观察到了什么？对方可能知道和想要什么？哪些解释仍然成立？我的动作会怎样改变对方与任务？新证据到来后应该撤销什么？

如果一项 ToM 研究不能改善这些决策，它最多是更精细的离线社会分类；只有当 mental belief 进入 closed-loop policy，并且允许不确定、纠正和审计时，它才真正成为全双工 Omni Agent 的能力。

## 参考资料

- Ziyi Wang et al. [MeetingToM: Evaluating Multimodal LLMs on Theory-of-Mind Reasoning in Multi-Party Meetings](https://arxiv.org/abs/2607.19235), 2026.
- MeetingToM authors. [Official GitHub repository](https://github.com/oliviaziyi/MeetingToM). 截至本文核验时数据与脚本标注为 Coming Soon。
- Hao Fei and Yiran Zhao. [Mental World Modeling](https://arxiv.org/abs/2607.27201), 2026.
- Mental World Modeling. [Project homepage](https://mental-world.github.io/).
- Mental World Modeling. [MENTIS reference implementation](https://github.com/mental-world/Mentis).
- Mental World Modeling. [Menti-Bench dataset page](https://huggingface.co/datasets/mental-world-model/menti-bench).
- Jean Carletta et al. [The AMI Meeting Corpus](https://link.springer.com/article/10.1007/s10579-005-1475-8), 2005.

> 资料边界：本文中的 benchmark 数量、标签、流程、结果与限制来自两篇论文及其当前公开仓库；“在线 belief trajectory schema”、五个 Toy Experiments 和面向全双工产品的研究路线是本文在这些证据上的方法设计，不是论文作者已发布的官方格式或实验结论。

# Omni Model Memory：为真正做事的 Agent 设计状态

> 从交互失败、第一性原理、Toy Experiment 到可部署产品。本文的核心判断是：Omni Model Memory 的首要目标不是记住更多内容，而是在持续做事时，让下一步动作依赖正确、最新、被授权、可追溯，并且与用户实际感知一致的状态。

谈到大模型记忆，最常见的反应是扩展 context window、接入向量数据库、总结历史对话，或者收集数据训练一个更会检索的模型。这些方法都有价值，但它们经常跳过了最重要的问题：**产品到底因为什么失败？**

一个只能回答问题的模型，记错一条信息可能只会生成一句错误答案；一个能发消息、改日程、取消订单、编辑视频和控制设备的 Omni Agent，记错同一条信息会改变现实世界。问题因此不再是“历史里有没有这句话”，而是：这条状态是谁说的、何时成立、是否仍有效、用户是否确认、哪些动作依赖它、发生纠正后如何撤销，以及系统能否解释自己为什么还记得。

这篇文章不从“应该训练什么模型”开始，而从六个产品失败开始。对每个问题，我采用同一条研究链路：

```text
观察具体失败
→ 追问失败为什么发生
→ 提出可以被证伪的根因
→ 用最小 Toy Experiment 隔离变量
→ 通过反事实干预验证因果关系
→ 扩展到大规模、多用户、长时程实验
→ 只有系统方案仍不够时，才讨论训练
```

## 1. Memory 不是长上下文：先看 Agent 怎么把事情做错

考虑一个实时语音客服场景：

```text
用户：帮我取消昨天买的耳机……
用户：不对，不要取消，查一下物流。
```

一个“记忆很好”的系统可能完整保存了这两句话，却仍然取消了订单。原因可能包括：第一句被提前写成稳定意图；取消工具已经提交；第二句只更新了 transcript，却没有使旧计划和旧工具调用失效；助手生成了“已经为您取消”，虽然这段语音还没真正播放，却被写入共同语境。

这里缺少的不是更多历史，而是以下契约：

- `不对` 是一次 supersession event，必须使旧 intent 失效；
- 查询是低风险、可撤销动作，取消订单是高风险、需确认动作；
- 工具执行有 `prepared / committed / cancelled / failed` 状态；
- 模型生成的内容不等于用户已经听到的内容；
- 被撤销的状态不能继续进入长期用户画像；
- 任一动作都应能追溯到它依赖的状态和授权。

这就是本文对 memory 的操作性定义：

> **Memory 是一个受部分观测约束、带生命周期和动作依赖的状态系统。它不是历史文本的同义词。**

![Memory problem stack](/images/blog/omni-memory-problem-stack.svg "图 1：长上下文、对话历史与检索只解决可见性；真正做事还需要有效性、授权、动作依赖、共同语境和回滚。")

## 2. 第一性原理：为什么交互式 Omni Agent 必须有 Memory

从最底层看，交互式 Omni Agent 同时具备三个性质。

**它是部分可观测的。** 模型在时间 `t` 看到的只是音频片段、当前视频帧、屏幕局部、用户动作和异步工具事件。用户真实目标、物体身份、他人信念和业务状态通常不可直接观测。

**它是有状态的。** “这个”“刚才那个人”“还是用第二版人物”等表达，只有结合过去发生过的事件才能解释。工具也有运行中、成功、失败、取消和过期状态。

**它的动作有后果。** 回复一句话、发出一条消息、点击购买和删除文件的风险完全不同。状态错误会被动作放大。

因此，模型需要从观测 `o_t` 维护一个 belief state `b_t`：

```text
o_t = audio + video + screen + pointer + user action + tool event

b_t = belief(
  world state,
  user intent,
  task progress,
  entity identity,
  authorization,
  common ground,
  uncertainty
)
```

下一步动作不是直接由全部历史决定，而是由当前 belief、动作风险和预期收益共同决定：

```text
a_t = policy(b_t, action_risk, expected_utility)
```

于是，memory design 的优化目标不应是“尽可能存得多”，而应近似为：

```text
minimize(
  future decision regret
  + unsafe action risk
  + stale-state cost
  + retrieval latency
  + privacy exposure
  + maintenance cost
)
```

保留一切会扩大隐私和冲突面；压缩一切会丢失动作所需的结构。正确目标是：**以足够低的成本保存会改变未来决策的状态，同时明确它的来源、有效期和权限。**

## 3. 为什么 Omni Memory 比纯文本 Memory 更难

纯文本对话通常至少有清晰的 token 顺序。Omni 交互中的“过去”却来自多个不同速率、不同可信度的通道。

### 3.1 多个时钟不天然对齐

音频按毫秒推进，视频按帧推进，屏幕对象按 UI 事件变化，工具结果可能数秒后异步返回，用户听到的语音又受网络和播放缓冲影响。同一秒内，模型可能已经生成一句话、扬声器只播放了前半句、用户同时指向了屏幕上的新对象，而旧搜索刚刚返回。

如果系统只有一个线性 transcript，它无法表示“生成时间”“播放时间”“观察时间”“工具提交时间”和“业务生效时间”的差异。

### 3.2 观察不等于事实

摄像头看到一只猫，只能支持“画面里可能有猫”；它不能直接支持“用户养猫”。视觉 caption 把观察压成自然语言时，往往丢失边界、来源和不确定性。一旦这个推断进入用户 profile，后续检索会不断强化它。

[Personal Visual Memory](https://arxiv.org/abs/2605.28806) 指出，图像中的身份、归属和稳定偏好有时确实不会被文本明确表达，但这不意味着系统应把任意视觉推断升级为事实。相反，它说明 personal visual memory 必须保留对话语境和证据关系。[Do Agents Dream of False Memories?](https://arxiv.org/abs/2607.15657) 进一步表明，多模态长期记忆会成为 poisoning 与 injection 的攻击面；论文报告的黑盒攻击成功率分别达到 `61.6%` 和 `58.4%`。这些数字来自论文设定，不应外推为所有产品的真实风险，但足以否定“看见就永久写入”的设计。

### 3.3 用户实际感知不同于模型内部历史

实时语音系统中，模型可能已生成 20 个音频 token，用户只听到前 8 个便打断。如果系统把全部生成内容写入 history，它会把用户从未听过的解释、确认或承诺当作共同语境。

因此至少要区分：

- `generated`：模型内部生成，但还未播放；
- `played`：用户设备已播放到的位置；
- `acknowledged`：用户通过语言或动作确认；
- `committed`：可以进入共同语境的内容；
- `discarded`：因中断而永不应被引用的尾部。

### 3.4 共同注意需要跨模态实体绑定

“把这个发给刚才开会的那个人”同时依赖当前指针、当前帧、文档实体、会议事件、参会者身份和发送授权。任何一个 binding 错误都会造成 wrong-send。更强的视觉 encoder 可能提高对象识别，却不会自动解决“这个观察是否对应那份 artifact”或“用户是否授权发送给此人”。

## 4. 真正需要记住的不是一句话，而是不同种类的状态

把所有内容都塞进统一 vector store 看似简单，却会混淆不同生命周期。一个可工作的 Omni Agent 至少需要区分以下 memory。

| Memory 层 | 典型内容 | 生命周期 | 失败后果 |
| --- | --- | --- | --- |
| Sensory buffer | 最近数秒音频、帧、指针、播放位置 | 毫秒到秒 | 错过打断、绑定错对象 |
| Working / task state | 当前意图、slot、待办、工具依赖 | 秒到小时 | 重复执行、旧计划未撤销 |
| Episodic event log | 谁在何时观察、说、做了什么 | 天到长期 | 无法追溯与重建 |
| Semantic / entity memory | 人、组织、设备、稳定关系和偏好 | 长期但可修订 | 人物合并、过期画像 |
| Commitment memory | 已确认约束、授权、承诺、共同语境 | 任务或会话 | 未授权执行、引用未听内容 |
| Artifact / project state | 文件、版本、局部编辑、依赖、反馈 | 项目周期 | 覆盖成果、无法复原版本 |
| Procedural / skill memory | 已验证流程、工具参数、失败模式 | 长期版本化 | 每次重新规划、重复犯错 |
| Policy / privacy ledger | 可见范围、保留期、撤销、用途 | 法规与用户设置决定 | 越权、无法删除、跨人泄露 |

这里的关键不是部署八个数据库，而是保持八种语义。实现可以是 event store、关系库、对象存储和向量索引的组合；但接口必须阻止一个低置信视觉推断被当成已确认授权，也必须阻止一个已经 superseded 的 task state 被 retrieval 排在最前面。

![Layered memory architecture](/images/blog/omni-memory-layered-architecture.svg "图 2：Omni Memory 是分层状态架构；不同层具有不同速率、可信度、保留期和动作权限。")

## 5. Memory item 应该有什么：从文本片段升级为状态对象

最小 memory item 可以写成：

```json
{
  "id": "mem-1842",
  "type": "observation | fact | hypothesis | commitment | action | artifact | skill",
  "subject": "order-391",
  "predicate": "requested_action",
  "value": "cancel",
  "source": {
    "modality": "speech",
    "event_id": "audio-882",
    "speaker": "primary-user"
  },
  "observed_at": "2026-07-30T10:03:21.440+08:00",
  "valid_from": "2026-07-30T10:03:21.440+08:00",
  "valid_to": null,
  "confidence": 0.72,
  "status": "tentative",
  "supports": ["event-audio-882"],
  "contradicts": [],
  "supersedes": [],
  "authorization_scope": "prepare_only",
  "visibility": "session",
  "retention": "until_task_closed",
  "action_dependencies": ["tool-cancel-17"]
}
```

这不是为了把 schema 设计得复杂，而是为了回答几个产品必须回答的问题：

1. 为什么记住？
2. 它是观察、推断，还是用户确认的事实？
3. 现在还有效吗？
4. 与哪条新状态冲突？
5. 哪些未完成动作依赖它？
6. 用户撤销后应删除、失效还是保留审计记录？
7. 它能被哪个会话、设备、协作者和工具使用？

如果一个系统无法回答这些问题，它拥有的更像是不可控的历史缓存，而不是 agent memory。

## 6. Memory 的核心是生命周期，不是 Vector Database

完整生命周期应是：

```text
Observe
→ Bind entity and time
→ Decide whether to write
→ Store with provenance, confidence and scope
→ Retrieve for a concrete action
→ Reconcile conflict and staleness
→ Prepare / commit / revise / revoke
→ Consolidate into fact, artifact or skill
→ Forget, archive or retain for audit
```

![Memory lifecycle](/images/blog/omni-memory-lifecycle.svg "图 3：写入只是生命周期中的一步。绑定、冲突协调、提交、撤销与遗忘共同决定 memory 是否能安全服务于动作。")

### 6.1 写入必须有门槛

不是每个观测都值得长期保存。可以先问：这条状态是否可能改变未来动作？是否已经存在？是否包含第三方隐私？它是稳定事实还是一次性上下文？用户是否允许跨会话使用？

一个实际的 write policy 可以将内容分成：

- 不写：环境噪声、重复观测、无动作价值的信息；
- 写入短期 task state：未确认 slot、当前指代、临时偏好；
- 写入 episodic log：重要事件及其来源；
- 升级为 semantic fact：多次证据一致或用户明确确认；
- 写入 commitment ledger：用户授权、系统承诺、不可逆动作确认；
- 沉淀为 skill：经过多次成功执行和验证的流程。

### 6.2 检索必须由动作问题驱动

“与当前 query 最相似”不是唯一目标。准备发消息时，系统更需要收件人身份、最新授权和附件版本；回答“我上次什么时候去医院”时，才需要长期 episodic retrieval。

检索接口应表达 action contract：

```text
retrieve(
  purpose = "send_artifact",
  required_types = [entity, artifact, authorization],
  freshness = "latest_valid",
  confidence_floor = 0.90,
  visibility = "current_user_only"
)
```

### 6.3 冲突不是删除旧值，而是建立状态演化

“我住在厦门”和“我现在搬到北京”可能是时间更新；“我不吃辣”和“今天可以微辣”可能是范围例外；“取消订单”和“不要取消”是明确撤销。将它们都处理成 last-write-wins 会丢失语义。

[SubtleMemory](https://arxiv.org/abs/2606.05761) 专门构造 complementary、nuanced 和 contradictory relation 的对照集合，说明细粒度关系判别应成为独立能力。产品实现则需要 `supersedes`、`exception_to`、`valid_during`、`contradicts` 等关系，而不是依靠 summary 自由改写。

### 6.4 遗忘必须可验证

删除向量索引中的一条 embedding 不代表系统真正遗忘。副本可能仍存在于摘要、缓存、项目 artifact、训练日志或其他设备。产品需要回答：删除传播到哪些存储？审计记录是否保留？保留的是内容还是仅保留删除事件？跨设备同步失败如何重试？

## 7. 六个产品失败，以及如何从根因设计 Toy Experiment

Toy experiment 的意义不是做一个漂亮 demo，而是**在最小环境里制造可控冲突，固定其他变量，只改变一个 memory 设计变量**。如果 toy 环境都不能稳定解释因果关系，直接扩展到十万条数据只会得到更难解释的平均分。

### 7.1 实验 A：用户中途改口，为什么旧动作没有消失

**观察。** 用户从“取消订单”改成“查询物流”，系统仍调用取消 API，或后续说“订单已经取消”。

**根因假设。** 失败来自 memory 缺少 supersession、tool dependency 与 playback state，而不是语言模型听不懂“不对”。

**Toy 环境。** 只提供一个订单、两个工具 `track_order` 与 `cancel_order`、一段固定音频和确定性的网络延迟。固定同一个模型与解码参数，只改变 memory representation：

1. Flat transcript；
2. Latest summary；
3. Append-only event log；
4. Event log + supersession；
5. Event log + tool state + playback position + two-phase commit。

**测量。** tool cancellation accuracy、不可逆误执行率、引用未播放承诺率、从纠正到状态一致的恢复延迟，以及需要用户额外重复多少信息。

**证伪条件。** 如果加入 supersession 和 tool state 后，在相同模型下错误没有显著下降，说明根因可能在流式感知或 action policy，而不是 memory schema。

**规模化。** 扩展到多 slot、多次改口、重叠语音、工具超时和多语言；报告 P50/P95 恢复延迟，不只报告平均准确率。

### 7.2 实验 B：屏幕里的“这个”，为什么发给了错误的人

**观察。** 用户指向屏幕说“把这个发给刚才开会的那个人”，系统选错文档或联系人。

**根因假设。** 视觉对象、pointer、会议事件、participant entity 和 authorization 分散在不同历史中，没有形成 typed binding graph。

**Toy 环境。** 两份视觉相似的 PDF、两位姓名相近的参会者、三次指代和一次鼠标移动。保持视觉 encoder 不变，比较 caption memory、frame embedding retrieval 与 typed entity-event graph。

**测量。** target artifact accuracy、target person accuracy、wrong-send rate、合理澄清率、从用户指向到系统建立 binding 的时间。

**关键对照。** 给系统一张更清晰截图，但移除 pointer timestamp；再保留低分辨率截图，却提供准确事件对齐。若后者更好，说明核心是时序绑定而非更高视觉分辨率。

**产品要求。** 高风险发送前，界面应展示“将第五版方案发送给王明”，让用户确认实体，而不是只播放模糊的“好的”。

### 7.3 实验 C：创作工具为什么恢复不了“第二版人物 + 第五版配色”

**观察。** 用户在海报、视频或 PPT 中连续修改，之后要求恢复旧人物但保留新配色。Agent 要么覆盖全部结果，要么重新生成一个近似版本。

**根因假设。** 聊天摘要无法表示 artifact version DAG、局部区域、工具参数、derived-from 和 rejected alternative。

**Toy 环境。** 三个 asset、五个版本、两类局部编辑和一次跨版本组合。比较 linear chat、flat artifact list 与 version graph。

**测量。** exact restoration、错误覆盖率、重复计算成本、人工修复时间和 provenance completeness。

[JarvisHub](https://arxiv.org/abs/2607.23588) 把 editable canvas 同时视为 workspace、external memory、action space 和 shared project state。这一方向的重要启示不是“所有产品都要做 canvas”，而是：对于做设计、视频和文档的 Agent，**artifact 本身才是主状态，聊天只是对 artifact 的操作日志。**

### 7.4 实验 D：多人场景为什么把群体规范当成个人偏好

**观察。** 群里大多数人吃辣，但小王不能吃辣；Agent 为小王订餐时仍选择辣菜。

**根因假设。** 系统将所有对话压成 single stream，丢失 speaker、subject、group norm 和 individual exception 的关系。

**Toy 环境。** 三个人、两个群体规则、三个个人例外、一次偏好更新。对比 session summary、speaker-separated retrieval 与 social entity graph。

**测量。** subject attribution、exception preservation、temporal update、cross-person leakage 和 abstention。

[SocialMemBench](https://arxiv.org/abs/2605.17789) 包含 430 个 persona、7,355 个 turn 和 1,031 个问题。论文报告四个开源 memory framework 的 question-weighted score 仅约 `0.12–0.18`，并观察到 single-stream conflation、temporal overwrite 和 entity merging 等错误。这里最重要的不是分数本身，而是它证明“召回了相关句子”仍可能无法支持正确的社会决策。

### 7.5 实验 E：一张照片为什么制造了永久用户画像

**观察。** 系统从朋友家的照片推断“用户养猫”，并在后续推荐、闲聊和健康建议中反复使用。

**根因假设。** memory 没有区分 observation、inference 与 confirmed fact，也没有 owner、confidence、source modality 和 expiration。

**Toy 环境。** 同一人物在不同地点与不同物体合影，加入一条文本纠正和一张对抗图片。比较 caption-to-profile、confidence-gated memory 与 provenance graph。

**测量。** false profile write、纠正后的残留率、攻击成功率、错误跨会话传播、用户发现并修复错误所需步骤。

**产品要求。** 用户应能看到“系统为什么认为我养猫”，并一键改为“朋友的猫”或“不要记住”。这不是设置页中的附属功能，而是多模态 memory 的安全主界面。

### 7.6 实验 F：Agent 为什么每次做同类任务都像第一次

**观察。** Agent 上周已经成功完成“读取实验结果、生成图表、插入论文并运行检查”，本周仍从头试错，重复使用失败参数。

**根因假设。** 系统只保存 factual/episodic memory，没有 procedural memory；成功 trajectory 没有被验证、抽象和版本化为 skill。

**Toy 环境。** 设计三个结构相同、表面名称不同的工具任务。第一次允许完整探索，后两次比较 transcript retrieval、raw trajectory replay 与 verified skill library。

**测量。** task success、工具调用数、重复错误率、迁移到轻微变化任务的成功率、skill 失效检测。

[Voyager](https://arxiv.org/abs/2305.16291) 展示了 executable code skill library 如何通过环境反馈和自验证积累能力，而不依赖参数微调。[SymbOmni](https://arxiv.org/abs/2607.12042) 则进一步探索从多模态操作经验中形成可复用 symbolic concept。它们支持一个更朴素的产品判断：先验证能否把成功流程保存为可执行、可测试、可回滚的外部 skill，再决定是否值得将其蒸馏到模型参数中。

![Toy experiment matrix](/images/blog/omni-memory-toy-experiments.svg "图 4：六个 Toy Experiment 分别隔离 supersession、entity binding、version graph、social attribution、provenance 和 procedural reuse。")

## 8. Toy Experiment 怎样才真的能证明根因

### 8.1 固定 backbone，不让“模型更强”掩盖系统变量

所有 memory 方案使用同一模型、同一 prompt budget、同一工具、同一温度和同一环境随机种子。否则更高分可能来自更强 reasoning，而不是 memory design。

[MemGym](https://arxiv.org/abs/2605.20833) 的价值正在于把动态 memory formation 放入 tau2-bench、deep research、SWE-Gym、code QA 和 WebArena-Infinity 等任务，并提出 memory-isolated score，尝试将 memory 与 reasoning/tool use 解耦。研究不应只问“带 memory 的 Agent 总分更高吗”，而要问“在控制基础任务能力后，memory 减少了多少由状态缺失导致的错误”。

### 8.2 使用确定性环境和最小可区分样例

工具结果、网络延迟和对象身份先固定。每个样例只改变一个关键关系，例如：

- 是否存在用户明确纠正；
- 联系人是王明还是王铭；
- 偏好是个人规则还是群体规则；
- 图片里的物体属于用户还是朋友；
- 某段语音已经播放还是仅生成。

如果输入变化同时改变十个因素，失败后无法知道应该修 memory、perception 还是 policy。

### 8.3 做反事实干预，不只做相关性比较

在相同历史上人工切换一条状态：将 `tentative` 改为 `confirmed`、将 `played_until` 从 20% 改为 80%、将 `superseded` 关系移除。观察动作是否按预期变化。

好的 memory system 必须满足可预测的反事实：未授权时不提交，授权后可提交；状态被撤销后依赖动作被取消；播放位置变化只改变共同语境，不应改变订单实体。

### 8.4 评估错误代价，而不是只算 QA accuracy

错误回答和错误退款不能等权。建议至少报告：

```text
decision regret
unsafe commit rate
wrong-entity action rate
stale-state usage rate
rollback correctness
clarification efficiency
user repair effort
privacy boundary violation
memory maintenance latency
```

### 8.5 预先定义收敛条件

Toy 实验不是无限调 prompt。开始前应写下：主要指标、证伪阈值、允许的澄清成本、最大 memory latency 和失败分类。如果方案只能靠不断增加特例规则维持，它可能没有找到正确抽象。

## 9. 从 Toy 到大规模：不是把样本数乘一千

一个合理的扩展顺序是：

### 阶段 1：最小因果验证

每个失败类型 20–50 个人工构造对照组，环境完全确定。目标是判断 representation 与 lifecycle 是否方向正确，而不是追求统计榜单。

### 阶段 2：组合压力测试

将两个变量组合，例如“多人 + 中途纠正”“视频指代 + 工具超时”“版本恢复 + 权限变化”。确认模块组合后不会产生新的状态不一致。

### 阶段 3：长时程仿真

运行数百轮、多 session、多设备和异步工具事件；主动注入网络延迟、重复事件、乱序返回、缓存丢失、跨设备同步冲突和用户撤销。

### 阶段 4：影子流量

在真实产品中只记录 memory proposal 和 planned action，不真正执行高风险动作。将系统决定与人工或现有生产系统比较，建立 failure ledger。

### 阶段 5：有限可逆执行

先开放搜索、预取、草稿和建议等可逆动作；删除、支付、发送和账户修改继续使用 prepare/commit 与显式确认。

### 阶段 6：长期产品指标

不仅测任务成功，还要看一个月后：用户纠正次数是否下降、错误画像是否累积、撤销是否真正传播、跨设备是否一致、存储和检索成本是否可控、用户是否理解系统记住了什么。

![Scale-up path](/images/blog/omni-memory-product-surface.svg "图 5：从因果 Toy 到产品上线，需要经过组合压力、长时仿真、影子流量和可逆执行，并配套用户可见的 memory control surface。")

## 10. 现有 Benchmark 已经告诉了我们什么

[LongMemEval](https://arxiv.org/abs/2410.10813) 包含 500 个问题，覆盖 information extraction、multi-session reasoning、temporal reasoning、knowledge update 和 abstention。论文将长期 memory 拆为 indexing、retrieval、reading，并报告商业聊天助手与长上下文模型在持续交互中出现约 30% 的准确率下降。它的重要贡献是把更新和拒答纳入长期 memory，而不只是查找旧事实。

[LoCoMo](https://arxiv.org/abs/2402.17753) 的长期对话平均约 300 turn、9K token，最多跨 35 个 session，并包含 QA、事件总结和多模态对话生成。它证明长时对话不能被短上下文 QA 替代，但其主要交互形态仍是“根据过去回答”。

[MemGPT](https://arxiv.org/abs/2310.08560) 用操作系统式分层存储和 virtual context 管理有限窗口；[Generative Agents](https://arxiv.org/abs/2304.03442) 将 observation、retrieval、reflection 和 planning 连接起来，使记忆影响行为而不仅是回答。这两项工作奠定了“memory 是 agent runtime 的一部分”的方向。

2026 年的新 benchmark 开始暴露更接近产品的问题：

- MemGym 关注 coding、web navigation、deep research 等实际做事中的动态 memory；
- SocialMemBench 关注多人身份、群体规则与个人例外；
- SubtleMemory 关注互补、细微变化与矛盾关系；
- LifeSide 关注长期陪伴中的 Memory-Emotion-Environment loop 和变化中的隐私边界；
- visual memory 工作开始研究显式/隐式证据以及恶意图像制造假记忆。

但仍有明显缺口：多数 benchmark 不连接真实的高风险 side effect；很少记录用户实际播放位置；很少同时评估 artifact version、tool state 和 authorization；删除与跨设备传播几乎没有统一协议；用户修复错误 memory 的成本也很少成为主要指标。

## 11. 产品真正还缺什么

### 11.1 Memory Inspector

用户需要看到系统记住了哪些稳定事实、当前任务状态和第三方信息，并能按“来源、用途、有效期、共享范围”筛选。单一的“开启/关闭记忆”开关过于粗糙。

### 11.2 Why remembered 与 evidence trail

每条重要 memory 都应回答“为什么记住”。视觉推断应能回到原图区域和对话上下文；工具状态应能回到调用与返回事件；用户偏好应标明是明确表达、重复行为还是系统推测。

### 11.3 Edit、revoke 与 scoped forgetting

“我不住北京了”不一定意味着删除过去经历；“不要再记住我的住址”则意味着撤销未来使用权限。产品必须区分事实更新、用途撤销、内容删除和审计保留。

### 11.4 Memory transaction

高风险动作应使用事务语义：读取一致快照、准备动作、确认授权、提交或回滚。用户纠正时，要原子地失效旧意图并取消依赖动作，不能只改一段摘要。

### 11.5 Playback-aware common ground

语音客户端应把实际播放进度作为事件返回。服务端不能凭生成进度猜测用户听到了什么。断网、蓝牙延迟、切换设备和用户按下停止都必须改变共同语境。

### 11.6 Artifact-native workspace

对于代码、PPT、视频、设计和研究项目，memory control surface 应围绕 artifact、版本、依赖和结果，而不是聊天气泡。用户需要比较、合并、恢复和标记“不要再用这个方案”。

### 11.7 Observability 与 replay

开发者需要重放一次失败：当时有哪些观测、检索到哪些状态、哪些冲突被忽略、动作依赖什么授权、哪个事件导致 commit。没有 replay，线上 memory 错误只能被归因于“模型偶发幻觉”。

### 11.8 Cross-device consistency 与 migration

手机、网页、耳机、车载和家庭设备可能同时更新状态。系统需要版本号、冲突解决、离线队列和 schema migration。否则“在手机上删除”不等于“车载助手不再使用”。

### 11.9 Privacy budget 与第三方边界

Omni 设备会看到旁人、房间和屏幕。默认用户授权不应自动覆盖第三方。memory write policy 需要本地过滤、用途限制、保留期和共享边界。

### 11.10 失败恢复，而不是假装永远正确

最成熟的产品不是从不出错，而是错误可见、可定位、可撤销且修复成本低。系统应有“我可能把两个人混淆了”“这个状态来自一次低置信视觉观察”等不确定性表达，而不是用自然流畅的语音掩盖状态错误。

## 12. 什么时候才需要训练、微调和标数据

先问三个问题。

**能否通过状态表示修复？** 如果问题来自缺少版本图、来源、有效期或 authorization，训练更大模型不会凭空创造这些产品状态。

**能否通过 deterministic controller 保证？** 删除、支付、发送等动作的两阶段提交，应由运行时强制，不应期待模型“学会谨慎”。

**能否通过检索与规则暴露正确证据？** 如果模型根本看不到播放位置或工具状态，标注再多也学不到不存在的观测。

只有在接口和状态正确后，仍存在以下问题，训练才成为合理下一步：

- 模型无法从流式多模态观测中稳定抽取 event；
- 对 subtle conflict、supersession 或 subject attribution 判断不稳；
- write/retrieve policy 在复杂组合中难以手写；
- 需要从大量成功 trajectory 中归纳可迁移 skill；
- 固定 controller 的澄清策略过于保守，需优化延迟与风险的权衡。

此时数据也不应只是“历史对话 → 最终答案”，而应包含 event、belief、provenance、memory operation、tool state、authorization、playback 与 rollback。训练目标是学会更好的状态估计和策略，而不是把数据库内容背进参数。

## 13. 一个可以真正开展的研究路线

### 研究问题 1：Playback-aware transactional memory

研究用户中断、工具并行和语音播放不同步时，如何维护 committed common ground。贡献可以是事件协议、rollback 算法和对应 benchmark。

### 研究问题 2：Action-conditioned memory retrieval

让 retrieval query 显式包含动作类型、风险、所需状态、freshness 和 authorization，比较它与语义相似度检索在真实任务中的 decision regret。

### 研究问题 3：Multimodal provenance and false-memory resistance

将观察、推断与确认分层，评估图像、音频和屏幕事件如何升级为 personal memory，以及对抗输入与用户纠正后能否彻底撤销。

### 研究问题 4：Artifact memory for long-horizon creation

构建可编辑 version graph benchmark，让 Agent 执行跨版本局部恢复、组合和回滚，并将人工修复时间作为主要指标。

### 研究问题 5：Memory-to-skill consolidation

研究何时一段成功经历足以沉淀为 skill、如何验证 skill 的适用范围、环境变化后如何使其失效。先用外部 executable skill 做实验，再考虑参数化蒸馏。

### 研究问题 6：Memory observability as a product metric

测量用户发现、理解和修复错误 memory 的成本。一个总分略低但错误可解释、可撤销的系统，可能比高 QA 分却不可控的系统更适合真实产品。

## 14. 我的判断：Omni Model 的竞争最终会变成状态质量的竞争

Omni 模型正在获得更自然的声音、更长的视频输入、更强的实时交互和更多工具权限。但当感知与生成逐渐商品化，真正决定产品能否长期工作的，将是它如何维护状态。

自然打断、附和和眼神交流只是表象。内在能力是：

- 持续更新用户目标，而不是固化第一句意图；
- 在证据不完整时保留多个假设，而不是过早写成事实；
- 将观察、推断、确认、授权和承诺分开；
- 知道用户实际看见和听见了什么；
- 让工具、artifact、记忆和对话共享一致的版本状态；
- 在新证据到来时撤销旧计划和依赖动作；
- 把成功经验沉淀为可验证 skill，同时让过期 skill 失效；
- 在记忆收益、动作风险、隐私和维护成本之间做决策。

因此，交互式 Omni Model 的本体确实更接近一个运行在连续时间上的 Agent Policy；而 memory 是这个 policy 对世界、用户、任务和自身行为的状态估计。没有可靠 memory，模型越能行动，错误后果越大。

一句话总结：

> **不要先问如何让 Omni Model 记住更多；先问它下一步要做什么、这个动作依赖哪些状态、这些状态为何可信、何时失效，以及出错后能否撤销。**

## 参考资料

1. [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813)
2. [Evaluating Very Long-Term Conversational Memory of LLM Agents (LoCoMo)](https://arxiv.org/abs/2402.17753)
3. [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
4. [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)
5. [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291)
6. [MemTools: A Unified Research Framework for Interoperable Agent Memory](https://arxiv.org/abs/2607.21404)
7. [Do Agents Dream of False Memories?](https://arxiv.org/abs/2607.15657)
8. [Personal Visual Memory from Explicit and Implicit Evidence](https://arxiv.org/abs/2605.28806)
9. [MemGym: a Long-Horizon Memory Environment for LLM Agents](https://arxiv.org/abs/2605.20833)
10. [SocialMemBench: Are AI Memory Systems Ready for Social Group Settings?](https://arxiv.org/abs/2605.17789)
11. [LifeSide: Benchmarking Agents as Lifelong Digital Companions](https://arxiv.org/abs/2606.04660)
12. [SubtleMemory: A Benchmark for Fine-Grained Relational Memory Discrimination](https://arxiv.org/abs/2606.05761)
13. [JarvisHub: An Open Harness for Canvas-Native Multimodal Creative Agents](https://arxiv.org/abs/2607.23588)
14. [SymbOmni: Evolving Agentic Omni Models via Symbolic Concept Learning](https://arxiv.org/abs/2607.12042)

本文中的架构、实验设计与产品判断是研究提案，不代表相关论文已经完成这些系统。文中带数值的结果均限定于对应论文的实验设置；真正的产品结论仍需要在固定模型、可审计状态与长期真实任务中验证。

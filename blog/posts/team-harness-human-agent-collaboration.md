# Human + Harness = Group：团队 Vibe Coding 的协作控制平面

当一个人同时带着多个 Coding Agent 工作时，瓶颈很快不再是“Codex 对话数量不够”，而是人已经没有精力持续查看每条管线。再增加 Agent 只会增加未读结果、互相冲突的修改和需要人工判断的异常。真正自然的下一步不是给负责人继续加会话，而是让更多成员各自带着 Agent 工作，再给整个团队增加一层协作 Harness。

这个想法可以从一句很直观的话开始：

```text
LLM + Harness = Agent
Human + Agent + Team Harness = Group
```

但“Human Harness”不能简单理解成监控员工、读取所有提示词和远程接管终端。那样确实能制造透明度，却也会制造表演性工作、权限滥用、隐私泄露和更强的信息噪声。一个真正可用的团队 Harness，核心不是“控制人不乱跑”，而是把**目标、责任、上下文、过程证据、干预和交接**组织成一个可持续运行的控制平面。

本文基于一段关于飞书机器人、Sub2API、Codex SID、团队 trace 和远程协作的讨论，进一步核对了 TeamAI、Omnigent、Sub2API、飞书开放平台、OpenTelemetry GenAI 语义约定、Temporal、MCP 授权、NIST Zero Trust 和团队认知研究。我的结论是：

> **Git 是团队的 artifact plane，聊天是 communication plane，Coding Agent 是 execution plane；团队仍然缺少一个把意图、过程、证据和组织决策连接起来的 coordination and control plane。**

资料核对截止到 2026 年 8 月 14 日。开源项目能力以其公开仓库为准；产品架构、数据模型、MVP 和实验设计是本文提出的工程方案，不代表这些项目已经实现。

![Team alignment layers](/images/blog/team-harness-alignment-layers.svg "图 1：Git、聊天、Agent trace 和 CI 分别保存结果、讨论、过程和证据；Team Harness 的作用是把四者绑定到同一个任务契约。")

## 1. 一页结论

- **Git 不够，但不是因为 Git 落后。** Git 很擅长保存可合并的代码结果，不负责解释任务为何这样拆、成员当前相信什么、Agent 正在验证哪条假设，以及领导应在何时介入。
- **全量对话记录也不够。** Prompt transcript 既可能信息过度，也可能信息不完整；它记录了说过什么，却不天然证明做过什么、结果是否真实、用户是否接受了结论。
- **团队真正需要的是共享任务状态。** 每个 `user + session/SID + workspace + branch` 必须绑定到明确目标、允许范围、验收证据和风险边界。
- **飞书适合做 control surface，不适合做唯一事实库。** 群聊、会议纪要、文档和卡片适合交互；持久任务状态、事件历史、权限与审计应在独立服务中保存。
- **网关可以统一配额与路由，但不应分发一个共享主密钥。** 每个成员和会话应获得可撤销、可限额、可审计的短期凭证；上游订阅共享还必须核对服务条款。
- **管理员干预应是显式事件，不是偷偷改 prompt。** `PAUSE`、`REQUEST_EVIDENCE`、`REPLAN`、`REASSIGN`、`TAKEOVER` 和 `APPROVE` 都应有操作者、理由、版本和审计记录。
- **负责人不需要读取所有 token。** 默认视图应展示目标偏离、长时间无证据、反复失败、危险动作、跨域修改和等待决策等异常。
- **交接不是转发聊天记录。** 三班倒或跨成员接续需要一个 handoff capsule：目标、已证实事实、未决假设、失败尝试、工作树状态、外部任务和下一步。
- **透明度不是越高越好。** 对模型输入输出、团队私聊、密钥和客户数据实行分级可见；OpenTelemetry 也把完整 input/output 定义为 opt-in 字段，而不是默认遥测。
- **先做 toy experiment，再做管理结论。** “能看到 trace”是否真的减少返工，需要用偏离发现时间、交接耗时、任务成功率和管理注意力成本验证。

## 2. 为什么 Git 只能解决结果对齐

团队成员提出“把所有东西放到一个 Git 项目里共同维护”，这个方向解决了重要的一部分：代码、配置、文档和测试都可以被版本化、评审和回滚。GitHub 也把 Pull Request 定义为提出、讨论和合并修改的协作机制，并通过 review、status check、CODEOWNERS 和 branch protection 约束结果。[1]

但在 Agentic coding 中，完成结果之前存在一段越来越长的中间过程：

```text
目标理解
→ 代码搜索
→ 假设形成
→ 尝试修改
→ 工具失败
→ 修改计划
→ 运行测试
→ 请求权限
→ 等待外部结果
→ 才形成 commit / PR
```

Git 通常只看见最后几步。一个成员声称“已经探索过方案 A”，Git 不知道他实际读取了哪些文件、运行了什么实验、排除了什么可能；一个 Agent 把算子抽离任务做成了 infra 调度优化，Git 只有在产生大量 diff 后才暴露偏离。

因此需要区分五类状态：

| 状态层 | 要回答的问题 | 合适载体 |
| --- | --- | --- |
| 意图 | 为什么做、什么不能做 | Task contract、决策记录 |
| 认知 | 当前相信什么、哪里不确定 | 假设与证据摘要 |
| 过程 | 做了哪些动作、遇到什么失败 | Structured trace |
| 产物 | 哪些文件和配置发生变化 | Git、对象存储 |
| 证明 | 是否满足验收条件 | CI、benchmark、设备日志 |

Git 主要覆盖产物和一部分证明。团队 Harness 要把其余三层与 Git commit、PR 和 CI run 关联起来，而不是取代 Git。

## 3. 为什么上传全部对话也不是答案

将每条 Codex 对话原样上传，看起来最完整，实际有四个问题。

第一，**粒度错误**。数万 token 的搜索、反思和重复报错不适合管理者连续阅读；管理者需要的是“目标是否偏离”“证据是否增长”“哪里需要决策”。

第二，**事实与叙述混在一起**。模型说“测试通过”只是叙述，退出码、报告文件和远端响应才是事实。原始对话没有自动区分 claim 和 evidence。

第三，**敏感信息暴露**。Prompt 可能包含源代码、客户资料、密钥片段、内部会议和个人信息。OpenTelemetry 的 GenAI agent semantic conventions 包含 `conversation.id`、`plan`、`invoke_agent` 和 `execute_tool` 等结构化 span，同时把完整 input messages、output messages 和 system instructions 标为 opt-in。[2] 这给出了正确方向：先采结构与结果，再按权限选择性采内容。

第四，**行为会被监控改变**。当每一句草稿都被上级长期查看，成员会减少探索、隐藏不成熟假设，或者把精力用于让 trace 看起来更漂亮。心理安全研究指出，团队成员能否放心承认错误和提出问题，会影响学习行为。[3] 因此系统目标应是更快暴露任务风险，而不是把每个人变成一份持续直播的绩效材料。

## 4. 从个人 Harness 到团队 Harness

个人 Agent Harness 解决的是一条模型执行管线：装载上下文、调用工具、约束权限、检查结果和恢复失败。[4] 团队 Harness 则多了三个实体：人、组织目标和成员之间的依赖。

可以写成：

```text
Team Harness =
  Identity and Session Binding
  + Task Contract
  + Shared Context
  + Structured Trace
  + Evidence Graph
  + Intervention Protocol
  + Handoff Protocol
  + Policy and Governance
```

更重要的性能公式不是“总 token 数”，而是：

```text
Team Throughput
  = useful verified work
  - rework from misalignment
  - waiting and handoff loss
  - merge and environment conflicts
  - manager attention cost
```

团队 Harness 的商业价值来自减少后四项，而不是让所有人生成更多代码。

## 5. 一个 AI 还是每个人一个 AI

“是否只需要一个 AI，所有人都和它对齐？”这个问题需要拆成两个角色。

### 5.1 团队协调 Agent

群聊机器人可以承担共享入口：回答项目状态、生成上下文包、发现冲突、提示决策、记录负责人确认。它拥有团队级读取权限，但默认不直接修改每个工作区。

### 5.2 成员执行 Agent

每个成员仍需要独立 Codex/Claude Code/其他 Coding Agent，因为成员拥有不同工作区、权限、硬件环境、专业背景和短期上下文。让一个中心 Agent 直接执行所有任务，会制造单点瓶颈、权限过宽和上下文污染。

因此合理结构不是“一个 AI 替代所有人”，而是：

```text
one coordination agent
        +
many scoped execution agents
        +
one shared event and policy plane
```

协调 Agent 负责共同状态，执行 Agent 负责局部闭环。两者通过任务和事件通信，不共享一个无限膨胀的聊天上下文。

## 6. 总体架构：Policy Monolithic，Execution Distributed

![Team Harness architecture](/images/blog/team-harness-control-plane.svg "图 2：飞书是交互控制面，Team Control Service 保存任务和权限；Agent Adapter、网关、Git/CI 与真实终端构成分布式执行面。")

架构可以分成六层。

### 6.1 Feishu Control Surface

负责群聊命令、状态卡片、异常提醒、审批、任务分配和手机端交接。飞书开放平台支持消息事件订阅，能接收单聊、群聊 @ 机器人或经敏感权限授权的群内消息；事件中包含 `message_id`、`chat_id`、`thread_id` 和发送者 ID。[5]

### 6.2 Team Control Service

保存 canonical task state、身份映射、依赖、策略、干预和审计。它才是事实源，不能把所有状态只放在飞书卡片里。

### 6.3 Context Service

读取飞书群聊、会议文档、设计文档、Issue、PR、代码和历史决策，生成带来源和版本的 context pack。飞书文档 API 可以在调用身份具有文档权限时获取纯文本内容，并有每应用每秒 5 次的公开频率限制，因此需要缓存、增量更新和退避。[6]

### 6.4 Agent Adapter

在 Codex、Claude Code 或其他终端附近运行，把不同 Harness 的事件统一为团队 schema，并接收可执行的控制事件。它不要求所有人更换同一个 CLI。

### 6.5 Gateway and Credential Broker

统一模型路由、额度、并发、成本和 session affinity。Sub2API 已提供多账号、用户 API Key、精确计费、sticky session、并发与限流等网关能力，但其 README 同时明确提醒上游服务条款风险。[7] 企业方案应把它视为可替换的数据面参考，而不是默认合规结论。

### 6.6 Artifact and Evidence Plane

Git、CI、benchmark、设备日志、模型输出和发布系统保存结果与证明。SLSA 将 provenance 定义为可以把产物追溯到复杂供应链来源的可验证信息；团队 Harness 同样需要把“谁的哪个 Agent 在什么环境下，用什么输入产生了哪个结果”连接起来。[8]

## 7. 身份绑定：不是只有 user + SID

`user + SID` 是入口，但生产系统至少需要下面的复合身份：

```text
tenant_id
project_id
human_user_id
role_id
device_id
agent_runtime
conversation_id / sid
workspace_id
repository + branch + worktree
task_id
credential_lease_id
```

原因是同一个成员可以同时运行多个会话，同一会话可以切换仓库，同一任务可能跨终端接续。只绑定 `user + SID` 会把“谁在聊”误当成“谁对哪个产物负责”。

绑定流程应是：

```text
飞书用户授权
→ 创建或领取 Task Contract
→ Agent Adapter 生成 session registration
→ Control Service 校验项目、角色与工作区
→ Credential Broker 发放短期 scoped token
→ 返回 task_id + session_id + policy_version
```

每次敏感操作都校验当前 task、scope 和 credential lease，不能因为某个 SID 曾被管理员看见就永久信任它。

## 8. 中转站应该发凭证，不应该发共享 Key

集中分发一个公共 Key 很容易实现，但会立即失去四种能力：无法精确撤销某个人、无法区分成本、无法限制工具范围、无法证明某次调用属于哪个任务。

正确做法是 Credential Broker：

| 能力 | 推荐设计 |
| --- | --- |
| 身份 | 每个用户、Agent、会话独立 principal |
| 凭证 | 短期 token，自动轮换，可单独撤销 |
| 配额 | 项目、用户、任务、模型四级预算 |
| 路由 | 模型与供应商策略，不把上游密钥下发到终端 |
| 审计 | 记录请求元数据、成本和状态，不默认记录敏感正文 |
| 合规 | 核对供应商条款、数据驻留和企业合同 |

MCP 的 HTTP 授权规范也采用 resource server、authorization server 和 OAuth 2.1 风格的访问 token，而不是让所有客户端共享一个长期秘密。[9] NIST Zero Trust 的基本原则同样是不因网络位置或资产归属授予隐式信任，而是在访问资源前分别认证和授权主体与设备。[10]

## 9. Task Contract：先约束任务，再观察过程

如果管理员只看 trace，却没有明确任务契约，那么“偏离”没有可比较的基准。每个任务应包含：

```yaml
task_id: op-extract-017
owner: user1
objective: 将目标模型的算子抽离为可复用模块
non_goals:
  - 不重构集群调度
  - 不修改公共训练框架
allowed_paths:
  - models/target/**
  - tests/operators/**
acceptance:
  - 原模型基准精度无显著下降
  - 指定数据集全部完成
  - 设备端与参考输出误差在阈值内
required_evidence:
  - commit_sha
  - benchmark_run_id
  - device_log_uri
risk_level: medium
budget:
  deadline: 2026-08-15T18:00:00+08:00
  model_cost_usd: 30
```

当成员去优化 infra 调度时，系统不是根据关键词主观判定“偷懒”，而是发现修改路径和工具行为超出 `allowed_paths/non_goals`，再触发澄清或变更申请。

## 10. Shared Context 不是把整个群聊塞进 Prompt

会议纪要和日常聊天确实是重要上下文，但全部注入会产生过时决策、闲聊干扰、隐私泄露和间接 prompt injection。OWASP 将来自网站、文件等外部来源并改变模型行为的内容定义为 indirect prompt injection，并指出 RAG 和微调不能完全消除风险。[11]

因此 Context Service 应生成可审计的 context pack：

```json
{
  "task_id": "op-extract-017",
  "generated_at": "...",
  "policy_version": "v12",
  "decisions": [
    {
      "id": "decision-42",
      "text": "本轮不修改 infra scheduler",
      "source": "feishu://chat/.../message/...",
      "approved_by": ["lead", "infra-owner"],
      "valid_from": "...",
      "supersedes": "decision-38"
    }
  ],
  "artifacts": ["git://repo@sha:path"],
  "open_questions": ["算子动态 shape 是否必须支持"],
  "access_scope": ["project:model-adaptation"]
}
```

关键是来源、时间、权限和 supersession。群里一句早已被推翻的话不能和正式决策拥有同等权重。

## 11. Trace 应该是一条事件流，不是一段录像

一个可跨 Harness 的最小事件可以是：

```json
{
  "event_id": "evt_...",
  "timestamp": "...",
  "tenant_id": "...",
  "project_id": "...",
  "task_id": "op-extract-017",
  "human_user_id": "user1",
  "session_id": "sid_...",
  "workspace_ref": "repo@branch:worktree",
  "type": "TEST_FINISHED",
  "summary": "operator parity test failed on dynamic shape",
  "evidence": {
    "exit_code": 1,
    "artifact_uri": "s3://.../report.json",
    "content_hash": "sha256:..."
  },
  "visibility": "project",
  "parent_event_id": "evt_...",
  "policy_version": "v12"
}
```

建议事件类型包括：

```text
TASK_ACCEPTED          PLAN_PROPOSED
CONTEXT_LOADED         HYPOTHESIS_UPDATED
TOOL_STARTED           TOOL_FINISHED
FILE_CHANGED           TEST_FINISHED
CLAIM_MADE             EVIDENCE_ATTACHED
BLOCKED                DECISION_REQUESTED
INTERVENTION_ISSUED    PLAN_REVISED
HANDOFF_CREATED        TASK_COMPLETED
```

OpenTelemetry 可以承担 trace/span 基础设施，但团队字段、任务状态和权限语义仍需要业务层定义。飞书事件采用至少一次投递，官方文档明确要求按 `message_id` 或 `event_id` 做幂等；Team Harness 的总线也应假设事件可能重复、延迟和乱序。[5][12]

## 12. 任务状态机比“在线/离线”更重要

![Task state machine](/images/blog/team-harness-task-state.svg "图 3：团队任务在执行、等待、偏离、阻塞、审查和交接之间流转；管理员干预是显式状态事件。")

一个任务至少有以下状态：

```text
DRAFT → ASSIGNED → ACTIVE → VERIFYING → DONE
                    ↘ BLOCKED
                    ↘ DIVERGED → REPLAN
                    ↘ WAITING_DECISION
                    ↘ HANDOFF_READY → TRANSFERRED
                    ↘ PAUSED / CANCELLED
```

系统不应该把“Agent 正在输出 token”当作进度。更可信的进度来自状态转换和证据增长：读取了目标模块、建立了基准、产生了首个可运行版本、发现动态 shape 失败、补齐设备端日志。

## 13. 管理员干预不是暗中注入 Prompt

直接通过网关修改其他人的 prompt 技术上可行，组织上却危险：成员不知道上下文为何改变，Agent 也无法区分用户意图、管理员命令和不可信聊天内容。

干预应成为一等协议：

| 干预动作 | 含义 | 默认效果 |
| --- | --- | --- |
| `REQUEST_STATUS` | 请求结构化状态 | 不暂停执行 |
| `REQUEST_EVIDENCE` | 要求补充可验证结果 | 阻止完成声明 |
| `CLARIFY_SCOPE` | 澄清目标或 non-goal | 更新 task contract |
| `PAUSE` | 暂停新动作 | 保留现场和运行中任务 |
| `REPLAN` | 基于新约束重规划 | 旧计划标记 superseded |
| `REASSIGN` | 更换负责人 | 先生成 handoff capsule |
| `TAKEOVER` | 临时接管会话 | 需要高权限、显式通知和审计 |
| `APPROVE_COMMIT` | 批准高风险动作 | 绑定准确参数和有效期 |

每次干预必须记录：谁发起、基于什么证据、修改了哪个契约版本、是否通知当前负责人、如何撤销。远程 shell 不应是普通飞书命令；最多先生成 command proposal，由终端侧策略引擎和当前操作者确认。紧急接管必须是单独权限，并限制时间和命令范围。

## 14. 飞书为什么适合做入口

飞书天然连接人、群、话题、文档、会议纪要和移动端通知。官方事件订阅支持服务端 Webhook，也支持 SDK 长连接；消息事件可以带 sender、chat、thread 和 message ID。[5][12]

因此它适合呈现：

```text
[任务卡]
算子抽离 / user1 / ACTIVE / 68%
最后证据：dynamic-shape parity test failed
风险：修改触及 infra/scheduler.py
等待：是否允许扩展任务范围？

[按钮]
查看证据  请求说明  保持范围  批准扩展  暂停
```

但必须把普通聊天与控制命令分开：

- 普通消息是 untrusted context candidate；
- 被授权用户通过卡片或签名命令产生 control event；
- 文档只有在访问身份有权限时才可读取；
- 控制服务校验当前 policy version，防止旧卡片重复执行；
- 消息重复投递必须幂等。

## 15. 负责人应该看异常，不是看直播

![Trace visibility ladder](/images/blog/team-harness-visibility-ladder.svg "图 4：默认展示任务状态和证据，只有在异常、授权和明确目的下逐级展开摘要、关键事件与原始内容。")

可以把可见性设计成五级：

| Level | 默认内容 | 适用对象 |
| --- | --- | --- |
| L0 | 任务状态、负责人、截止时间 | 团队成员 |
| L1 | 里程碑、证据、阻塞和风险 | 项目负责人 |
| L2 | 计划摘要、失败类别、关键工具事件 | 技术负责人 |
| L3 | 经脱敏的 prompt/output 片段 | 故障调查人员 |
| L4 | 原始会话、终端和敏感上下文 | 临时授权的安全/管理员 |

默认 dashboard 应按异常排序：

```text
scope drift score ↑
time since last evidence ↑
repeated failure count ↑
blocked dependency age ↑
risky action pending ↑
claim-evidence mismatch ↑
handoff readiness ↓
```

这把管理注意力从“轮询每个人”变为“处理系统无法自行关闭的例外”。

## 16. 交接必须把认知状态变成可传递对象

三班倒接续的难点不是新成员拿不到仓库，而是不知道上一班为什么走到这里。Transactive Memory System 研究关心团队是否知道“谁知道什么”，Shared Mental Model 研究则关心成员是否对任务、设备和协作方式形成相近表征。[13][14] Team Harness 可以把这两种组织能力部分外化。

一个 handoff capsule 应包含：

```text
Task contract version
Current milestone and confidence
Verified facts with evidence links
Rejected hypotheses and why
Open hypotheses
Changed and uncommitted files
Running jobs and external IDs
Credentials/permissions still needed
Known risks
Exact next two actions
Owner acknowledgements
```

接手者先运行 `HANDOFF_VALIDATE`：检查 commit、工作树、外部 job 和关键证据是否仍存在；确认后才将 owner 切换。这样“接手”不是复制摘要，而是一次状态重建测试。

## 17. 真实进度必须由 Evidence Graph 支撑

“有人说他做了探索，真实性怎么判断？”不能靠读取更多聊天解决。应把声明与证据分开：

```text
Claim: 已完成模型 A 在板端 B 的适配
  ├─ source commit: abc123
  ├─ build artifact: sha256:...
  ├─ environment: board-B / runtime-v7
  ├─ dataset: benchmark-X@version
  ├─ run log: run-8821
  ├─ metric: accuracy_delta = -0.03%
  └─ verifier: CI policy model-adaptation-v4
```

没有证据的 claim 可以显示，但状态只能是 `UNVERIFIED`。管理员看到的是缺失边，而不是根据语气判断成员是否可信。

## 18. Durable Workflow：任务不能依赖一条在线连接

飞书回调、Agent 终端、模型网关和 CI 都可能断开。Temporal 将 Durable Execution 定义为通过 Event History 保存工作流状态，使其在网络或服务器故障后从已记录事件恢复。[15] 不一定要在第一版引入 Temporal，但数据模型必须遵循同一原则：

- 接收事件后先持久化，再异步处理；
- 所有 handler 幂等；
- 长任务有 checkpoint；
- 外部动作有唯一 operation ID；
- 重试不会重复提交危险操作；
- 状态可由事件重放或 snapshot 恢复；
- Agent 断线后可重新绑定同一 task，而不是丢失事实。

## 19. 现有项目做到了哪里

### 19.1 TeamAI CLI

腾讯的 `teamai-cli` 直接将自己定义为 “The team harness for AI agents”。它使用 Git 仓库分发 skills、rules、docs、hooks 和 MCP 配置，在会话开始时同步；还会根据中断、纠正、拒绝工具调用和失败重试等 friction 信号，提示成员把经验整理进团队知识库。[16]

这很好地解决了“团队 Agent 使用同一套经验和规则”，但它主要是 Harness 资产与知识的 Git-native 分发，不是实时任务控制、负责人干预和跨成员 session handoff 平台。

### 19.2 Omnigent

Omnigent 把自己定义为跨 Claude Code、Codex、Cursor 等系统的 meta-harness。其公开 README 描述了多设备继续会话、监督多个 Agent、共享 session、共同驾驶、策略审批、预算限制和云 sandbox；同时标注项目仍是 alpha。[17]

它已经接近 session collaboration plane，但本文强调的组织层仍需补充：任务契约、飞书决策来源、成员职责、Evidence Graph、管理异常视图和组织治理。

### 19.3 Sub2API

Sub2API 是 AI API gateway，公开能力包括多账号、用户 Key、token 计费、sticky session、并发和限流。它可以成为调用与身份遥测入口，但它不是 Team Harness 的任务状态库，也不能单独解决上下文对齐和接管协议；其项目还明确提醒订阅转发可能违反上游条款。[7]

### 19.4 本文方案的位置

```text
TeamAI       → shared harness assets and team knowledge
Omnigent     → cross-agent session and runtime collaboration
Sub2API      → model gateway, quota, routing, session affinity
Team Harness → organizational task, context, evidence, intervention, handoff
```

它们不是互斥替代关系。最现实的产品可以复用前三者的数据面，把研发集中在第四层。

## 20. 安全边界：这套系统为什么可能非常危险

它连接了团队聊天、会议文档、源代码、模型请求、终端和管理员权限，等于把多个高价值信任域接在一起。主要威胁包括：

| 威胁 | 失败方式 | 必要控制 |
| --- | --- | --- |
| 共享 Key 泄露 | 无法定位和撤销 | 短期 scoped credential |
| SID 劫持 | 冒充成员或接管会话 | 设备绑定、重认证、租约 |
| 群聊 prompt injection | 普通消息变成控制命令 | 数据/指令隔离、签名 action |
| 跨项目泄露 | Agent 读取无关群和文档 | project namespace、ABAC |
| 管理员滥用 | 偷看原始对话或远程执行 | 分级权限、双人审批、审计 |
| 旧卡片重放 | 重复暂停、提交或授权 | event ID、nonce、版本检查 |
| Agent 伪造完成 | 自己写一份成功摘要 | 独立 CI 与 evidence verifier |
| 供应商条款风险 | 账号封禁或合规失败 | 企业合同、可替换 gateway |

“管理员可以干预所有人的 Codex”不是默认能力，而应拆成独立权限：`view_status`、`view_summary`、`request_evidence`、`pause_session`、`view_raw_trace`、`takeover_session`、`approve_external_action`。高权限动作设置短时授权和不可变审计。

## 21. 最小可行产品：一晚能做什么

“今晚就能实现”在功能 demo 层面并不夸张，但要严格限制目标。

### 21.1 MVP 可以做

```text
1. 飞书 user_id ↔ 本地 agent session_id 绑定
2. Sub2API 或网关记录 session、模型、token、状态
3. Agent Adapter 上报 START / TOOL / TEST / BLOCKED / DONE
4. 飞书卡片展示每人当前任务和最后证据
5. 管理员发送 REQUEST_STATUS / PAUSE / MESSAGE
6. 从指定飞书文档生成带版本的 context pack
7. 生成 handoff summary + git status + running job list
8. 所有控制动作写审计日志
```

### 21.2 MVP 不应该做

```text
1. 默认读取所有群聊和私聊
2. 暗中修改 prompt
3. 通过飞书直接执行任意 shell
4. 自动判断员工勤奋或欺骗
5. 把 token 数和在线时长当绩效
6. 允许 Agent 自己证明任务完成
7. 使用一个永久共享主 Key
```

第一版最好只有一个项目、十个以内用户、有限事件类型和三种干预动作。先证明“更早发现偏离和更快交接”，再扩展管理功能。

## 22. Toy Experiment：用板端模型适配验证

“在华为板子上适配多种模型、运行数据集并确认精度无损”是很合适的实验场景，因为任务重复、验收明确、环境真实，也容易出现依赖和设备差异。

![Team Harness experiment](/images/blog/team-harness-experiment.svg "图 5：用三组对照实验区分共享 trace、主动干预与完整团队 Harness 的真实收益。")

可以设计三组：

| 组别 | 工具 | 目的 |
| --- | --- | --- |
| A Baseline | Git + 飞书聊天 + 独立 Codex | 当前协作基线 |
| B Visibility | A + 结构化 trace + 状态卡片 | 测量透明度本身 |
| C Control | B + Task Contract + 干预 + handoff | 测量闭环控制 |

任务随机分配模型、板卡和数据集，并预埋几类问题：错误精度阈值、过时会议决策、设备运行时版本差异、Agent 擅自修改 infra、夜班交接和一次网关中断。

核心指标：

```text
task_success_rate
time_to_first_verified_artifact
time_to_detect_scope_drift
rework_lines / total_changed_lines
manager_attention_minutes
handoff_recovery_time
claim_evidence_mismatch_rate
unsafe_intervention_rate
developer_interruption_count
subjective autonomy and psychological safety
```

最重要的是区分效率来源。如果 B 比 A 好，说明可见性有效；如果 C 进一步提高，说明干预和任务契约有效；如果 C 让 manager attention 或成员中断显著增加，说明系统只是把会议换成了卡片轰炸。

## 23. 可以写成论文的研究问题

这套系统可以形成 AI for Management / CSCW / Software Engineering 交叉研究，而不只是工程 demo。

### RQ1：团队 trace 的最小充分表示是什么

比较原始 transcript、LLM summary、结构化事件和 Evidence Graph，测量管理者发现偏离的准确率、时间和认知负担。

### RQ2：何时干预能减少返工而不破坏自主性

训练或学习 intervention policy：继续观察、请求证据、澄清范围、暂停或接管。奖励同时考虑任务成功、返工、成员中断和管理时间。

### RQ3：Agent 是否可以改善团队 common ground

Clark 和 Brennan 将 grounding 描述为参与者建立足够共同理解的过程。[18] 可以在任务前后测试成员对目标、约束、依赖和角色的共同回答是否更一致。

### RQ4：Handoff capsule 是否优于聊天记录和文档总结

控制交接材料，比较新成员恢复工作所需时间、重复探索量和错误假设保留率。

### RQ5：透明度何时转化为指标博弈

逐步增加 trace 可见级别，观察成员是否增加无价值动作、减少承认不确定性，或把任务拆成更容易显示“有进度”的形式。

## 24. 商业上真正卖什么

这确实是一个直接的商业机会，但卖点不能只是“老板可以看员工的 AI”。真正付费对象是需要管理大量 Agentic work 的研发负责人、交付团队和外包/集成团队，他们购买的是：

- 更低的偏离发现时间；
- 更低的返工和合并成本；
- 更短的新成员和跨班次交接时间；
- 更少的四五小时状态会议；
- 对客户项目、模型调用和危险操作的可审计性；
- 在成员能力不均时仍能维持交付下限。

更合理的定价单位可能是 active human-agent seat、并发 session、受管项目和企业治理模块，而不是 token 加价。长期壁垒也不只是飞书机器人，而是：

```text
cross-harness adapters
organization event schema
task/evidence graph
intervention policy data
handoff quality
enterprise identity and audit integration
```

最大的产品风险是滑向电子监工。短期看，全量监控会让管理者感到掌控；长期却可能让优秀成员拒绝接入、普通成员学习刷指标、组织把系统产生的 summary 当成事实。产品必须把“透明度服务于任务”写进权限、保留周期和 UI，而不能只写在价值观页面。

## 25. 建议的工程路线

### Phase 0：只做事件和任务契约

定义 task、session、event、evidence 和 intervention schema；用假终端和飞书测试群验证幂等、权限和状态机。

### Phase 1：接一个真实 Coding Agent

只接 Codex 或一个最熟悉的 Harness，实现结构化事件、Git/CI 证据和三种干预：请求状态、暂停、澄清范围。

### Phase 2：上下文和交接

接飞书群、文档和会议纪要，但使用显式 allowlist；生成 versioned context pack 和可验证 handoff capsule。

### Phase 3：跨 Harness 与异常视图

增加 Claude Code 等适配器；负责人页面按偏离、阻塞和证据缺失排序，不展示“在线时长排行榜”。

### Phase 4：策略学习

积累足够真实数据后，才学习摘要、风险检测和干预时机。模型只能建议高风险干预，最终授权仍由 policy engine 和人完成。

## 26. 最终判断

你提出的直觉是对的：个人 Harness 已经把 LLM 变成更可靠的 Agent，下一层自然是让多个“人 + Agent”形成可协调的团队。只是公式不能停在：

```text
human + harness = group
```

更准确的表达是：

```text
Group = Humans + Scoped Agents + Shared Task State
        + Evidence + Intervention + Handoff + Governance
```

Git 继续负责结果，飞书继续负责人类沟通，Sub2API 或企业网关负责调用数据面，个人 Harness 继续负责每条执行管线。团队 Harness 的独特价值，是让负责人和成员围绕同一个可验证任务状态协作：知道目标在哪里、谁正在做什么、证据增长到哪里、何时发生偏离、谁有权介入，以及下一班如何无损接手。

一句话总结：

> **不要把团队 Vibe Coding 做成“老板读取所有 prompt”的监控平台；应当把它做成一个以任务契约为中心、以证据为进度、以异常为管理入口、以授权干预和可验证交接为闭环的 Human-Agent Team Control Plane。**

## 参考资料

1. GitHub Docs, [Collaborating with pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests).
2. OpenTelemetry, [Semantic Conventions for GenAI agent and framework spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/).
3. Edmondson, [Psychological Safety and Learning Behavior in Work Teams](https://doi.org/10.2307/2666999), 1999.
4. Tianyu Xie, [Harness 工程：把模型能力变成可靠执行，以及四个值得研究的开源项目](https://teeryxie.github.io/blog/agent-harness-engineering-four-projects/), 2026.
5. 飞书开放平台, [接收消息事件](https://open.feishu.cn/document/server-docs/im-v1/message/events/receive).
6. 飞书开放平台, [获取文档纯文本内容](https://open.feishu.cn/document/server-docs/docs/docs/docx-v1/document/raw_content).
7. Wei-Shaw, [Sub2API](https://github.com/Wei-Shaw/sub2api).
8. SLSA, [Provenance](https://slsa.dev/spec/v1.2/provenance).
9. Model Context Protocol, [Authorization](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization).
10. NIST, [SP 800-207: Zero Trust Architecture](https://csrc.nist.gov/pubs/sp/800/207/final), 2020.
11. OWASP GenAI Security Project, [LLM01:2025 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/).
12. 飞书开放平台, [事件订阅概述](https://open.feishu.cn/document/server-docs/event-subscription-guide/overview).
13. Lewis, [Measuring Transactive Memory Systems in the Field: Scale Development and Validation](https://doi.org/10.1037/0021-9010.88.4.587), 2003.
14. Mathieu et al., [The Influence of Shared Mental Models on Team Process and Performance](https://doi.org/10.1037/0021-9010.85.2.273), 2000.
15. Temporal, [What is Temporal?](https://docs.temporal.io/temporal).
16. Tencent, [TeamAI CLI](https://github.com/Tencent/teamai-cli).
17. Omnigent AI, [Omnigent](https://github.com/omnigent-ai/omnigent).
18. Clark and Brennan, [Grounding in Communication](https://doi.org/10.1037/10096-006), 1991.

> 证据边界：飞书事件、文档权限与频控、OpenTelemetry 字段、MCP 授权、Temporal durable execution、TeamAI、Omnigent 和 Sub2API 的能力来自公开文档或仓库。本文提出的 Team Harness 数据模型、管理员干预协议、visibility ladder、Evidence Graph、实验设计和商业判断属于架构推演，仍需通过真实团队实验验证。

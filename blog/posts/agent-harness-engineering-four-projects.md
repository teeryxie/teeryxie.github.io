# Harness 工程：把模型能力变成可靠执行，以及四个值得研究的开源项目

Claude Code 装好了，Skills 配了，MCP 也接了，为什么 Agent 仍会做到一半就宣布完成、声称已经发布但线上什么都没有，或者换一个会话便忘记昨天的决定？最直觉的解释是“模型不够强”。但在很多真实任务里，失败并不是因为模型不会写代码，而是因为系统没有给它**完成定义、真实环境、权限边界、外部状态、验证证据和失败恢复路径**。

这正是 Harness Engineering 想解决的问题。

本文从一份流传很广的 Harness 入门教程出发，但不复述其中未经核实的排行榜数字，也不把 `CLAUDE.md + Skills + MCP` 当作完整答案。我查阅了 Mitchell Hashimoto 的原始文章、Anthropic 关于有效 Agent 与长任务 Harness 的工程文章，并逐项阅读 OpenHands、mini-SWE-agent、Aider、Goose 的官方仓库和文档。我的核心判断是：

> **Harness 不是“模型之外的一切”，而是把模型的概率性决策约束成可执行、可验证、可恢复过程的运行时与控制平面。**

一个强模型可以提高单步决策质量；一个好的 Harness 则让错误更早暴露、影响范围更小、恢复成本更低，并让成功过程可以重复。

![Harness control loop](/images/blog/harness-control-loop.svg "图 1：Harness 不是模型外围的一圈配置，而是从任务契约、上下文、执行到验证、恢复和经验沉淀的闭环。")

## 1. 一页结论

- `Agent = Model + Harness` 是有用的入门公式，但 **Harness 不能被无限扩张成“除权重外的一切”**，否则它失去工程边界。
- Prompt、`CLAUDE.md`、Skills 和 MCP 都只是 Harness 的部件。它们分别提供指令、可复用过程和工具接口，不天然提供完成证明、权限隔离或失败恢复。
- Harness 的关键不是“让模型听话”，而是把任务变成一个带有**契约、状态、反馈和停止条件**的闭环。
- 最可靠的验证顺序通常是：确定性检查、真实环境测试、外部状态核对、独立实现交叉验证，最后才是 LLM judge。
- “另一个 Agent 来审查”并不天然独立。Actor 与 Verifier 若共享模型、上下文和错误假设，会产生相关失败。
- `CLAUDE.md` 不应只是愿望清单，也不应只是一份不断膨胀的翻车记录。它更适合保存稳定 invariant、项目 contract、危险边界和已被真实失败证明必要的规则。
- Harness 不是越厚越好。每个机制都应回答：它拦截什么失败、用什么证据判断、误拦截成本是多少、何时可以删除。
- OpenHands、mini-SWE-agent、Aider 与 Goose 并不是同一赛道的“四强”。它们分别代表**平台控制面、极简研究循环、Git 原生结对编程、MCP 可扩展本地 Agent**四条路线。
- 项目选择应从任务、风险和运行方式出发，而不是从 Star 数或功能清单出发。

## 2. Harness 为什么在今天变得重要

Mitchell Hashimoto 在 2026 年 2 月的文章 **My AI Adoption Journey** 中把第五阶段命名为 **Engineer the Harness**。他描述的不是一个新框架，而是一种工作方式：拆分清晰任务，把计划与执行分开，给 Agent 可以验证工作的手段，并在发现重复错误时改造环境，使下一次更难犯同类错误。[1]

这里最重要的变化是工程对象发生了转移：

```text
Prompt engineering: 如何让模型这一次更可能给出好答案？
Context engineering: 这一步应把哪些信息放进上下文？
Harness engineering: 如何让整个过程可执行、可观测、可验证、可恢复？
```

这三者不是替代关系。Prompt 和 Context 都位于 Harness 内部，但 Harness 还要处理模型看不到或不应自行决定的事情，例如：

- 当前工作区到底有哪些文件；
- 命令是否真的成功，退出码是什么；
- 浏览器中是否真实渲染；
- 一个 GitHub Pages URL 是否已经返回新内容；
- 数据库写入是否需要二次确认；
- 任务是完成了 3/10，还是 10/10；
- 上一个上下文窗口做了什么，哪些假设仍然有效；
- 失败后应该重试、回滚、换策略，还是请求人工判断。

模型能力提升会降低某些错误概率，却不会自动创造这些外部事实。只要任务跨越多步、多个系统或多个上下文窗口，Harness 就不再是可选装饰。

## 3. 先纠正一个过宽的定义

“Harness = 模型之外的一切”容易记忆，但不够可操作。按这个定义，操作系统、云厂商、显示器甚至办公室网络都属于 Harness，工程讨论会迅速失焦。

更严格的定义是：

> **Agent Harness 是围绕模型决策循环建立的运行时与控制平面。它负责装载任务状态和上下文，暴露可执行动作，实施权限和资源约束，收集环境反馈，判定结果是否满足契约，并支持恢复、追踪与审计。**

可写成：

```text
Harness =
  Task Contract
  + Context Loader
  + Agent Loop
  + Tool Runtime
  + State and Memory
  + Policy and Permissions
  + Verification
  + Recovery
  + Observability
```

Model 仍负责语义理解、规划、代码生成和不确定条件下的判断。Harness 不替代智能，而是规定智能**在哪里行动、看见什么证据、最多走多远、怎样证明完成**。

![Harness layers](/images/blog/harness-six-layers.svg "图 2：一个可操作的 Harness 可划分为契约与上下文、决策循环、工具运行时、状态与记忆、验证与恢复、权限与可观测性六层。")

## 4. 从 Guides / Sensors 到完整闭环

入门教程把 Harness 分成前置 `Guides` 和后置 `Sensors`，这是很好的第一步：前者告诉 Agent 应该怎样做，后者检查它是否做对。但生产系统还需要补上中间的 **Actuators** 与跨轮次的 **State**。

### 4.1 Guides：减少搜索空间

包括系统提示、项目说明、Skills、示例、工作流模板和工具文档。它们的作用不是强制，而是让模型更快找到合理路径。

### 4.2 Actuators：把意图变成受控动作

包括 shell、文件编辑器、浏览器、数据库、MCP、API、Git 和部署系统。一个工具接口是否清晰、返回值是否结构化、错误是否可区分，会直接影响 Agent 的上限。SWE-agent 论文把这类设计称为 Agent-Computer Interface；Anthropic 也强调工具文档与测试是 Agent 工程的核心。[2][3]

### 4.3 State：让新一步知道旧一步发生了什么

状态不只是聊天历史，还包括任务清单、工作树 diff、测试结果、预算、外部任务 ID、已获得的用户确认和可恢复 checkpoint。长任务若只依赖上下文压缩，往往会逐步把“做过什么”和“为什么这样做”混在摘要里。

### 4.4 Sensors：把环境事实送回循环

退出码、测试报告、DOM、截图、HTTP 内容、数据库读取和远端 commit hash 都是传感器。只有结果进入下一轮决策，Agent 才是真的闭环，而不是“生成后顺便跑一下命令”。

### 4.5 Governor：决定哪些动作不能只靠模型自觉

删除、支付、发布、迁移和权限变更不能只写一句“请小心”。真正的边界应由 allowlist、sandbox、审批门、幂等键、两阶段提交和不可变审计记录实施。

因此，完整过程应是：

```text
目标 -> 可检查契约 -> 装载最小上下文 -> 规划一步
     -> 权限检查 -> 在真实环境执行 -> 收集证据
     -> 验证契约 -> 完成 / 修复 / 回滚 / 请求人工
     -> 把必要经验写回稳定状态
```

## 5. 四类常见失败，根因不都在“后置缺位”

入门教程列出的四种失败很有代表性，但它们需要更精确地映射到机制。

| 症状 | 更深根因 | 优先机制 | 不充分的做法 |
| --- | --- | --- | --- |
| 提前交卷 | 没有机器可检查的完成契约 | manifest、计数器、acceptance tests、stop gate | 提示“必须做完” |
| 环境盲区 | 推理环境与交付环境分离 | sandbox、真实运行、浏览器/设备测试 | 只读代码自检 |
| 虚标完成 | Actor 自选容易通过的证据 | 外部 oracle、端到端测试、独立验收清单 | 再问同一模型“对吗” |
| 跨会话失忆 | 状态只存在于易丢失的对话 | Git、progress file、任务状态、决策记录 | 无限增长聊天历史 |

Anthropic 在长任务实验中观察到两个主要模式：Agent 试图一次完成过多工作，以及后续上下文留下半成品环境。其工程方案包括初始化 Agent、结构化 feature list、每次只做一个功能、描述性 Git commit、进度文件，以及把浏览器端到端测试作为完成证据。[4]

这不是证明某种固定模板“最佳”，而是给出一条可迁移原则：

> **把主观完成感替换为外部可读状态，把一次性生成替换为可恢复增量。**

![Failure map](/images/blog/harness-failure-map.svg "图 3：四类常见失败分别对应完成契约、真实环境、独立证据和持久状态；仅增加提示词无法形成硬约束。")

## 6. Prompt、Skill、MCP 与 Harness 到底是什么关系

### 6.1 Prompt 是策略先验，不是权限系统

Prompt 能说明目标、风格和约束，但它最终仍由同一个概率模型解释。高风险规则若只存在于 Prompt，就无法提供强制保证。

### 6.2 Skill 是可按需装载的过程知识

Skill 适合封装“如何做 PDF 验收”“如何发布博客”“如何审查前端”等稳定流程。它能降低上下文噪声并复用经验，但必须与实际工具和验收器连接，否则只是更长的说明书。

### 6.3 MCP 是工具协议，不是可靠性本身

MCP 解决工具、资源和提示如何被标准化发现与调用。它不自动解决：

- MCP Server 是否可信；
- 工具参数是否需要确认；
- 调用是否产生不可逆副作用；
- 返回“成功”是否等于业务状态已更新；
- Prompt injection 是否经由工具内容进入上下文。

接入更多 MCP 会扩大能力面，也会扩大攻击面和失败面。Harness 要做的是对工具进行分级、授权、追踪和结果核验。

### 6.4 Subagent 是并行执行方式，不是独立真相

用第二个 Agent 做 review 可以增加视角，但不能默认得到独立性。如果两个 Agent 使用相同模型、相同上下文和同一套测试，它们很可能一起忽略同一个问题。

## 7. “验证 Agent”为什么不等于验证

验证强度可按证据与被验证对象的独立程度排序：

1. **确定性 oracle**：schema、类型检查、hash、计数、编译器、数据库约束。
2. **真实环境测试**：浏览器、设备、API、部署后的 URL 和外部状态。
3. **性质与差分测试**：不只检查样例答案，而是检查 invariant 或与独立实现对比。
4. **人工 checkpoint**：高风险、价值判断和需求歧义处由人确认。
5. **LLM critic / judge**：适合开放文本、视觉观感和难形式化问题，但应保留输入、评分依据与不确定性。

例如 Agent 声称“博客已经推送”：

```text
弱证据：git push 命令看起来没有报错
较强证据：本地 HEAD == origin/main
更强证据：线上 URL 返回 200 且包含新标题
完整证据：文章、索引、图片资源和移动交互均在线验证
```

Harness 的工作不是让 Agent 更自信，而是让结论绑定到对应强度的证据。

## 8. 评价一个 Harness 项目，应该看什么

功能越多不代表越适合。本文用十二个问题判断项目：

1. 它服务于交互结对、异步任务、批量 benchmark，还是团队自动化？
2. Agent loop 是否清晰可读、可替换、可中止？
3. 上下文如何选择、压缩和恢复？
4. 工具接口是 shell、专用 ACI、MCP，还是多种协议？
5. 默认在宿主机、容器、VM 还是远端执行？
6. 权限是提示、逐次确认、allowlist 还是 sandbox？
7. 是否保存 trajectory、diff、事件和成本？
8. 完成由模型声明，还是由外部验收器判定？
9. 失败后能否 retry、undo、resume、revert？
10. 是否支持多模型，以及切换模型会损失哪些能力？
11. 是否适合复现实验和批量评测？
12. 运维复杂度与任务价值是否匹配？

下面四个项目不是按“最好”排序，而是四种不同答案。仓库状态和 Star 仅是 **2026-08-14 的动态快照**，不作为质量结论。

![Project landscape](/images/blog/harness-project-landscape.svg "图 4：四个项目分布在从轻量人机结对到持续自动化、从专用代码工作流到通用扩展平台的不同位置。")

## 9. OpenHands：平台与控制面路线

**官方定位。** OpenHands 主仓库当前的产品名称是 Agent Canvas，定位为自托管的开发者控制中心：可以运行 OpenHands 自身，也可以连接 Claude Code、Codex、Gemini 或其他支持 Agent Client Protocol 的 Agent；后端可位于本地、Docker、VM、企业基础设施或云端，并可通过 schedule 或 webhook 运行自动化。[5]

截至快照时间，主仓库约 8.4 万 Star、MIT License，最新 release 为 `v1.13.0`。更值得关注的不是数字，而是它已从“一个会改代码的 Agent”演进为**多 Agent 后端的操作平面**：

- Agent Canvas 管会话、后端切换和自动化入口；
- Agent Server 提供运行 Agent 的服务接口；
- Software Agent SDK 提供 Agent、Conversation、Tool 等模块；
- 工作区可以直接使用本机，也可置于 Docker/Kubernetes 等临时环境；
- Automation Server 把对话式 Agent 接到 GitHub、Slack、Linear 等事件流。

### 9.1 它体现了什么 Harness 思想

OpenHands 把 Harness 的重点放在**执行位置、持久运行、团队入口和后端治理**。当任务来自 webhook、需要在笔记本关闭后继续，或多个 Agent 共享远端算力时，单个 CLI 循环已经不够，必须有控制面。

### 9.2 最适合谁

- 希望自托管 coding agents 的团队；
- 需要把 issue、review、依赖升级等任务做成持续自动化的工程组织；
- 需要本地、容器、VM 与云后端统一入口的用户；
- 想基于 SDK 构建自有开发者体验的研究者或平台团队。

### 9.3 主要边界

它的部署面与信任面也最大。官方 Quickstart 明确警告：不使用 sandbox 时，Agent Server 会直接访问宿主文件系统。[5] 因此“能连接很多 Agent”不等于“默认安全”。生产使用仍需要网络隔离、secret 管理、最小权限、镜像治理和自动化触发源验证。对于只想在一个仓库里结对改两处代码的人，完整平台可能过重。

## 10. mini-SWE-agent：极简、可读与可复现实验路线

原始 SWE-agent 是 NeurIPS 2024 工作，核心贡献是通过专门设计的 Agent-Computer Interface 让语言模型处理真实 GitHub issue。[2] 但截至 2026 年，SWE-agent 官方 README 已明确表示主要开发转向 **mini-SWE-agent**，并建议新用户优先使用后者。[6]

截至快照时间，mini-SWE-agent 约 6.5 千 Star、MIT License，最新 release 为 `v2.4.6`。它最有价值的特点不是“功能最多”，而是默认 Agent 类只有约百行 Python，核心循环可以直接读懂：

```text
render task and system templates
-> model.query(messages)
-> parse actions
-> environment.execute(action)
-> append observations
-> repeat until exit or limit
```

代码中显式实现了 step limit、cost limit、wall-time limit、连续格式错误上限、trajectory 保存和环境 timeout。也就是说，它把 Harness 的关键机制暴露为几个小而可测的组件，而不是藏在庞大框架里。[7]

### 10.1 它体现了什么 Harness 思想

mini-SWE-agent 证明：Harness 的价值不等于框架体积。一个很小的循环，只要任务模板、动作格式、环境反馈、停止条件和轨迹记录设计得好，就可以成为强研究基线。

### 10.2 最适合谁

- 想研究 prompt、action format、环境接口和模型差异的研究者；
- 需要在 SWE-bench、ProgramBench 等任务上批量复现实验的团队；
- 想从代码层真正理解 Agent loop 的开发者；
- 需要快速构造一个可控专用 Agent，而不是先部署平台的人。

### 10.3 主要边界

极简也意味着很多生产能力不在核心循环里。默认 `LocalEnvironment` 会直接通过 shell 在给定工作目录执行命令；隔离、企业权限、人工审批、复杂记忆与团队控制面需要由使用者补充。它适合作为可读的“实验底盘”，不应因 benchmark 表现就直接等同于生产安全系统。

## 11. Aider：Git 原生的人机结对路线

Aider 把自己定义为终端中的 AI pair programming。它不以长时间无人值守为第一目标，而是把**代码上下文、编辑和 Git 恢复**做成稳定的交互工作流。[8]

截至快照时间，Aider 约 4.8 万 Star、Apache-2.0 License；GitHub 最新正式 release 为 `v0.86.0`。其几个代表性 Harness 设计是：

- 用 tree-sitter 构建 repository map，在 token 预算内向模型提供全仓库关键符号；
- 编辑后自动形成描述性 Git commit，支持 `/undo` 和历史审查；
- 对已有 dirty files 先做隔离提交，减少覆盖用户工作的风险；
- 可在每次变更后自动运行 lint，并支持 `/test` 或配置测试命令；
- 提供 ask、code、architect 等模式，把讨论、规划与直接编辑区分开。

### 11.1 它体现了什么 Harness 思想

Aider 的强项是利用开发者已经信任的 Git 作为**状态、审计和恢复机制**。它没有发明一套新的版本系统，而是把 Agent 改动映射到现有工程原语。Repo map 则体现了另一条原则：上下文工程不是把仓库全塞进窗口，而是建立可预算的结构摘要。

### 11.2 最适合谁

- 希望保持人类在回路中、逐步审查改动的开发者；
- 已经使用 Git，希望每次模型修改都可见、可撤销的个人或小团队；
- 需要快速切换讨论、架构与代码编辑模式的工作流；
- 不需要额外服务端控制面的本地开发场景。

### 11.3 主要边界

Aider 的默认范式仍偏结对交互，而非跨系统长任务编排。自动 lint/test 提供了有价值的传感器，但测试是否覆盖真正需求仍由项目决定。另一个值得注意的细节是，Aider 默认自动 commit 可跳过 pre-commit hooks，只有显式启用 `--git-commit-verify` 才会运行它们；因此团队不能把“有 commit”误认为“已通过全部门禁”。[9]

## 12. Goose：MCP、Recipes 与本地通用 Agent 路线

Goose 最初由 Block 发起，目前仓库位于 Agentic AI Foundation 组织下。它是一个开源、可扩展的本地 Agent，支持多种模型提供商，并通过 MCP 连接扩展。官方 README 当前列出 15+ provider 与 70+ extension。[10]

截至快照时间，Goose 约 5.3 万 Star、Apache-2.0 License，最新 release 为 `v1.46.0`。与前三个项目相比，它更像一个通用的本地 Harness：

- MCP Extensions 连接开发工具、数据源和应用；
- Recipes 把 extensions、prompt 和 settings 打包成可复用工作流；
- Sessions 保存连续交互与上下文，可恢复和检索；
- 内置 Analyze、Skills、Todo、Chat Recall、Top of Mind 等平台扩展；
- 提供 permission mode、tool permissions、extension allowlist；
- 安全文档还包括 prompt injection detection 与独立监视工具调用的 Adversary Mode。[11]

### 12.1 它体现了什么 Harness 思想

Goose 把 Harness 的重心放在**扩展生态与可重复工作流**：MCP 负责能力接入，Recipe 负责把一组能力和目标打包，Session 与上下文机制负责延续任务，权限与安全层负责约束执行。

### 12.2 最适合谁

- 希望在本地使用多模型、避免绑定单一提供商的用户；
- 已有多个 MCP Server，希望统一编排的团队；
- 需要把成功会话固化为可共享 Recipe 的组织；
- 任务不只改代码，还要连接文档、数据库和业务系统的通用 Agent 场景。

### 12.3 主要边界

扩展越多，供应链、权限和 prompt injection 风险越大。Goose 文档明确说明默认可自主执行命令和修改文件，想获得更强控制需要配置 permission mode 与 tool permissions。[12] Adversary Mode 是有价值的防线，但“独立 Agent 监视”仍不是形式化安全边界；真正高风险动作仍应由确定性策略和人工审批控制。

## 13. 四个项目放在同一张表里

| 维度 | OpenHands | mini-SWE-agent | Aider | Goose |
| --- | --- | --- | --- | --- |
| 核心路线 | 多后端控制面与自动化 | 极简研究循环 | Git 原生结对编程 | MCP 可扩展本地 Agent |
| 主要交互 | Web/服务/自动化 | CLI/批处理 | CLI 人机协作 | Desktop/CLI/Recipe |
| 执行环境 | 本地、Docker、VM、云 | 可替换环境，核心含本地 shell | 本地仓库 | 本地为主，工具由扩展提供 |
| 上下文优势 | Conversation/Server 状态 | 模板与完整 trajectory | repo map 与显式文件集 | session、context、skills、recall |
| 恢复机制 | 持久会话与后端运行 | trajectory、limit、可重跑 | Git commit、`/undo` | session resume、todo、recipe |
| 验证重点 | 取决于 Agent/Automation 配置 | 环境 observation 与 benchmark | lint、test、Git diff | 工具反馈与可配置工作流 |
| 权限重点 | sandbox/部署隔离必须配置 | 使用者负责环境隔离 | 本地 Git 边界，命令仍需谨慎 | permission、tool policy、allowlist |
| 最佳场景 | 团队平台、异步自动化 | 研究、评测、专用循环 | 日常结对开发 | 通用本地 Agent 与 MCP 编排 |
| 主要代价 | 部署与治理复杂 | 生产能力需自行补齐 | 长任务编排较弱 | 扩展面带来信任与安全成本 |

这张表也说明“哪个最好”是一个错误问题。更合理的问题是：**你的主要不确定性在哪里？**

![Selection matrix](/images/blog/harness-selection-matrix.svg "图 5：项目选择应由自治时长、任务范围、治理需求和研究可解释性共同决定，而不是由单一热度指标决定。")

## 14. 怎么选：四种直接决策

### 14.1 我只想在本地与 Agent 一起改代码

先选 **Aider**。它对 Git、上下文和逐步审查的默认设计最贴近日常工程。你仍需配置真实测试命令和 pre-commit 验证。

### 14.2 我要研究 Harness 本身或做 benchmark

先选 **mini-SWE-agent**。循环短、状态明确、trajectory 可保存，更适合做消融。不要先用大平台遮住你真正研究的变量。

### 14.3 我要让多个 Agent 持续处理团队任务

评估 **OpenHands**。这里的核心需求已不是一个更好的 CLI，而是后端、会话、触发器、运行位置和团队治理。

### 14.4 我要连接很多 MCP，并把流程复用给团队

评估 **Goose**。但先建立 extension allowlist、权限模式和 secret 边界，再增加扩展数量。

## 15. 从零搭自己的 Harness：不要一上来造平台

可以按风险逐层建设。

### 第 0 层：先定义完成

把“做一个登录页”改成机器可检查的 contract：URL、关键交互、错误状态、视口、测试命令和交付物清单。没有完成定义，后面所有自动化都只是在加速不确定性。

### 第 1 层：建立真实反馈

至少让 Agent 看到命令退出码、测试报告、Git diff 和真实运行结果。Web 任务必须用浏览器；部署任务必须检查远端和线上资源。

### 第 2 层：加入恢复点

使用小步提交、任务状态文件、幂等 action ID 和可恢复 checkpoint。失败时优先回到已知正确状态，而不是让模型在污染环境上继续补丁。

### 第 3 层：把高风险动作移出模型自觉

读操作、可撤销写操作和不可逆提交应有不同权限。删除、支付、发布和数据库迁移设置显式审批；能用 schema、ACL 和 transaction 保证的，不依赖自然语言提醒。

### 第 4 层：再增加并行 Agent 与自动化

只有当单循环的瓶颈已经被测量，才引入 subagent、队列、schedule、webhook 和团队控制面。否则你只是把不可观测的错误并行化。

## 16. Harness 也需要测试

Harness 自己可能制造失败：

- 旧规则与新工具冲突；
- context loader 把过时文档排在最新事实前；
- verifier 只检查容易通过的路径；
- permission hook 误拦截合法动作；
- retry 把非幂等调用执行两次；
- memory 固化了已经被用户纠正的假设；
- observability 保存了不该记录的 secret。

所以 Harness 应有自己的评测集，至少包括：

```text
任务成功率
虚假完成率
未经授权动作率
错误恢复成功率
人工介入次数
每个成功任务的成本和时延
规则误拦截率
跨会话状态一致性
```

不要只跑“模型能完成多少任务”，还要做 Harness 消融：关闭 repo map、移除浏览器验证、改变权限策略、替换模型，观察哪个机制真正贡献可靠性。

## 17. 什么时候应该删除 Harness

从真实失败长出的规则并不意味着永远保留。一个健康 Harness 也要持续做减法：

- 模型或工具已经稳定消除该失败；
- 两条规则可以被一个确定性 invariant 替代；
- 规则增加的 token、时延或误拦截超过收益；
- 项目结构变化后旧经验已失效；
- 某个 verifier 与 actor 高度相关，实际没有增加证据强度。

最好的 Harness 不是最长的 `AGENTS.md`，而是**最小但覆盖主要风险的可执行 contract**。

## 18. 最终判断

Harness Engineering 的价值不在于发明一个流行名词，而在于把 Agent 工程从“提示词抽卡”推进到可检验系统设计。它迫使我们回答：

- 目标如何被外部检查？
- 模型能看见哪些真实环境反馈？
- 哪些动作必须由系统强制限制？
- 新上下文如何恢复任务，而不是猜测历史？
- 完成结论绑定了什么证据？
- 失败后怎样回到已知正确状态？
- 哪些经验值得固化，哪些规则已经过时？

如果只记一句话，我会选择：

> **模型负责提出下一步，Harness 负责让每一步有边界、有反馈、有证据，并且失败后还回得来。**

OpenHands、mini-SWE-agent、Aider 和 Goose 分别从平台、研究、结对与扩展生态回答了这个问题。选择其中任何一个之前，先写下自己的任务 contract、风险边界和验收证据。否则装再多 Skills、MCP 或 Agent，也只是在给一个没有闭环的系统增加更多动作。

## 参考资料

1. Mitchell Hashimoto, [My AI Adoption Journey](https://mitchellh.com/writing/my-ai-adoption-journey), 2026-02-05.
2. John Yang et al., [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793), NeurIPS 2024.
3. Anthropic, [Building Effective AI Agents](https://www.anthropic.com/research/building-effective-agents), 2024-12-19. 该页面已注明工具生态自 2024 年以来发生变化，本文只采用其通用设计原则。
4. Anthropic, [Effective Harnesses for Long-Running Agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents), 2025-11-26.
5. OpenHands, [Agent Canvas Repository and Architecture](https://github.com/OpenHands/OpenHands), accessed 2026-08-14.
6. SWE-agent, [SWE-agent Repository](https://github.com/SWE-agent/SWE-agent), accessed 2026-08-14. README 当前建议优先使用 mini-SWE-agent。
7. SWE-agent, [mini-SWE-agent Repository](https://github.com/SWE-agent/mini-swe-agent), accessed 2026-08-14.
8. Aider, [AI Pair Programming in Your Terminal](https://github.com/Aider-AI/aider), accessed 2026-08-14.
9. Aider, [Git Integration](https://aider.chat/docs/git.html), [Repository Map](https://aider.chat/docs/repomap.html), [Linting and Testing](https://aider.chat/docs/usage/lint-test.html), accessed 2026-08-14.
10. Goose, [Goose Repository](https://github.com/aaif-goose/goose), accessed 2026-08-14.
11. Goose, [Recipes](https://goose-docs.ai/docs/guides/recipes/), [Using Extensions](https://goose-docs.ai/docs/getting-started/using-extensions/), [Managing Sessions](https://goose-docs.ai/docs/guides/sessions/), accessed 2026-08-14.
12. Goose, [Security](https://goose-docs.ai/docs/guides/security/), accessed 2026-08-14.

> **证据说明。** 文中的项目 Star、release 与仓库活跃状态来自 GitHub API 在 2026-08-14 的快照，会随时间变化；产品能力以官方 README 和文档为依据；项目适用场景与局限属于本文的工程分析。附件中“6.7% 到 68.3%”“Terminal-Bench 2.0 从 30 名外到前 5”等说法未找到足以复核实验条件的原始来源，因此没有作为事实引用。

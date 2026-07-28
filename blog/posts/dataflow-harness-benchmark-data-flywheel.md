# DataFlow-Harness：从可编辑数据流水线到 Benchmark-Grounded 数据飞轮

> **论文**：[DataFlow-Harness: A Grounded Code-Agent Platform for Constructing Editable LLM Data Pipelines](https://arxiv.org/abs/2607.16617)，arXiv:2607.16617v2，2026 年 7 月。作者来自北京大学、上海算法创新研究院与北京中关村学院。本文不只介绍论文，还回答一个更具体的问题：能否用 DataFlow-Harness 结合数据飞轮，参照 benchmark 的标准协议自动生成、筛选和回流训练数据？

我的结论是：**可以，而且 DataFlow-Harness 很适合作为飞轮的“可编辑控制平面”；但它本身还不是完整的数据飞轮，也不能仅靠 DAG 校验和 LLM-as-a-Judge 保证数据质量。** 真正可用的方案，需要在 DataFlow-Harness 之上增加机器可读的 Benchmark Contract、数据级 provenance、确定性验证器、污染隔离、难度与覆盖控制、shadow training 和 promotion gate。

一句话概括这项工作的价值：

> Code Agent 过去擅长把自然语言变成一次性脚本；DataFlow-Harness 把它变成对平台原生 DAG 的增量、受约束、可回滚编辑。这个变化使数据工程从“Agent 写过一段代码”升级为“团队拥有一个可查看、复用、审计和持续优化的流水线资产”。

![DataFlow-Harness overview](/images/blog/dataflow-harness-overview.svg "图 1：DataFlow-Harness 通过 Skills、MCP、typed mutation 和 WebUI，把自然语言需求落成平台原生、持久化的可编辑 DAG。")

## 1. 论文解决的不是 NL2Code，而是 NL2Pipeline gap

很多 Code Agent 已经能写出数据清洗脚本：读 JSONL、调用模型、过滤低分样本、再导出数据。这在一次实验中可能足够，但在持续造数和团队协作中会暴露四个问题。

1. **脚本与平台状态分离。** Agent 不知道平台当前有哪些 operator、字段 schema、模型 endpoint 和历史 pipeline，只能依赖 prompt 或仓库搜索。
2. **结果是 disposable script。** 代码能运行，不等于能在可视化平台中继续编辑、复用、治理和比较版本。
3. **自由代码容易幻觉。** Agent 可能调用不存在的算子、旧接口或错误字段，并把所有逻辑揉成一个难审计的程序。
4. **人工修改无法实时回流。** 用户在 GUI 中调整了参数或连线，下一轮 Agent 仍可能基于旧状态继续修改。

论文把这种自然语言意图与平台原生 pipeline artifact 之间的断裂称为 **NL2Pipeline gap**。其目标不是证明 DAG 比 Python 更有表达力，而是让 Agent 生成的对象从一段文本代码变成平台能够长期管理的工作流。

## 2. DataFlow-Harness 的四个组成部分

### 2.1 Data Pipeline Backend：唯一状态真源

论文将 pipeline 表示为：

```text
P = (D, O, E, S, R)

D: 数据源及 URI
O: 已配置的 operator 实例
E: operator 间的数据依赖边
S: 输入和输出字段 schema
R: 模型服务 endpoint 等运行时状态
```

Backend 是 conversational interface、WebUI 和程序化访问的 authoritative source of truth。Agent 不直接重写一个完整 pipeline 文件，而是发出 typed mutation：增加或删除 operator、更新参数、连接边、变更数据源。每次 mutation 都只修改显式状态。

这种增量编辑有两个工程优势。第一，失败的变更可以被拒绝，而不是等整段脚本生成后才发现。第二，用户与 Agent 共用同一个 artifact，GUI 的人工修改会立即成为下一轮 Agent 的上下文。

### 2.2 MCP Tools Layer：把 Agent ground 到实时平台

每次 pipeline 变更遵循 **Request-Validate-Commit**：

```text
读取最新 pipeline + operator registry
→ Agent 提交 typed mutation
→ 检查 DAG 与字段 schema
→ 通过后写入 backend
→ WebSocket 广播给 WebUI
```

MCP 在这里不是“再接一个工具”这么简单。它向 Agent 暴露的是**当前真实可用的 operator 集合和实时 workflow state**，从而缩小 Agent 的动作空间。相比把整个代码仓库塞进上下文，registry metadata 更紧凑，也更容易约束。

### 2.3 DataFlow-Skills：补足 operator 文档没有写出的程序知识

只知道“有哪些 operator”仍然不够。比如从教材 PDF 提取 VQA，需要依次考虑文档解析、layout recovery、OCR、图表提取、题目与答案跨页匹配、质量过滤。每个算子的签名都正确，也可能组成语义错误的流程。

DataFlow-Skills 因此编码两类知识：

- **procedural blueprint**：schema 推断、operator 选择、参数配置、模型服务验证和推荐组装顺序；
- **compositional constraint**：模态匹配、nested field 流转、上下游字段兼容和前置步骤。

这也是论文消融中最明确的结果：在 QA basic、QA with filter 和 Text-to-QA chain 三类依赖隐式程序知识的任务上，MCP-only 的总成功次数为 `18/30`，加入 Skills 后升至 `29/30`；而字段重命名、nested flatten、长度过滤等简单任务，两者都是满分。

### 2.4 DataFlow-WebUI：对话和画布同步编辑

WebUI 同时提供自然语言对话和 DAG 画布。用户可以观察 Agent 添加了哪些 operator、直接改参数和连线，再让 Agent 基于新状态继续。这一点对数据飞轮尤其重要：评测协议和过滤阈值不应该永久埋在脚本里，而应成为可以审阅和版本化的节点。

## 3. Typed mutation 验证了什么，没有验证什么

DataFlow-Harness 在 commit 前检查两项性质：

- 更新后的图仍是 DAG；
- 相邻 operator 的输出字段 schema 与输入字段 schema 兼容。

这能排除循环依赖、缺字段和明显的类型错误，但论文明确承认：**structural validity 不等于 semantic correctness，也不保证 endpoint 可用或输出质量。**

例如下面的 pipeline 在结构上完全合法：

```text
Benchmark Test Set
→ LLM 改写题目
→ LLM 生成答案
→ LLM Judge 高分过滤
→ 加入训练集
```

但它在研究上可能是灾难：测试集被变体改写后泄漏进训练数据，同一家族模型同时生成和评分造成 correlated bias，最终 benchmark 上升可能只是 contamination，而不是泛化提升。

因此，用 DataFlow-Harness 做飞轮时，必须把 validation 从“图是否能运行”扩展到至少四层：

1. **结构验证**：DAG、schema、必填字段、资源配置；
2. **数据验证**：格式、执行结果、答案一致性、重复、PII、license；
3. **协议验证**：训练源是否越过 benchmark 隔离边界，metric 与后处理是否固定；
4. **效果验证**：相同训练预算下是否提升目标能力，同时不损害保留集。

## 4. 论文实验应该怎样读

### 4.1 Pipeline 构建可靠性与成本

论文使用 Claude Opus 4.7，在 12 个任务上每种方法独立运行 10 次，共 120 次。任务覆盖 QA generation、review governance、long-document processing、multi-field scoring、schema normalization 和 low-quality filtering。

| 方法 | Artifact | E2E Pass | 平均成本 | 生成延迟 |
| --- | --- | ---: | ---: | ---: |
| Vanilla Claude Code | Disposable Script | 91.7% | $0.950 | 190.7s |
| Context-Aware Claude Code | Disposable Script | 94.2% | $0.456 | 115.9s |
| MCP-only | Native DAG | 83.3% | $0.321 | 105.5s |
| DataFlow-Harness | Native DAG | 93.3% | $0.261 | 95.5s |

相对 Vanilla CC，Harness 的测量成本降低 `72.5%`，生成延迟降低 `49.9%`；相对 Context-Aware CC，成功率只低 `0.9` 个百分点，但成本低 `42.8%`。不过论文没有做预注册的 non-inferiority test，因此“数值接近”不能写成统计等价。

更重要的是，四种方法的动作空间不同：自由脚本可以生成任意 Python，Harness 只能组合 registry 中已有 operator。这个比较反映的是**完整系统的工程 trade-off**，不是纯模型能力排行榜。

### 4.2 Textbook-to-VQA

教材到 VQA 的 case study 中，Harness 达到 `0.972` precision 和 `0.873` coverage，Context-Aware CC 为 `0.893/0.801`，MCP-only 为 `0.784/0.621`。结果支持“成熟 operator + 程序知识”优于每次从零实现，但论文也指出需要重复实验和更完整的 annotation protocol 才能确认泛化性。

### 4.3 数据质量的下游训练证据

论文真正与数据飞轮相关的是两个 controlled case study。

**数学 pipeline。** 同一 prompt 要求 Agent 验证并过滤 seed problem、丢弃 ill-posed item、每道题扩展两个新问题、生成 reasoning trace、再做 n-gram dedup。用生成数据微调 Qwen2.5-32B-Instruct，在相同训练预算下，Harness pipeline 的八项平均分一轮为 `51.6`，Vanilla CC 为 `49.9`；两轮为 `55.7 vs. 54.5`。一轮训练时 AIME24@32 为 `35.9 vs. 25.1`，AIME25@32 为 `34.5 vs. 21.6`。

**General SFT pipeline。** 从零生成 10K instruction-response pair，流程为 topic-conditioned generation、critique-then-rewrite、LLM judge 过滤。相同配置微调 Qwen2.5-7B-Base 后，Harness pipeline 在九项 benchmark 上平均 `63.8`，Vanilla CC 为 `61.5`；差异主要来自 code，MBPP 为 `75.4 vs. 64.6`。

这些结果说明 grounded pipeline **有可能**产生更有用的数据，却不能视为普遍因果结论。每个场景只有一条 Agent 构建的 pipeline，没有多个独立 pipeline seed 和训练 seed；数学表中 Harness 并非每个 benchmark 都更强；最敏感的 AIME 提升还需要独立污染审计。论文自己也明确列出了这些限制。

![Evidence boundaries](/images/blog/dataflow-harness-evidence.svg "图 2：93.3% 证明的是 pipeline 构建与执行通过率；数据质量需要独立的数据审计和下游训练证据。")

## 5. 能否结合数据飞轮造数据？可以，但要重构目标函数

一个简单“生成更多高分样本”的循环不是数据飞轮，而是自我复制。真正的数据飞轮应由模型在固定 evaluation contract 下暴露能力缺口，再针对缺口采集或合成数据，通过多道 gate 后训练候选模型，最后只有在独立保留集上获得可信增益的数据策略才能进入下一版本。

我建议把系统拆成三个平面。

### 5.1 Control Plane：DataFlow-Harness

负责 pipeline authoring、operator discovery、typed mutation、版本比较、人工编辑、审批和回滚。它回答“流程是什么、谁改了什么、现在执行哪个版本”。

### 5.2 Execution Plane：DataFlow + RayOrch / serving

负责批量生成、模型调用、deterministic verifier、dedup、embedding、sandbox execution、统计聚合和 artifact 存储。它回答“数据如何被计算出来”。

### 5.3 Optimization Plane：LoopAI 或独立 supervisor

OpenDCAI 的 [Dataflow-LoopAI](https://github.com/OpenDCAI/Dataflow-LoopAI) 已经把 Judger、Analyzer、Obtainer、Constructor 和 Trainer 组织成闭环，覆盖评测、错误分析、数据获取和训练。它可以负责“为什么启动下一轮、针对哪个 slice、何时停止”。

Harness 和 LoopAI 不是替代关系：前者适合成为可治理的 pipeline control plane，后者适合成为跨 evaluation、data、training 的 flywheel supervisor。

## 6. 核心新增层：Benchmark Contract

用户提出“参考 benchmark 的标准协议来构建筛选过滤”，关键不是把 benchmark 名字写进 prompt，而是把 benchmark 变成不可随意漂移的机器可读 contract。

```yaml
benchmark_id: math_reasoning_v3
version: 2026-07-28
task_schema:
  input: {problem: string}
  output: {answer: string, reasoning: string}
splits:
  train_seed: allowed
  dev: diagnostics_only
  test: quarantined
generation:
  temperature: 0
  max_tokens: 8192
normalization:
  answer: math_equivalence_v2
metrics:
  - exact_or_symbolic_match
  - pass_at_k
contamination:
  protected_corpora: [dev, test, canary]
  exact_hash: reject
  normalized_ngram: {n: 13, threshold: 0.8}
  semantic_review_band: [0.82, 0.94]
promotion:
  target_delta_min: 1.0
  retain_delta_min: -0.2
  seeds: [17, 42, 73]
```

一份完整 contract 至少固定以下内容。

- **task schema**：输入、输出、必填字段、允许工具、答案类型；
- **split policy**：哪些数据可用于生成，哪些只能诊断，哪些必须 quarantine；
- **prompt/protocol**：system prompt、few-shot、stop sequence、sampling、max tokens；
- **normalizer**：大小写、单位、数学等价、代码入口、SQL canonicalization；
- **metric**：exact match、F1、pass@k、execution success、judge rubric；
- **test environment**：依赖版本、sandbox、timeout、随机种子；
- **contamination policy**：protected corpus、hash、n-gram、embedding 阈值与人工复核带；
- **promotion rule**：目标集增益、retain set 退化上限、方差、成本和安全 gate。

可以参考四类现有协议设计：

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 的 task config、request construction、filter 和 metric 定义；
- [HELM](https://github.com/stanford-crfm/helm) 的 scenario、适配、metric 与透明报告；
- [EvalPlus](https://github.com/evalplus/evalplus) 的代码执行与扩展测试，而不是只让 LLM 判断代码；
- [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) 的时间切分与污染控制思想。

Benchmark Contract 应是只读版本化 artifact，Agent 可以选择它、引用它、生成配套 pipeline，但不能在优化过程中悄悄修改 metric 或 test split。

## 7. 一个可落地的飞轮 DAG

![Benchmark-grounded flywheel](/images/blog/dataflow-harness-flywheel.svg "图 3：Benchmark-grounded 数据飞轮从失败切片开始，经候选生成、多道过滤、shadow training 和 promotion gate 回流。")

### Stage A：固定评测并生成 failure ledger

先运行当前模型，逐条保存：benchmark version、sample id、model snapshot、raw output、normalized answer、verifier result、latency、token、错误类型与 uncertainty。不要只保存总分。

Analyzer 将失败归类为可操作 slice，例如：

- 数学中的约束遗漏、代数化简、几何构造；
- 代码中的边界条件、状态污染、复杂度超限；
- Agent 中的错误 tool selection、参数遗漏、恢复失败；
- Omni 中的说话人判断、时序 grounding、社会意图错误。

failure ledger 只记录缺口，不直接把测试题送进生成器。

### Stage B：把 failure slice 转成 generation specification

从错误类型抽取能力标签、约束模式和难度，而不是复制原题文本。生成 specification 应包含：目标 skill、允许知识源、禁止相似字段、答案验证器、难度范围、数量预算和多样性 quota。

例如 AIME 错题只能提供抽象能力标签 “modular arithmetic + hidden divisibility constraint”，不能把原题或详细解法交给 generator。否则所谓 targeted synthesis 很容易变成 benchmark paraphrase。

### Stage C：多源候选生成

候选数据最好来自独立来源组合：真实许可数据、程序化模板、symbolic generator、retrieval-grounded source、多个模型家族生成和人工 seed。每条记录必须附 provenance：

```json
{
  "sample_id": "...",
  "source_uri": "...",
  "source_license": "...",
  "generator_model": "...",
  "generator_prompt_hash": "...",
  "pipeline_version": "...",
  "parent_ids": ["..."],
  "created_at": "...",
  "target_slice": "constraint_reasoning",
  "benchmark_contract": "math_reasoning_v3"
}
```

没有 lineage 的样本，不应该进入正式训练集。

### Stage D：八道筛选 gate

以下 gate 应拆成独立 operator，输出原因码，不要做成一个黑盒总分。

**Gate 1：Schema 与格式。** JSON schema、字段完整、长度、语言、编码、答案可解析。失败直接拒绝。

**Gate 2：确定性正确性。** 数学用 symbolic equivalence、数值回代和约束检查；代码用 sandbox unit test、EvalPlus 风格扩展测试、静态和复杂度检查；SQL 用隔离数据库执行与 result equivalence。能执行验证时，不应让 LLM judge 取代执行器。

**Gate 3：内部一致性。** 问题、reasoning、final answer、引用证据相互一致；多次采样或独立 solver 的答案分歧进入复核带。

**Gate 4：质量与可学性。** rubric 分开评估 clarity、self-containedness、reasoning validity、pedagogical value 和 verbosity。至少使用与 generator 不同的 judge family，并对 judge 做人工校准。

**Gate 5：重复与近重复。** exact hash → normalized text hash → MinHash/LSH → embedding cluster 分层执行。重复不是只浪费 token，还会改变训练权重并放大某类模式。

**Gate 6：Benchmark contamination。** 对 quarantine corpus 运行 exact、normalized n-gram、结构 fingerprint 和 semantic retrieval。高相似直接拒绝，中间带人工复核，低相似才通过。污染扫描必须早于训练。

**Gate 7：难度与覆盖。** 依据 solver pass rate、推理步数、执行复杂度、discriminator margin 和能力标签分桶。按 quota 采样，而不是简单取 judge 最高分；否则数据会集中在“写得漂亮但容易”的区域。

**Gate 8：安全、隐私与许可。** PII、敏感内容、版权、来源许可、域策略和高风险行为。过滤结果和原因需要保留审计日志。

![Multi-gate filtering](/images/blog/dataflow-harness-filter-gates.svg "图 4：候选样本必须通过可解释的多道 gate；单个 LLM 总分不能同时承担正确性、污染、难度和安全判断。")

### Stage E：Dataset manifest 与 immutable snapshot

输出不应只是 `train.jsonl`，而应包含：dataset version、pipeline commit、operator version、contract version、source distribution、filter attrition、difficulty histogram、dedup cluster、contamination report、sampling weights 和 rejected-reason statistics。

训练使用 immutable snapshot，后续 pipeline 变化不能修改已经报告过的 dataset。这样才能回答“哪一版数据导致哪一版模型变化”。

### Stage F：Shadow training 和 promotion

在固定 base checkpoint、训练预算和多个 seed 下做 candidate-vs-control。至少同时评估：

- target slice：是否修复预期错误；
- benchmark holdout：是否泛化，而不是只拟合诊断模板；
- retain suite：通用能力是否退化；
- adversarial/canary：是否发生污染或策略绕过；
- calibration 与 safety：模型是否更自信地犯错；
- 成本：每单位增益需要多少生成、过滤和训练成本。

只有满足 promotion contract 的数据版本才进入主训练池。失败版本不删除，保留 pipeline、数据 manifest 和结果，避免未来重复踩坑。

## 8. 筛选不是“分数大于 0.8”

数据飞轮最常见的设计错误，是让一个 LLM 输出 `quality_score`，然后设阈值。这会混淆四种不同问题。

1. **Validity**：样本格式是否完整、程序能否运行；
2. **Correctness**：答案是否正确、推理是否忠实；
3. **Utility**：样本对目标模型是否有学习价值；
4. **Risk**：是否泄漏 benchmark、侵犯隐私或违反许可。

一个样本可以“正确但无用”，例如模型已经 100% 掌握的简单题；也可以“高质量但有污染”，例如对测试题做了精美改写。总分无法表达 hard gate 和 soft ranking 的区别。

推荐的 selection 形式是：

```text
eligible(x) = schema_ok
           ∧ deterministic_verify
           ∧ contamination_risk < hard_limit
           ∧ privacy_license_ok

rank(x) = learning_value
        + coverage_gain
        + diversity_gain
        - redundancy
        - uncertainty_penalty
        - generation_cost
```

先用 hard gate 决定能否进入候选池，再用多目标 ranking 和 quota 选择训练集。阈值、权重和 quota 都应进入 pipeline version，而不是散落在 prompt 中。

## 9. 如何估计 learning value

真正的数据飞轮不是寻找“最像标准答案”的数据，而是寻找对当前模型最有增益的数据。可以组合以下信号。

- **当前模型失败、强 verifier 通过**：优先级高；
- **多个同级模型分歧**：可能处于能力边界；
- **训练前 loss 高但非噪声**：有潜在学习价值；
- **influence / gradient similarity**：与目标 failure slice 的梯度方向相关；
- **小规模 probe training gain**：在廉价代理模型上验证收益；
- **覆盖增益**：填补能力 taxonomy 的低覆盖桶；
- **重复惩罚**：与已选样本相似时边际价值降低。

K-center greedy 之类的 diversity operator 可以作为一部分，但不能单独决定质量。最佳流程通常是 hard gate 后，在每个 difficulty × skill bucket 内做多样性选择。

## 10. 如何避免 benchmark 驱动变成 benchmark overfitting

Benchmark-grounded 不等于 benchmark-derived。前者用 benchmark 定义能力和评测 contract，后者直接从题目派生训练数据。二者必须隔离。

建议建立三层仓库：

```text
public/dev diagnostics
    可用于错误分类，但默认不进入生成 prompt

private holdout
    只允许 evaluator 读取，generator 和 trainer 无权限

canary / temporal set
    定期更新，用于发现策略对固定 benchmark 的过拟合
```

训练数据构造服务只能得到聚合 failure taxonomy，不能访问 private item text。Contamination operator 在受控环境中读取 protected set，只返回相似度和原因码，不返回命中的 benchmark 内容。

对于代码，可借鉴 LiveCodeBench 的时间切分；对于可执行任务，用 hidden tests；对于数学，用新的程序化实例和 theorem constraints；对于 Agent，用私有数据库状态和任务变体。每轮飞轮还应保留从未参与错误分析的 outer holdout。

## 11. DataFlow 中需要新增哪些 operator

现有 DataFlow 已有 generate、filter、refine、eval operator，以及 benchmark evaluator、K-center greedy、PromptedFilter、GeneralFilter 等组件。要实现严谨飞轮，建议新增或标准化以下 operator contract。

| Operator | 输入 | 输出 | 关键要求 |
| --- | --- | --- | --- |
| BenchmarkContractLoader | contract URI | immutable spec | hash + version pin |
| FailureSliceMiner | eval ledger | taxonomy + counts | 不泄露 private item text |
| ProvenanceAttacher | sample + runtime | lineage record | parent/prompt/model/pipeline hash |
| DeterministicVerifier | task sample | pass + evidence | sandbox/symbolic/database |
| DecontaminationScanner | candidate + protected index | risk + reason | exact/ngram/semantic 分层 |
| JudgeEnsemble | candidate + rubric | vector scores | generator-judge family 分离 |
| DifficultyCalibrator | sample + solver pool | difficulty bucket | 固定 solver snapshot |
| CoverageBalancer | candidate pool + taxonomy | weighted subset | quota + diversity |
| DatasetManifestWriter | accepted/rejected logs | manifest | attrition 与版本信息 |
| PromotionGate | train/eval reports | promote/reject | 多 seed、retain 与安全阈值 |

这些 operator 的输出应是结构化 evidence，而不是只输出 `keep=true`。WebUI 中点击样本或节点时，应能看到“在哪一关被拒绝、使用了哪个规则、匹配了哪个受保护 corpus 的 hash、由谁批准”。

## 12. 如何把 Skills 写成研究协议，而不是经验提示词

DataFlow-Skills 在论文中主要注入程序知识。用于飞轮时，Skill 应进一步包含：

- 前置条件：输入 schema、允许数据源、所需 verifier；
- 流程模板：operator 顺序和可替换节点；
- hard invariant：test quarantine、执行验证优先、provenance 必填；
- failure recovery：endpoint 失败、judge 分歧、schema 漂移如何处理；
- acceptance criteria：每一步的最小覆盖率、最大拒绝率和审计要求；
- forbidden shortcut：禁止用 test item 生成、禁止同模型自评即通过；
- reporting：必须输出的 manifest、统计和 diff。

Skill 不是越详细越好。论文消融中 `Review governance` 任务 MCP-only 为 `10/10`，Harness 为 `9/10`，说明过度 prescriptive guidance 可能压缩合法策略空间。正确做法是把不可违背的 invariant 固定，把可替换策略作为显式 choice，而不是把整条 pipeline 写死。

## 13. 一个最小可行版本

如果现在就开始做，我建议不要直接追求全自动训练闭环。先做一个窄领域 MVP。

**范围**：选择数学、代码或你自己的 Omni interaction benchmark 中一个可验证子任务。

**第一阶段：只做数据工厂。** 固定 benchmark contract，建立 generator、deterministic verifier、dedup、contamination、difficulty bucket、manifest 六类 operator。每轮输出候选池与审计报告，不自动训练。

**第二阶段：shadow training。** 固定一个小模型或 LoRA recipe，比较 baseline data、Harness data 和 ablation data；至少三个 seed，报告平均与方差。

**第三阶段：加入 failure-driven routing。** 评测错误只以 taxonomy 和 aggregate statistics 回流，private benchmark 内容与 generator 物理隔离。

**第四阶段：加入 promotion/rollback。** 让 LoopAI 或 supervisor 自动触发，但高风险 gate、数据发布和主模型更新保留人工审批。

MVP 的成功标准不是“生成了 100 万条”，而是：

- pipeline 可复现，重新执行得到相同 selection；
- 每条训练样本有 lineage；
- protected benchmark 无越权访问；
- 过滤 attrition 可解释；
- 相同训练预算下，多 seed 的 target gain 为正；
- retain suite、safety 和 calibration 不显著退化；
- 数据版本失败时可以完整回滚。

## 14. 研究上可以形成什么新工作

DataFlow-Harness 已经证明 Agent 可以低成本构建 editable pipeline。下一篇更有价值的工作，不应只是“增加更多 filter operator”，而可以提出 **Benchmark-Contract-Grounded Data Flywheel**：

1. 用机器可读 contract 绑定 generation、filter、evaluation 和 promotion；
2. 用权限隔离阻止 benchmark item 泄漏到 generator；
3. 用 typed evidence graph 记录每个样本从来源到训练的 lineage；
4. 用 deterministic verifier + judge ensemble + contamination scan 做多道 gate；
5. 用 failure slice 和 learning value 驱动 selection，而不是只追求静态质量分；
6. 用 shadow training、多 seed 与 retain suite 作为最终 pipeline evaluator；
7. 让 Agent 只在允许的 mutation space 内优化 pipeline，并支持人工编辑与 rollback。

可以设计如下研究问题：

- Contract grounding 是否降低 Agent 私自改变 metric、split 和阈值的比例？
- 多道 evidence gate 相比单 LLM judge，能否提升真实下游收益并降低污染？
- failure-driven selection 是否比 top-quality selection 更节省 token 和训练算力？
- editable DAG 是否提高 pipeline 修复速度、跨人复用和审计准确率？
- 在相同生成预算下，哪些 operator mutation 真正产生因果增益？
- 自动飞轮迭代多轮后，能力覆盖是否扩大，还是发生 diversity collapse？

实验应至少包含 pipeline seed、training seed、outer holdout、人工污染审计和成本曲线。否则很容易把一次幸运 pipeline 当成系统性提升。

## 15. 最终判断

DataFlow-Harness 的贡献不是一种更花哨的低代码界面，而是把 Code Agent 的输出从一次性脚本变成**平台约束下的可编辑状态机**。对数据飞轮而言，这正好补上了可视化、复用、审计和增量优化的控制平面。

但数据飞轮的闭环不能是：

```text
benchmark 低分 → LLM 仿写题 → LLM 自己打高分 → 加入训练 → benchmark 上升
```

这条路径会把污染、judge bias 和 benchmark overfitting 伪装成进步。更合理的闭环是：

```text
冻结 Benchmark Contract
→ 生成 failure taxonomy
→ 从独立来源构造候选数据
→ 确定性验证 + 多道过滤 + 污染隔离
→ 难度、覆盖和 learning value 选择
→ immutable dataset snapshot
→ shadow training + outer holdout
→ promotion / reject / rollback
```

因此，我对“能不能基于 DataFlow-Harness 结合数据飞轮造数据”的回答是：

> **能。最合理的组合是 DataFlow-Harness 负责可编辑 DAG 和治理，DataFlow 负责 operator 执行，LoopAI 负责评测到训练的闭环，而新增的 Benchmark Contract 和 evidence-based gates 负责保证飞轮优化的是真实能力，不是 benchmark 泄漏或 judge 偏好。**

## 参考资料

1. He et al., [DataFlow-Harness: A Grounded Code-Agent Platform for Constructing Editable LLM Data Pipelines](https://arxiv.org/abs/2607.16617), 2026.
2. [OpenDCAI/DataFlow-WebUI](https://github.com/OpenDCAI/DataFlow-WebUI), source code for the Harness interaction and engineering layer.
3. Liang et al., [DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation](https://arxiv.org/abs/2512.16676), 2025.
4. [OpenDCAI/DataFlow](https://github.com/OpenDCAI/DataFlow), operator and pipeline framework.
5. [OpenDCAI/DataFlow-Skills](https://github.com/OpenDCAI/DataFlow-Skills), procedural pipeline construction skills.
6. [OpenDCAI/Dataflow-LoopAI](https://github.com/OpenDCAI/Dataflow-LoopAI), closed-loop evaluation, data acquisition, and training framework.
7. Li et al., [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794), 2024.
8. [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
9. Liang et al., [HELM: Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110).
10. Liu et al., [EvalPlus: Rigorous Evaluation of LLM-Synthesized Code](https://github.com/evalplus/evalplus).
11. Jain et al., [LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code](https://github.com/LiveCodeBench/LiveCodeBench).

*Research note by Tianyu Xie. Last reviewed: July 28, 2026.*

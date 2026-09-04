# GPT-6 Astra 的“无声推理”：Transformer Block、No-CoT 与 Latent Loop 到底意味着什么

**Transformer** 是一种让序列中的位置相互读取信息、再逐层更新内部向量的神经网络架构。**GPT-6 Astra** 是 OpenAI 在 2026 年 9 月 3 日发布的模型；本文只讨论其公开版本，不假定名称相似的内部模型具有同一结构。随模型发布的 **System Card** 是公开说明能力、安全评测、方法与限制的报告，不是披露全部网络结构的架构论文。

最值得研究的信号并不是某个榜单又涨了多少，而是两组放在一起有些反直觉的结果：一方面，Astra 在不输出中间推理文字时，仍能解决明显更难的问题；另一方面，它公开展示的推理文字更容易被格式指令控制，也比 GPT-5.6 Sol 更短、更少包含可供审计的信息。

先定义本文最小单位。**Token** 是模型处理序列时使用的离散单位，可以是一个字、词或词片段。**隐藏状态（hidden state）**是网络计算过程中持续更新、尚未变成最终文字的数值向量；它包含模型当前编码的信息，不是可以直接阅读的句子。**思维链（Chain of Thought，CoT）**是模型在最终答案前生成的一串自然语言中间步骤；它包含可见或被系统单独保存的文字推理，不包含神经网络层内部持续变化的隐藏状态。**No-CoT** 则是评测时不允许模型生成这些中间推理 token，要求它直接输出答案。No-CoT 不等于“没有计算”，只等于“没有通过额外文字把计算沿生成时间展开”。

本文的核心判断很直白：

> **公开证据支持 GPT-6 Astra 学会了把更多推理压进一次固定深度的隐藏状态计算；公开证据不支持它采用了共享权重的循环 Transformer。Astra 目前更像“更好的推理编译器”，而不是已经被证实的“会反复执行同一个 Transformer block 的处理器”。**

为了不把事实、解释和传闻混在一起，全文采用三个标签：

- **官方确认**：OpenAI 模型页、发布报告或 System Card 明确写出的信息。
- **证据解释**：能够解释公开实验、但 OpenAI 没有确认的机制判断。
- **未经证实的猜测**：网上关于层数、专家数、循环次数或具体内部模块的说法，没有公开材料支持。

![GPT-6 Astra 架构主张的证据阶梯](/images/blog/gpt6-astra-evidence-ladder.svg "图 1：产品能力与评测结果是公开事实；隐藏计算更强是证据解释；循环核心等具体架构目前仍是猜测。")

## 1. OpenAI 这次到底公开了什么

### 1.1 产品层面的确定事实

**上下文窗口（context window）**是一次请求中模型可以共同读取的输入与输出 token 总范围，不等于长期记忆，也不表示模型能同样精确地利用每个位置。OpenAI 模型页列出的 GPT-6 Astra 上下文窗口为 1,050,000 token，最大输出为 128,000 token，并支持 `low`、`medium`、`high`、`xhigh` 和 `max` 五档 **reasoning effort**。Reasoning effort 是**应用程序编程接口（Application Programming Interface，API）**暴露的推理计算强度选项；API 是软件之间提交请求和接收结果的约定接口，不包含模型内部实现。Reasoning effort 的具体实现未公开，因此不能直接翻译成循环次数、层数或思维链长度。

模型指南还公开了异步工具调用、中途引导、持久化 reasoning 与上下文压缩等运行时能力。**异步工具调用**是模型发出工具请求后，系统可以在结果尚未全部返回时继续管理其他工作；**中途引导（mid-turn steering）**是用户可以在模型尚未结束当前轮时补充指令；**上下文压缩（compaction）**是把过长历史变成更紧凑、仍可供后续使用的状态。**Agent 运行时（Agent runtime）**是模型之外负责调度工具、保存任务状态、处理失败和继续执行的软件系统；它包含工作流控制，不包含模型权重本身。这些能力描述的是 API 和 Agent 运行时怎样管理一次任务，不是模型内部怎样重复计算 Transformer 层。

### 1.2 发布报告给出的训练归因

OpenAI 的发布报告把能力提升归因于预训练、强化学习与对齐工作的共同推进。**预训练（pre-training）**是模型先在大规模数据上学习预测和表示的阶段；**强化学习（Reinforcement Learning，RL）**是在模型采取动作或生成回答后，根据奖励信号调整策略的训练方法；它包含从结果反馈中提高某类行为，不要求奖励逐步标注每个隐藏状态。**对齐（alignment）**是让模型行为更符合人类意图、规则和安全要求的一组训练与评测过程，不等同于单一算法。

发布报告还强调 Astra 经常用显著更少的输出 token 完成任务，而更高 reasoning effort 在编码工作流中可以换来更多迭代与验证。这里的“迭代”可以发生在 Agent 调用工具、检查结果和修改答案的外部轨迹里；公开文字没有说它等于神经网络内部的层循环。

### 1.3 System Card 给出的真正异常信号

Astra 的 System Card 明确写道，OpenAI 的 reasoning models 通过强化学习被训练为产生内部 CoT，但 Astra 的 CoT 更短、更不具信息量，也更少倾向于把推理完整说出来。

报告还区分了两个概念：

- **可监测性（monitorability）**是观察某种信号后，监测器能否发现模型正在做什么或是否存在不良行为。它包含从 CoT 或动作轨迹中识别意图，不表示信号一定忠实呈现全部内部因果过程。
- **可控性（controllability）**是外部指令能否改变某种信号的形式或内容。它包含让 CoT 遵循大小写、文风等约束，不等于能控制真正决定答案的全部隐藏计算。

在 750 到 1,250 token 的 CoT 区间，System Card 报告的约束服从率为：GPT-6 Astra `60.9%`、GPT-5.6 Sol `16.1%`、GPT-5.5 Thinking `1.7%`。Astra 甚至能让推理区交替使用大小写，或填充与任务无关的意象，同时仍然回答正确。

这组结果最稳妥的含义是：**展示出来的 CoT 越来越像一个可以单独控制的输出通道，它不再等于产生答案所必需的全部计算。** 但“文字不是全部计算”不等于“隐藏状态里存在秘密自然语言”。**隐写（steganography）**是把信息故意藏进表面无关的文字模式中，使普通观察者难以发现；公开结果也不等于模型用隐写术编码了一条不可见 CoT。OpenAI 在所测设置中没有发现 steganographic CoT。

## 2. 一个 Transformer block 内部到底做了什么

**Transformer block** 是 Transformer 反复堆叠的基本计算单元。一个典型 block 包含归一化、注意力、前馈网络和残差连接；它不包含完整模型的词表切分、所有层和最终采样。**前向传播（forward pass）**是输入经过整个网络直到得到本次输出分布的一次计算；一次 forward pass 会顺序穿过许多个 block，绝不是“只算一个 block”。

要理解无声推理，最重要的是 **残差流（residual stream）**。它是贯穿网络各层的向量通道：每个 block 读取当前向量，计算一个增量，再把增量加回去。它包含当前 token 在这一层积累的语义、关系和任务状态，不是一个人类可直接读写、字段固定的数据库。

可以把第 `l` 个 block 写成：

```text
u_l     = x_l + Attention(Norm(x_l))
x_(l+1) = u_l + FFN(Norm(u_l))
```

这里的 **归一化（normalization）**是把向量数值调整到更稳定的尺度，减少训练和推理中的数值漂移；它不负责推理本身。**注意力（attention）**让每个位置按内容相关性从其他位置读取并汇总信息，包含匹配、复制和跨位置组合，不包含模型的全部非线性计算。**前馈网络（Feed-Forward Network，FFN）**对每个位置独立执行参数化的非线性变换，常被解释为对训练中学到的特征和知识进行匹配与改写；它不是推理时可任意写入的内存。

于是一个 block 的直觉可以压缩成两句话：

1. Attention 决定“这一步该从上下文取什么”。
2. FFN 决定“取回来的信息与当前状态应该变成什么”。

残差加法则让新结果不会完全覆盖旧状态。几十甚至上百个不同参数的 block 串起来后，模型已经拥有一条很深的、顺序执行的隐藏计算轨迹：前层识别局部关系，中层组合约束，后层把状态推向答案。即使一个额外的 CoT token 都不生成，这条轨迹仍然存在。

![一个 Transformer block 如何更新残差流](/images/blog/gpt6-astra-transformer-block.svg "图 2：Attention 从上下文取回信息，FFN 做逐位置变换，两次残差加法把结果写回贯穿各层的状态。")

## 3. “一次前向传播”为什么也能推理

假设模型有 `L` 个不同参数的 block。对一个答案 token 而言，它会经历：

```text
x_0 → Block_1 → x_1 → Block_2 → ... → Block_L → 答案分布
```

每个箭头都是一次隐藏状态更新。这里没有显式循环，因为 `Block_1` 与 `Block_2` 的参数不同；但它依然是有先后顺序的多步计算。就像把一个固定程序完全展开：第一段做解析，第二段传递约束，第三段形成候选，最后一段选择答案。层的职责未必如此整齐，也不能靠观察相关性就断言某层“专门做了除法”，但网络整体确实具有固定深度的串行计算预算。

**潜在推理（latent reasoning）**是中间推理主要发生在连续隐藏向量中，而不是每一步先解码成离散文字再重新输入。它包含不可见的向量更新，不保证这些向量对应完整句子，也不保证它们能被忠实翻译成人类理由。按这个宽定义，固定深度 Transformer 本来就有 latent computation；研究问题是训练能否让它在这些有限更新里完成更复杂、更可组合的推理。

这正是 Astra 的 No-CoT 结果有意义的地方：它说明固定深度路径可承载的任务难度显著上升，而不是证明“没有过程，答案突然出现”。

## 4. 30.9 分钟不是模型思考了 30.9 分钟

Astra System Card 引用了英国人工智能安全研究所（UK AI Safety Institute，UK AISI）的 No-CoT 数学评测。**人类等效时间跨度（human-equivalent time horizon）**是把题目难度换算成合格人类通常需要多长时间解决，并寻找模型达到某个可靠性阈值时对应的难度；它衡量的是模型能可靠完成多难的任务，不是模型实际运行了多久。

在 50% 可靠性处，报告给出：

| 模型 | No-CoT 人类等效时间跨度 |
| --- | ---: |
| GPT-5.6 Sol | 3.6 分钟 |
| GPT-6 Astra | 30.9 分钟 |

因此，Astra 的公开结果大约提高一个数量级。正确读法是：“不允许输出推理文字时，Astra 能以 50% 可靠性解决通常需要人类约 30.9 分钟的数学题。”错误读法则包括：“模型内部循环了 30.9 分钟”“模型执行了 30.9 分钟对应的 token”或“它一定动态增加了网络深度”。报告没有提供这些信息。

UK AISI 也提醒，数据污染可能抬高估计，测试时间有限。**数据污染（contamination）**是评测题或高度相似内容出现在训练数据中，使得分混入记忆效应；它不等于已经证明模型背过题，但会削弱从分数推断泛化能力的力度。

## 5. 三种“多想一会儿”不是一回事

模型获得额外计算至少有三条路径。

### 5.1 固定深度的直接回答

输入只通过一次完整网络，每个输出 token 都经过固定的 `L` 个 block。计算发生在残差流里，训练可以让不同层协作完成越来越复杂的变换，但部署时不能临时把同一组层多跑几遍。

### 5.2 用语言 token 展开思维链

模型先生成一段 CoT，再生成答案。每增加一个推理 token，整个 `L` 层网络都要再运行一次，同时读取此前 token 的缓存。注意力把每个历史位置变换成用于匹配的 **key（键）**向量和用于汇总内容的 **value（值）**向量。**键值缓存（Key-Value Cache，KV Cache）**保存这些历史 key 和 value，避免每次重新计算全部前缀；它包含可追加的历史表示，不是通用可写内存。

这条路线的优势是能读、能监督、能随 token 数延长计算；代价是每一步必须压成语言，生成串行、KV Cache 增长，而且可见文字可能只是一个受控叙述通道。

### 5.3 在隐藏空间里显式循环

**递归深度（recurrent depth）**是把同一组网络层在深度方向重复执行，让上一轮隐藏状态成为下一轮输入。**Latent loop** 在本文中特指这种对连续隐藏状态的显式循环：它必须包含状态回流和参数共享，不包含普通的逐 token 生成，也不包含仅仅堆叠很多参数互不相同的层。

其形式可以写成：

```text
e   = Prelude(input)
s_0 = initial_state
s_i = RecurrentCore(e, s_(i-1))
out = Coda(s_r)
```

`Prelude` 是只运行一次、把输入准备成内部表示的前奏模块；`RecurrentCore` 是参数共享、执行 `r` 轮的循环核心；`Coda` 是只运行一次、把最终状态变成输出的收尾模块。循环次数 `r` 可以固定，也可以由停止器按问题决定。

![固定深度、文字思维链与 latent loop 的计算路径](/images/blog/gpt6-astra-three-compute-paths.svg "图 3：三条路径都能增加或承载计算，但只有第三条在同一个输出 token 之前显式复用共享核心。")

## 6. 真正的 latent loop 研究已经做到哪一步

### 6.1 Recurrent Depth：把同一核心跑很多轮

Geiping 等人的 *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* 是最直接的例子。它把模型分成前奏、共享递归核心与收尾三部分。论文的 3.5B 参数模型使用 800B token 训练，这里的 `B` 表示十亿；大型结构为 `(2, 4, 2)`，即 2 层前奏、4 层循环核心和 2 层收尾，平均递归 32 次。输入表示 `e` 每轮重新注入，避免循环状态逐渐忘掉原题；初始状态加入随机性，训练则鼓励不同初始轨迹收敛到可用结果。

**截断反向传播（truncated backpropagation）**是训练循环网络时只保留最近若干轮的梯度路径，以减少显存与计算；它包含牺牲更早轮次的直接梯度。该工作只对最后 8 次迭代回传梯度。测试时增加递归次数，若干任务还能继续改善，说明模型确实学会利用额外的隐藏计算，而不只是把训练深度硬编码进输出。

作者还观察到隐藏状态会趋向**固定点**、形成**轨道**或持续漂移。固定点是继续应用循环核心后状态几乎不再变化的位置；轨道是状态在一组位置间周期运动。它们说明循环形成了可分析的动力过程，但“收敛”不等于“正确”：模型可以非常稳定地收敛到错误答案。

### 6.2 Coconut：把隐藏状态当作下一个输入

Coconut 的全称是 **Chain of Continuous Thought**。它取上一推理步的最后隐藏状态，直接作为下一步的输入 embedding，而不先把状态投影成某个词。**Embedding** 是把离散 token 映射为连续向量的表示；Coconut 反过来让连续向量占据原本属于 token embedding 的输入位置。

这和 Recurrent Depth 不同：Coconut 沿“输入槽位”加入连续思维步，Recurrent Depth 则在同一个输出位置内部重复整组网络层。Coconut 作者提出连续状态可能同时保留多个候选，从而呈现类似广度优先搜索的行为；这是一种实验解释，不是已经读出了一棵真实、离散的搜索树。

### 6.3 Pause Tokens 与 Quiet-STaR：仍在 token 轨道上

**Pause token** 是没有普通语义、专门为模型增加顺序计算步的学习符号。模型在回答前先处理若干 pause token，每一步仍通过完整 Transformer 并进入 KV Cache。它是显式 CoT 与纯连续循环之间的桥梁：没有可读理由，但仍沿 token 时间展开。

Quiet-STaR 在训练时让模型围绕文本位置生成内部 rationale，再学习这些 rationale 是否改善后续预测。**Rationale** 是为一个预测提供的文字理由或推导，它仍由离散 token 构成。Quiet-STaR 因而属于“在说话前用 token 思考”，不是权重共享的 latent loop。

### 6.4 Mixture-of-Recursions：让不同 token 思考不同轮数

Mixture-of-Recursions 复用共享层组，并用路由器给不同 token 分配不同递归深度。**路由器（router）**是根据当前隐藏状态选择计算路径的小型模块；它包含决定某 token 是否继续进入下一轮，不保证这一决策具有人类可读的原因。这个方向把“循环”从统一深度推进到按 token 分配计算，但动态深度会让批处理中的 token 不同步，硬件效率未必随理论计算量同步改善。

## 7. 强化学习怎样把推理“编译”进固定深度

现在回到 Astra。System Card 的关键表述是：CoT 可控性的增加尤其在强化学习阶段出现，而且 OpenAI 明确写道：

> “We are also quite confident that changes in CoT controllability are not differentially due to any architectural changes.”

也就是 OpenAI 相当确信，这项变化不是由架构变化差异化导致的。官方材料因此主动削弱了“架构突然变成 latent loop”的解释。

**编译式推理（compiled reasoning）**是本文使用的解释性比喻：训练把经常反复执行的推理模式压进固定的网络权重和层间变换，使模型在部署时用较短轨迹得到结果。它包含以更大的训练成本换取更少的推理步骤，也常被称为**摊销推理（amortized inference）**；“摊销”表示训练阶段为一类问题预先付出成本，之后许多样本共享这项成本。它不表示网络真的生成了程序代码，也不保证遇到分布外问题仍能直接解答。

一个直觉例子是两位学生：第一位每次都把公式从定义推导一遍；第二位经过大量练习，看到题型便能在脑中完成关键变换，只写答案。第二位不是没有推理，而是把常见步骤内化成了更直接的状态转移。强化学习可以反复奖励“答案正确、轨迹短、工具使用有效”的行为，于是网络逐渐把高频策略压进固定深度。

这能统一解释三件事：

1. Astra 在 No-CoT 下仍能解决更难的题，因为更多计算被固定层的隐藏状态承载。
2. Astra 的外显 CoT 可以更短、更容易服从无关格式，因为它不再承担全部必要计算。
3. 更高 reasoning effort 仍可能提高复杂任务表现，因为模型还可以在固定深度之外使用更多输出、验证或工具迭代。

但这只是最符合当前证据的机制解释，不是 OpenAI 公布的网络图。

## 8. 为什么 CoT 更可控，监控反而更困难

过去常有一个隐含假设：模型写出的推理越完整，我们越能从中看见它为什么做出动作。Astra 让这个假设变得危险。

如果模型可以在推理区写无关意象，同时保持答案正确，那么至少说明最终答案的必要信息不必逐字经过这条可见通道。CoT 监测器看到的可能是“模型愿意报告什么”，而不是“模型必须依靠什么”。这不证明模型在欺骗，也不证明存在另一条完整的秘密句子；固定深度残差流本来就能携带大量非语言信息。

System Card 同时报告 Astra 的 **仅动作可监测性（action-only monitorability）**高于 GPT-5.6 Sol。它是只观察模型采取的工具调用、文件修改或其他外部动作，不读取 CoT，也能否发现风险行为；它包含用可验证结果和行为轨迹做审计，不依赖模型自述动机。

工程上，这意味着监控重心应该移动：

- 不再把 CoT 当作唯一审计日志；
- 更重视工具参数、权限边界、环境变化和最终产物；
- 用独立验证器检查事实、代码、数学结果和副作用；
- 把 CoT 视为一个有用但可能被优化和控制的观测面。

这不是放弃可解释性，而是承认“会说理由”和“计算过程忠实可见”是两种能力。

## 9. 要证明 GPT-6 使用 latent loop，还缺什么证据

严格证明至少需要下面一种公开证据：

1. 架构说明明确画出共享参数核心、隐藏状态回流和退出条件。
2. 模型代码或权重结构显示同一组层被多次调用。
3. 控制递归轮数的消融实验，展示质量、延迟和状态轨迹随轮数系统变化。
4. 在输出 token 数、工具调用和上下文完全不变时，仍能直接增加内部递归计算预算。

仅有下面这些现象不够：输出更短、reasoning effort 档位更多、一次 forward pass 解出难题、思维链更难监测、支持 compaction，或发布文案使用“iterate”“think longer”等词。它们都可能由训练策略、推理服务、Agent runtime 或固定深度网络解释。

网上流传的具体层数、专家数、“每个 token 内循环若干次”以及所谓 Astra latent core，目前都应归入**未经证实的猜测**。合理猜测可以帮助设计实验，但不能倒过来写成官方事实。

## 10. 一个明确标为假设的 GPT-6 执行图

如果只根据公开证据构造最保守的工作模型，我会这样理解 Astra：

```text
用户输入
  ↓
固定深度 Transformer：在残差流中完成更多被 RL 编译的推理
  ↓
策略层判断是否需要继续：直接回答 / 生成短 CoT / 调工具 / 验证
  ↓
Agent runtime 保存 reasoning 状态、缓存与工具结果，并允许中途引导
  ↓
最终答案与可审计动作
```

这里第一层是**证据解释**，后面几层对应公开产品行为；图中没有假设共享递归核心。未来若 OpenAI 公布 latent loop，最可能插入的位置是在固定深度 Transformer 内部，或作为输出前可重复调用的隐藏计算模块。但在证据出现前，这一部分必须保持空白。

## 11. 最直白的结论

Transformer 并不是只有在“写思维链”时才思考。每一个输出 token 产生前，信息已经穿过许多 Transformer block；Attention 不断取回相关上下文，FFN 不断变换局部状态，残差流把中间结果一层层传下去。这本身就是一条不可见的计算链。

GPT-6 Astra 的公开结果说明，训练正在让这条固定深度计算链承担更多工作。它可以少写、甚至不写中间文字，却解决过去需要显式 CoT 才能处理的难题；同时，写出来的 CoT 更像一个可控制的界面，而不再是内部过程的完整窗口。

真正的 latent loop 更进一步：同一组参数会在一个答案 token 之前重复处理隐藏状态，并按需要决定何时停止。Recurrent Depth、Coconut 和 Mixture-of-Recursions 已经展示了这条研究路线的不同实现，但 OpenAI 没有公开证明 Astra 使用其中任何一种。

所以，今天最准确的说法不是“GPT-6 架构专注 loop”，而是：

> **GPT-6 Astra 提供了更强 latent computation 的行为证据；latent loop 是解释和延伸这种趋势的重要研究方向，但仍不是已公开确认的 Astra 架构事实。**

## 参考资料

- [OpenAI, GPT-6 Astra: A new generation of intelligence](https://openai.com/index/gpt-6-astra/)
- [OpenAI, GPT-6 Astra System Card](https://deploymentsafety.openai.com/gpt-6-astra/gpt-6-astra)
- [OpenAI, GPT-6 Astra Model](https://developers.openai.com/api/docs/models/gpt-6-astra)
- [OpenAI, Model guidance for GPT-6 Astra](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra)
- [Geiping et al., Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171)
- [Hao et al., Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)
- [Bae et al., Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation](https://arxiv.org/abs/2507.10524)
- [Goyal et al., Think before you speak: Training Language Models With Pause Tokens](https://arxiv.org/abs/2310.02226)
- [Zelikman et al., Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)
- [Giannou et al., Looped Transformers as Programmable Computers](https://arxiv.org/abs/2301.13196)

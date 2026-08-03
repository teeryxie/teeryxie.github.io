# The Silent Thought 论文精读：全双工语音模型如何在倾听时进行隐式推理

人在对话中并不是等对方说完最后一个字，才从零开始理解。我们会随着语音展开持续修正意图判断：听到“帮我订一张明天……”时先形成出行假设，听到“去上海的高铁”后补全目的地与交通方式，听到“算了，改后天”时撤销旧日期。真正自然的交互，依赖的不是一句话结束后的单次推理，而是**监听期间不断演化的内部状态**。

《[The Silent Thought: Modeling Internal Cognition in Full-Duplex Spoken Dialogue Models via Latent Reasoning](https://arxiv.org/abs/2603.17837)》正面研究了这个空白。Donghang Wu、Tianyu Zhang、Yuxin Li、Hexin Liu、Chen Chen、Eng Siong Chng 与 Yoshua Bengio 提出的 **FLAIR（Full-duplex LAtent and Internal Reasoning）**，不让全双工模型在用户说话时反复生成没有信息的 `<SIL>`，也不强迫它输出一条可见的流式 Chain-of-Thought，而是把上一时刻的词表概率分布投影回 embedding 空间，递归地作为下一时刻输入。训练时，一个能看到完整用户语音和未来助手回复的非因果 Global-aware Expert 提供“后验”latent target；推理时移除 Expert，只留下严格因果的流式模型。

这篇工作已被 ICML 2026 接收。它最值得关注的贡献不是把“思考”包装成新的拟人化概念，而是提出了一个具体的计算问题：**全双工模型在等待用户说完之前，那些已经支付的逐帧计算到底应该形成什么状态？**

![Listening-time computation problem](/images/blog/silent-thought-problem.svg "图 1：传统全双工模型、显式流式 CoT 与 FLAIR 在用户说话期间采用的三种计算路径。本文根据论文机制整理。")

## 1. 一页结论：这篇论文做了什么

先给出我的总体判断。

- **问题抓得准确。** 全双工模型必须与音频时间轴同步；用户说话期间连续预测 silence/padding，实际上浪费了已经发生的前向计算。
- **方法很克制。** FLAIR 没有增加一条可见文本思维链，而是在统一 LLM 中循环连续 embedding；这更容易随时中止并切换到回答。
- **训练设计是核心。** latent state 没有人工真值，论文用能看完整对话的 Global-aware Expert 形成 posterior，再通过 KL 正则把“后见之明”蒸馏给 causal student。
- **最可信的证据是同架构对照。** 加入 latent reasoning 后，Llama Questions 从 73.0 提升到 78.0，MMSU 从 50.2 提升到 56.2；但 TriviaQA 从 53.8 降到 51.2，绝不是所有指标全面提高。
- **交互代价不为零。** 在 Full-Duplex-Bench 上，barge-in 回复质量从 4.08 提升到 4.22，但 turn latency 从 0.37 秒增加到 0.43 秒。论文说明质量收益没有严重破坏实时性，却不能据此声称“没有任何延迟成本”。
- **它还不是完整的 agent cognition。** 当前 latent state 面向下一段语言回复，没有显式工具状态、长期记忆、假设置信度、动作撤销和 playback-aware rollback。它更像 streaming belief state 的一个起点，而不是连续时间 Agent Policy 的终点。

一句话概括：**FLAIR 把“用户说话时的空白占位”改造成“可递归更新的连续内部状态”，并用训练期的全局 Expert 教会严格因果模型如何形成这个状态。**

## 2. 为什么 `<SIL>` 是结构性浪费

端到端全双工 Spoken Dialogue Language Model 同时接收用户音频并生成助手流。为了让输入和输出在同一时间轴上对齐，用户说话而助手保持沉默时，模型通常也要逐步运行，只是输出 `<SIL>`、`<PAD>` 或 pause token。

这并非完全“没有做事”：模型的 hidden state 仍然读取了新增音频。但在标准自回归接口中，下一步反馈的往往还是离散 silence embedding。于是最显式、最稳定的反馈信号只有“继续不说话”，而不是“当前听到了什么、哪些假设被支持、哪些候选答案被排除”。如果一个用户说 12 秒，12.5 Hz 的 LLM 已经执行约 150 次更新；让这 150 次更新只服务于 silence prediction，显然不是理想的计算分配。

问题的本质并非 silence token 本身有害，而是：

1. 模型缺少一个专门承载监听期认知演化的自回归通道；
2. 最终回答的监督只发生在用户 turn 结束之后，较难把信用分配给早期音频前缀；
3. 用户何时结束不可预知，任何监听期计算都必须可抢占、可继续、严格因果。

这也是 FLAIR 与普通“先 ASR、后 LLM”的差别。级联系统可以在 ASR partial transcript 上提前推理，但会引入 transcript revision、模块同步与额外缓存。FLAIR 尝试在同一个流式 backbone 的时间轴内解决问题。

## 3. 为什么不直接生成流式 Chain-of-Thought

一个直觉方案是：每收到一段音频，就在后台生成几句文本推理。例如听到“请比较 A 和 B……”时先写出“需要识别两个对象和比较维度”。Chronological Thinking、SHANKS 以及 streaming CoT 方向都在探索类似的 think-while-listening。

但语音流和离散 CoT 存在天然冲突。

**第一，partial observation 会反复推翻结论。** 用户说“我不要红色的……不，我是说不要太深的红色”，前缀上的文本 CoT 很可能过早收敛。后续每次修改都要显式撤销先前 token，但自回归文本天然是追加式的。

**第二，思维链长度与 turn boundary 不同步。** 用户可能在任意时刻结束。如果后台正在生成固定长度 CoT，系统要么等待它结束而增加延迟，要么粗暴截断并处理残留状态。

**第三，数据构造昂贵。** 必须为每个音频前缀生成时间正确、不能偷看未来的 reasoning target。一个最终答案正确的 Teacher，也未必能给出严格因果的中间轨迹。

**第四，显式 CoT 会占用 token 带宽。** 全双工系统同时承担音频理解、轮次控制和语音生成；长文本 reasoning 会与实时输出竞争计算，并扩大缓存。

因此，FLAIR 的选择是：监听时只让连续状态演化，开口后才恢复可见文本 token。它牺牲了可读性，换取可抢占性与时间对齐。

## 4. FLAIR 的核心：把词表分布变回输入 embedding

FLAIR 不是简单地把最后一层 hidden state 原样反馈。监听阶段，LLM 先照常产生 vocabulary logits，再经过 Softmax 得到整个词表上的概率分布，最后对 vocabulary embedding matrix `E` 加权：

```text
Z_(t-1) = Softmax(y_txt_(t-1)) · E
```

`Z_(t-1)` 是一个“soft token”：它不是某个确定词的 embedding，而是词表中所有 token embedding 的加权和。下一时刻，模型把它作为自回归输入，同时读取新的语音表示。这样，latent state 既可以表达多个候选概念的叠加，又处在与普通文本 token 相同的输入空间，避免额外设计完全不同的 latent interface。

论文还训练一个 timing head，预测 `G_t`：

```text
G_t = 0  → 继续监听，反馈 latent embedding，并向语音模块输出 <SIL>
G_t = 1  → 开始/继续回答，输出离散文本 token，并反馈 token embedding
```

当用户 barge-in，模型预测结束当前输出，重新回到 `G_t=0` 的 latent reasoning。于是同一个 backbone 在两种计算状态间切换：

- listening mode：连续概率分布形成的 soft embedding；
- speaking mode：普通离散文本 token。

![FLAIR causal timeline](/images/blog/silent-thought-flair-timeline.svg "图 2：FLAIR 推理时间轴。监听期递归 latent embedding；预测到回答边界后切换到显式 token；用户插话后再次进入监听。")

这个设计与 Coconut、CODI 等 continuous thought 方法有家族相似性：都避免过早提交到单一离散 token。差别在于 FLAIR 的 latent step 不是为一道静态推理题额外循环若干次，而是被真实音频时间轴驱动；用户每提供一个新片段，内部状态更新一次。它的“计算深度”因此与用户说话时长绑定。

## 5. 没有 latent 标签，如何训练

真正困难的不是定义 `Z_t`，而是监督它。最终文本回复有 ground truth，latent embedding 却没有。若直接用最终答案反向传播，监听窗口很长，模型可能学到退化状态；若把某段文本 CoT 当标签，又回到了专门构造流式推理轨迹的问题。

FLAIR 的解决方式是引入训练期的 **Global-aware Expert `Q_phi`**。它不是部署时的另一个 Agent，而是一个非因果 encoder：

- 输入完整用户语音 embedding `X`；
- 输入完整助手文本回复 embedding `H_txt`；
- 可以看到未来，因此形成“知道最终应如何回答时，监听阶段理想状态应是什么”的 approximate posterior；
- 输出经过线性层和 Softmax 后，同样对词表 embedding 加权，得到 Expert latent `Z_t`。

相比之下，部署模型 `P_theta` 在时刻 `t` 只能看到 `X_<=t` 与过去状态，形成 causal prior。训练目标是让 causal prior 尽量接近 Expert posterior。

这里需要非常谨慎地使用“后验”一词。论文采用 variational inference/ELBO 的表述，但实际对齐对象是词表 Softmax 分布，并不意味着 latent cognition 具有可直接解释的概率语义。Expert 也不保证产生唯一正确的思维过程；它只是借助未来信息构造更有利于回复重建的训练信号。

## 6. ELBO 风格目标逐项拆解

训练时，完整输入可概括为：

```text
H_in = X + (1 - G) ⊙ Z + G ⊙ H_txt
```

在用户 turn，`G=0`，输入语音与 Expert latent；在助手 turn，`G=1`，输入语音与真实回复 token embedding。总损失包含三部分。

**Conditional Reconstruction** 只在助手回复区域计算，要求模型在用户语音、latent history 和过去回复条件下重建正确 token：

```text
L_reco = -Σ_t G_t log P_theta(y_t | X_<=t, Z_<=t, y_<t)
```

**Variational Regularization** 只在监听区域计算，让 causal model 当前输出的词表分布接近 Expert posterior：

```text
L_regu = Σ_t (1-G_t) KL(
  stop_gradient[W_expert_t]
  || Softmax(y_hat_t)
)
```

`stop_gradient` 很关键：KL 更新 student，不让 student 的错误反过来拖动 Expert target。

**Timing Loss** 学习何时监听、何时开口：

```text
L_time = -Σ_t [
  G_t log G_hat_t
  + (1-G_t) log(1-G_hat_t)
]
```

最终目标为：

```text
L = L_reco + α L_regu + β L_time
α = 3, β = 5
```

为了强化 turn boundary，论文把 `<BOS>` 的重建权重乘 20，把 `<EOS>` 乘 10。换言之，模型能否及时开口和停下，不只是 latent reasoning 自发涌现，而是被明确的边界监督强力塑造。

![ELBO training and inference](/images/blog/silent-thought-elbo-training.svg "图 3：训练时 Global-aware Expert 利用完整语音和未来回复提供 posterior target；因果 student 通过 reconstruction、KL regularization 与 timing loss 学习；推理时 Expert 被移除。")

## 7. 训练与推理不能混为一谈

这篇论文最容易被误读成“模型推理时偷看未来答案”。实际不是。

训练阶段，Expert 确实使用完整对话和未来回复，这类似知识蒸馏中的强 Teacher。Student 在 teacher forcing 下学习两件事：一是何时从监听切换到回答；二是在只见过前缀时，让自己的 latent prior 逼近全局 posterior。

推理阶段：

1. Global-aware Expert 被完全丢弃；
2. 模型只能读取当前和历史音频；
3. `G=0` 时，把自己的上一时刻词表分布投影成 soft embedding；
4. `G=1` 时，恢复标准离散 token 生成；
5. speech generator 把文本侧结果转换为流式语音。

因此，训练允许 non-causal hindsight，部署仍是 causal streaming。真正值得研究的问题不是“是否偷看”，而是 posterior-to-prior distillation 在多大程度上学到了可泛化的前缀更新规则，而非只对合成数据的时间模板过拟合。

## 8. 620K 小时数据是如何构造的

论文的数据规模很大，总计约 620K 小时，分为三类。

| 数据子集 | 规模 | 主要用途 |
| --- | ---: | --- |
| Speech continuation | 530K hours | 预训练全双工时序、语义连续性与轮次切换 |
| Instruction-following QA | 70K hours | 单轮/多轮问答与回复质量 |
| ASR-QA | 20K hours | 真实语音、背景噪声与转写内容理解 |

Speech continuation 从连续文本构造双人对话：文本段落交替分配给 user 与 agent，80% 概率在单句后结束一个 turn，超过 200 words 则强制换角色；两位不同 speaker 合成双方语音，并交换角色提高利用率。这类数据便于大规模制造时间对齐，却不等价于真实人类对话中的犹豫、抢话、共同完成句子和语用修复。

Instruction QA 包含约 10K 小时单轮数据与最长约 4 分钟的多轮数据，Wikipedia 等上下文作为主题锚点。ASR-QA 使用真实 ASR corpus 及其 transcription 生成问题，因此保留了一部分真实噪声与声学变化。

文本生成使用 GPT-OSS-120B、Qwen2.5-72B-Instruct 与 Llama-3.1-70B-Instruct；TTS 使用 Chatterbox、Magpie-TTS 与 MoonCast。声音池来自 LibriTTS、YODAS、HiFi-TTS 中超过 100K 个语音片段和 20K 位说话人。噪声池包含来自 Freesound 与 MUSAN 的 10,000 段背景音，以 50% 概率注入，SNR 覆盖 0–60 dB。

打断数据也采用显式协议：当 agent 回复超过 4 秒时，以 10% 概率在 utterance 的 20%–80% 位置插入用户语音；target 中加入 8-token reaction delay，约 0.64 秒后预测 EOS，再返回 latent reasoning。

这个细节决定了我们应如何归因结果：**FLAIR 的打断能力不只来自 latent state，还来自人工设定的中断采样与固定反应延迟标签。** latent reasoning 可能改善被打断后的内容理解，但“多久闭嘴”主要由 timing supervision 和数据协议塑造。

## 9. 模型结构与三阶段训练

文本 backbone 是 Qwen2.5-7B-Instruct。语音侧使用约 600M 参数、基于 Parakeet 的 streaming encoder，并设置 causal convolution context；1024 维 modality adapter 将其接入 LLM。语音生成由 audio codec 与 streaming flow matching 组件组成，整体参考 CosyVoice 2 风格。

Speech encoder 与 LLM 以 12.5 Hz 运行，audio codec 为 25 Hz，因此每个 LLM frame 对应两个 speech token。Speech encoder 与 audio codec 在训练中冻结。

训练分三步：

1. **Pre-training。** 先在 speech continuation 上训练普通全双工模型，此时还没有 latent reasoning，建立语音理解、文本预测和 turn-taking 基础。
2. **Latent Reasoning SFT。** 先只用 `L_reco` 训练 Expert，使其 latent 能帮助重建回复；随后联合优化完整目标，让 causal LLM 对齐 Expert posterior。
3. **Speech Synthesizing SFT。** 冻结其他参数，只训练 speech generation module，降低联合优化造成的音质扰动。

训练使用 64 张 A800 80GB、NeMo Toolkit、BFloat16 和 AdamW；`beta=(0.9, 0.98)`、weight decay 为 0、gradient clipping 为 1.0。预训练学习率 5e-4、warmup 2500 steps；SFT 学习率 5e-5，并采用 inverse square root annealing。

这不是一个“小模块即插即用”的实验。即使 latent mechanism 本身简洁，结论依赖 620K 小时数据、强 backbone、600M encoder 和大规模训练。任何复现都必须固定数据、backbone 与计算预算，否则无法判断收益来自 latent reasoning 还是更强预训练。

## 10. QA 结果：有效，但不是全面提升

论文最重要的 Table 1 应优先比较同一个 FLAIR 系统的 `w/o thk` 与 `w/ thk`，因为 Moshi、SALMONN-omni 的 backbone、数据与训练规模并不完全相同。

| Benchmark | FLAIR w/o latent | FLAIR w/ latent | 变化 |
| --- | ---: | ---: | ---: |
| Llama Questions | 73.0 | 78.0 | +5.0 |
| Web Questions | 41.7 | 43.0 | +1.3 |
| TriviaQA | 53.8 | 51.2 | -2.6 |
| SDQA | 54.4 | 56.2 | +1.8 |
| AlpacaEval GPT score | 3.80 | 3.85 | +0.05 |
| CommonEval GPT score | 3.54 | 3.65 | +0.11 |
| OpenBookQA | 72.9 | 74.2 | +1.3 |
| MMSU | 50.2 | 56.2 | +6.0 |

最大的绝对增益出现在 MMSU 与 Llama Questions，说明监听期 latent accumulation 对需要整合较长问题或选择性推理的任务可能更有帮助。WebQ、OpenBookQA 的改善较小；TriviaQA 明显下降。这可能来自 soft distribution 对实体细节的平滑，也可能只是评测方差，论文没有给出多随机种子置信区间，无法区分。

作为参照，Moshi 在 LlamaQ、WebQ、TriviaQA、SDQA 上分别为 54.5、22.1、16.7、15.6；SALMONN-omni 在 LlamaQ、WebQ、TriviaQA 上为 73.6、43.7、56.0。FLAIR 整体很强，但跨模型领先同时包含 backbone、encoder、620K 小时训练数据和方法差异。**只有同架构开关 latent 的结果，才能相对干净地支持方法贡献。**

![FLAIR benchmark changes](/images/blog/silent-thought-results.svg "图 4：同架构加入 latent reasoning 后的关键变化。绿色表示提高，红色表示下降；交互质量提高但部分延迟也增加。")

## 11. 交互结果：质量与时延必须一起看

在 Impatient dataset 上：

| Model | E2E latency | Turn-taking latency | Barge-in success | MOS |
| --- | ---: | ---: | ---: | ---: |
| Moshi | - | 0.81 s | 55.1% | 3.9 |
| FLAIR w/o latent | 0.33 s | 0.49 s | 100% | 4.3 |
| FLAIR w/ latent | 0.39 s | 0.46 s | 100% | 4.3 |

latent 版本的 turn-taking latency 从 0.49 降到 0.46 秒，但 E2E latency 从 0.33 增加到 0.39 秒，成功率和 MOS 不变。可以说 latent mechanism 没有破坏交互稳定性，却不能说所有 latency 都变好。

Full-Duplex-Bench 给出更细的 trade-off：

| Model | Turn TOR | Turn latency | Barge-in TOR | Barge-in quality | Barge-in latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Moshi | 94.1 | 0.27 s | 100 | 0.77 | 0.26 s |
| FLAIR w/o latent | 94.1 | 0.37 s | 89.0 | 4.08 | 0.35 s |
| FLAIR w/ latent | 93.0 | 0.43 s | 92.0 | 4.22 | 0.36 s |

加入 latent reasoning 后，barge-in response quality 从 4.08 提升到 4.22，barge-in TOR 从 89 提升到 92；同时 turn TOR 从 94.1 降到 93.0，turn latency 从 0.37 增到 0.43 秒，barge-in latency 从 0.35 增到 0.36 秒。

因此，最准确的结论是：**latent reasoning 改善了部分问答与被打断后的回复质量，交互成本较小但真实存在。** 论文摘要中“without incurring any inference latency”更适合理解为没有增加额外串行 reasoning pass 或外部模块，而不是每个实测 latency 数字都不变。

## 12. 消融实验告诉了我们什么

论文附录还提供了几个重要线索。

**预训练不可省略。** 仅 pretraining 的模型在 LlamaQ、WebQ、OpenBookQA、MMSU 上分别达到 63.6、40.4、61.3、49.5。后续 instruction 与 latent SFT 的收益建立在已经会稳定全双工交互的 base 上。

**语音 encoder 尺寸影响的首先是“能否开口”。** 120M encoder 在 LlamaQ 上 response success 只有 70.7%；换成 600M encoder 加简单线性 projection 后达到 100%。这说明模态对齐不是旁枝：如果声学 embedding 与 LLM 空间错配，timing head 和 latent reasoning 都无从谈起。

**更大的 Expert 收益有限。** BERT 与 T5-Large 结果几乎相同，T5-3B 只有小幅增益却显著增加计算，最终选用 BERT。这支持“训练信号结构比 Teacher 参数规模更重要”的判断。

**latent reasoning 与 waveform backend 大体正交。** Streaming Flow Matching 与 Multi-layer Audio Codec 的 MOS 分别约为 4.2 与 4.3。latent 部分位于文本/状态层，声码器替换主要改变音质和生成实现。

**t-SNE 只能提供线索。** 作者在 300 个 Llama Questions 上可视化，latent embedding 位于 input audio 与 target text 之间，并解释为语音到答案的 bridge。二维 t-SNE 会强烈扭曲全局距离；“点落在中间”不能证明 state 执行了多步推理，更不能证明它 faithful 地表达模型决策原因。

## 13. 它与相关工作的关系

| 方向 | 代表工作 | 计算发生在哪里 | FLAIR 的差异 |
| --- | --- | --- | --- |
| Full-duplex speech | Moshi、SALMONN-omni | 双流音频/文本时间轴 | 专门利用监听窗口形成 latent state |
| Explicit streaming thought | Chronological Thinking、SHANKS | 音频 chunk 间生成文本 reasoning | 不产生可见 CoT，更容易随 turn boundary 抢占 |
| Continuous thought | Coconut、CODI | 静态问题上的多步 hidden/soft state | latent step 与真实语音时间绑定 |
| Looped latent compute | Ouro、Looped Transformer | 重复层或自适应计算深度 | FLAIR 不额外循环，使用原本就存在的监听 frame |
| Soft token reasoning | Soft Thinking、SoftCoT | 词表分布的 embedding mixture | 机制相近，但目标是 full-duplex causal listening |

Coconut/CODI 更关注“是否可以不把推理写成句子”；Chronological Thinking/SHANKS 更关注“是否可以在听完之前开始推理”；FLAIR 把两者交叉起来：**在听完之前推理，但不写出句子。**

它与 Moshi 的关系也不是简单替代。Moshi 证明了单模型、流式 codec 和双 speaker stream 可以实现约数百毫秒级自然全双工；FLAIR 关注的是监听期内部计算是否能提升后续语义回复。音频生成、轮次控制、latent cognition 是三个相关但可分解的层次。

## 14. 论文真正证明了什么

基于现有实验，可以较有把握地说：

1. 在一个 Qwen2.5-7B 级全双工 speech model 中，把监听期输出递归为 soft vocabulary embedding 是可训练的，不会必然陷入全静音或无法切换状态。
2. non-causal Expert 加 KL distillation 能为没有人工标签的 latent state 提供有效训练信号。
3. 在固定 backbone 与数据配置下，latent 版本改善多个 QA/reasoning 指标，尤其是 LlamaQ 与 MMSU。
4. 该机制与实时双工交互可以共存；它没有引入一个必须在回复前串行完成的长 CoT 阶段。
5. 监听窗口可以被视为有价值的 test-time computation，而不只是等待区。

这些结论已经足够重要。它把实时语音系统的目标从“尽快检测用户说完”推进到“用户没说完时也持续形成可用状态”。

## 15. 论文没有证明什么

同样重要的是边界。

**没有证明 latent state 可解释。** 词表分布可以检查 top tokens，但 soft embedding 经过多层递归后不等于一句隐含自然语言。它可能编码声学、时序、答案倾向和模型偏差的混合。

**没有证明 latent reasoning faithful。** 最终回答正确，不代表 latent trajectory 是导致答案的真实原因。Expert 看过未来回答，student 可能学习一种压缩答案提示，而不是可组合的推理程序。

**没有固定计算预算比较。** latent 版本利用用户说话期间已有 frame，但仍进行了连续计算。若与“用户结束后获得等量额外 latent steps”的模型比较，质量来源可能是总计算量，而非边听边想这一特定安排。

**没有证明对真实对话分布普遍有效。** 620K 小时中的大部分由文本和 TTS 合成，turn length、中断位置及 reaction delay 都受到模板控制。真实世界的口音、半句话、笑声、共同补句、多人串话和自我修复更复杂。

**没有形成 agentic state。** 模型输出仍主要是语言与语音。工具执行进度、长期记忆版本、用户是否听到某句话、可撤销动作和风险边界，都不在显式建模范围内。

## 16. 从 latent thought 到连续时间 Agent Policy

我更愿意把 FLAIR 的 `Z_t` 称为 **streaming belief state candidate**，而不是“内部认知”的最终形态。它已经具备三个重要属性：随观测递归更新、严格因果、随时可被回答动作消费。但一个真正做事的交互式 Omni Agent 还需要更完整的状态分解。

![From latent thought to agent policy](/images/blog/silent-thought-agent-gap.svg "图 5：FLAIR 已覆盖从流式观测到语言回复的 latent bridge；产品级连续时间 Agent 还需显式管理不确定性、工具、记忆、播放进度与回滚。")

假设用户说：“把明天下午的会改到三点……等一下，是后天。”在听到“明天下午”时，模型可以提前查询 calendar availability；听到纠正后，需要取消旧 query 或让结果失效；如果助手已经播出“已经为您修改”，还需要区分“模型生成过”与“用户实际听到过”。这不是更自然的 TTS 能解决的，而是 policy state 问题。

一个产品级状态至少应包含：

- intent hypotheses 与置信度，而不是单一隐式答案倾向；
- slot values、证据时间与 supersession link；
- tool call 的 pending/succeeded/cancelled 状态；
- memory read/write 的版本和来源；
- assistant generated、buffered、played 三种不同进度；
- 动作是否可逆、是否需要确认、错误成本多大；
- 当前应 WAIT、BACKCHANNEL、CLARIFY、SPEAK、ACT 还是 ABORT。

因此，我仍然认为：**交互式 Omni Model 的本体是运行在连续时间上的 Agent Policy；自然接话、停顿、打断只是这个 policy 的可见动作。** FLAIR 的贡献，是证明监听期可以维护比 silence token 更丰富的递归状态；下一步需要让这种状态服务于决策、行动与修复，而不只服务于下一段回答。

## 17. 五个可证伪的 toy experiments

在投入更大训练前，可以先用小实验验证 latent state 是否真的具备所宣称的性质。

### 17.1 Prefix revision

构造最小对：

```text
A: “把会议改到周三下午三点。”
B: “把会议改到周三下午……不，周四三点。”
```

逐帧 probe 日期/时间分布。理想状态应在听到纠正后快速从 Wednesday 转向 Thursday，而不是保留平均化混合。测量 revision latency、旧值残留和最终工具参数正确率。

### 17.2 Contradiction injection

在问答中随机插入与早期前缀冲突的后续事实，比较 silence baseline、explicit streaming CoT 与 FLAIR。若 latent state 只是压缩早期答案倾向，它会在冲突后表现出更强 anchoring；若是有效 belief update，则应主动重分配概率。

### 17.3 Interruption rollback

让助手边说边启动 speculative tool call，用户在不同播放位置打断。分别记录生成文本、已合成音频、已播放音频和工具状态，测试模型能否只撤销用户未确认的动作，并避免在后续对话中引用用户从未听到的承诺。

### 17.4 Latent-state probing

冻结模型，在每个时间点用轻量 probe 预测 intent、slot、question completeness、uncertainty 和 turn readiness。重要的不是 probe 准确率本身，而是信息何时出现、后续是否可修正，以及 probe 方向是否对不同说话人和噪声保持稳定。

### 17.5 Shuffled posterior 与 fixed compute

把 Expert posterior 在同 batch 样本间打乱，或只保留相同 turn length 的 posterior，检验收益究竟来自内容相关监督还是平滑正则。同时给 baseline 在用户结束后分配等量 latent compute，区分“更多计算”与“边听边更新”的贡献。

这五个实验都比直接扩到更大参数更有诊断价值。只有当 revision、causality、content-specific posterior 和 fixed-compute 优势成立，才值得进入真实客服或会议 Agent 的大规模训练。

## 18. 产品视角：用户真正感受到的不是 latent，而是修复成本

普通用户不会关心系统是否使用 ELBO、soft token 或 Global-aware Expert。他们感受到的是：

- 我还没说完，系统是否过早回答；
- 我改口后，它是否真的更新了理解；
- 我打断时，它多久停止；
- 它是否记住了错误版本；
- 它是否未经确认执行了不可逆操作；
- 弱网、噪声和多人说话时，错误能否低成本修复。

所以产品评测不能只报告 average latency 和最终 QA accuracy。至少应同时测：从语义已经充分时刻 `t_ready` 到首个有用动作的延迟、premature action rate、revision recovery、barge-in stop latency、false interruption、用户重复信息次数、未播放内容引用率，以及 P50/P95/P99。

FLAIR 让系统有机会在用户说话期间提前积累信息，但“提前想”不等于“提前做”。查询与预取可以 speculative execution；支付、删除和发送必须 prepare-confirm-commit。latent state 若没有风险和可逆性约束，速度越快反而可能越危险。

## 19. 最终评价

《The Silent Thought》提出了一个足够简单、又足够关键的转变：不要把全双工模型的监听时间当作空白。通过递归 soft vocabulary embedding，模型可以在不生成显式 CoT 的前提下持续更新内部状态；通过 non-causal Expert 与 ELBO 风格蒸馏，这种无标签状态获得了可训练目标。

论文的实验支持“latent listening 有助于部分语义任务，且能与实时交互共存”，但不支持“全面提升”“零成本”或“已经获得可解释认知”。它更像一块架构积木：把 silence 变成 state，把 hindsight teacher 蒸馏进 causal stream。下一步真正困难的工作，是让 state 可修正、可探测、可连接工具与记忆，并用用户修复成本而不是拟人化演示衡量它。

我认为这篇论文最重要的启示可以写成一句话：

> **全双工的核心不是同时听和说，而是在任何时刻都维护一个足以支持下一次正确决策、并能被新证据推翻的状态。**

## 参考资料

- [The Silent Thought: Modeling Internal Cognition in Full-Duplex Spoken Dialogue Models via Latent Reasoning](https://arxiv.org/abs/2603.17837)
- [Moshi: A Speech-Text Foundation Model for Real-Time Dialogue](https://arxiv.org/abs/2410.00037)
- [SALMONN-omni: A Standalone Speech LLM without Codec Injection for Full-duplex Conversation](https://arxiv.org/abs/2505.17060)
- [Training Large Language Models to Reason in a Continuous Latent Space (Coconut)](https://arxiv.org/abs/2412.06769)
- [CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation](https://arxiv.org/abs/2502.21074)
- [SHANKS: Simultaneous Hearing and Thinking for Spoken Language Models](https://arxiv.org/abs/2510.06917)
- [Chronological Thinking in Full-Duplex Spoken Dialogue Language Models](https://arxiv.org/search/?query=Chronological+Thinking+in+Full-Duplex+Spoken+Dialogue+Language+Models&searchtype=all)
- [Full-Duplex-Bench: A Benchmark for Full-Duplex Spoken Dialogue Models](https://arxiv.org/search/?query=Full-Duplex-Bench&searchtype=all)

本文的论文事实与数值以 arXiv v5（更新于 2026-06-04）为准；结构图为依据论文机制重新绘制。跨模型结果因 backbone、数据和训练设置不同，仅作为上下文，方法归因以 FLAIR 同架构消融为主。

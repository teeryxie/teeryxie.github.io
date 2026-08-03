# Learning When to Think While Listening：大音频语言模型如何学习“何时思考”

实时语音模型面临一个无法回避的决策：用户还在说话时，模型应该继续等，还是已经可以开始整理答案？如果永远等到 endpoint 才推理，回答可能更可靠，却把全部计算暴露为用户可感知的停顿；如果每 0.5 秒都思考一次，模型会被不完整语义反复误导，也会浪费算力；如果太早提交答案，又可能在决定性证据到来前犯错。

Zhiyuan Song、Weici Zhao、Yang Xiao、Suhao Yu、Cheng Zhu 与 Jiatao Gu 的《[Learning When to Think While Listening in Large Audio-Language Models](https://arxiv.org/abs/2605.27190)》把这个问题写成一个显式的在线控制任务：在每个音频决策点，Qwen2.5-Omni-7B 从 `<wait/>` 与 `<think>...</think>` 中选择；语音结束后再生成一次 final think 和 `<answer>...</answer>`。作者先用 75,723 条对齐的音频-文本轨迹做 SFT，再用 DAPO 和六类轨迹奖励优化准确率、动作合法性、更新时间、残余延迟、局部 thought quality 与整条 reasoning chain consistency。

这篇论文与上一篇 [The Silent Thought / FLAIR](https://teeryxie.github.io/blog/silent-thought-flair-latent-reasoning/) 恰好构成一组互补问题：FLAIR 问“监听时的内部计算用什么表示”，本文问“什么时候值得显式更新状态”。但精读后也必须明确：论文还没有学习真正的 endpoint-free ANSWER timing，主延迟指标也不是毫秒，而是用户说完后 final-think 的 token 长度。

![Wait-think-answer overview](/images/blog/when-to-think-controller.svg "图 1：wait-think-answer 控制器在 partial audio 上周期性决策。实验中 endpoint 前只允许 WAIT 或 THINK，ANSWER 固定发生在完整语音之后。")

## 1. 一页结论

- **研究对象是 reasoning placement。** 它不只是让模型“多想”，而是把一定量推理从 endpoint 后移动到用户说话期间。
- **动作语言清晰。** `WAIT` 不改变可见 memory，`THINK` 追加短语义状态，后续决策同时读取完整音频前缀与此前 thought。
- **SFT 只教会协议，不保证任务能力。** Synthetic SRQA 平均准确率从 base 67.6% 降到 SFT 66.1%；DAPO 六奖励版本才提升到 70.3%。
- **六项 reward 是方法核心。** 只奖励答案会退化成 wait-all；只奖励 latency 会产生空 thought、过早动作或格式错误。
- **合成集结果有说服力但有限。** 8,959 条、六任务上，六奖励版本同时把准确率提高 2.7 个百分点，并把 final-think 从 10.44 降到 8.99 token，约减少 14%。
- **真人录音只足以证明系统没有立即失效。** Real Audio Bench 只有 5 位说话人、186 条录音；各控制器 95% bootstrap CI 大量重叠，不能据此精细排序。
- **它没有直接证明真实响应更快。** final-think token 只是 residual reasoning proxy；full-prefix replay 的平均 RTF 为 1.134，且 repeated prefill 本身很昂贵。
- **答案时机尚未真正学习。** endpoint 前合法动作只有 WAIT/THINK，ANSWER 被固定在 utterance 完成之后。因此论文学习的是“何时形成中间 state”，不是完整的 WAIT/THINK/ANSWER optimal stopping。

我的总体评价是：**这是一篇把“边听边想”从生成技巧推进到轨迹优化的扎实工作，但它解决的是 reasoning scheduling 的第一阶段，不是生产级实时 Agent 的完整控制问题。**

## 2. 从“能否边听边想”到“何时值得想”

许多 streaming speech work 已经证明，可以在用户说话期间执行计算。真正困难的是计算分配：哪些前缀已经带来 answer-relevant state change，哪些只是语气词、背景铺垫或尚未完成的句法结构？

假设用户说：

```text
“小王有 12 本书，送给同学 4 本，
后来又买了原来剩余数量的一半。现在有多少本？”
```

合理轨迹不是每个 chunk 都总结，也不是一直等到最后：

```text
0.5s  WAIT   只有人物，没有可计算状态
1.5s  THINK  initial = 12
3.0s  THINK  remaining = 12 - 4 = 8
4.5s  WAIT   “后来又买了……”信息尚未完成
5.5s  THINK  added = 8 / 2 = 4; total = 12
END   THINK  answer cue = 12
       ANSWER 12
```

这背后是一个 value-of-computation 问题。一次 THINK 的收益是减少 endpoint 后残余计算并固化关键状态；成本是生成 token、阻塞下一次决策，以及对不完整信息过早承诺。理想 controller 应估计：

```text
V(THINK | current prefix, memory)
  - V(WAIT | current prefix, memory)
  - compute cost
  - revision risk
```

论文没有显式学习这个 value function，而是通过动作轨迹和 reward 间接塑造 policy。这使问题可训练，但也让效果高度依赖 thought anchor 与 reward label 的质量。

## 3. 控制器到底观察什么

令完整音频为 `x_1:T`，决策时刻为 `t_k`，已经提交的可见 thought memory 为 `z_<k`。第 `k` 次决策的 observation 是：

```text
o_k = (x_1:t_k, z_<k)
a_k = argmax_a πθ(a | o_k)
```

注意它读取的是 **full prefix**，不是只有最新 0.5 秒 chunk。每次决策都重新看到从开头到当前时刻的全部音频，再加上此前写出的 thought。这减少了状态丢失，却带来 repeated prefix prefill。

endpoint 前的 action set 为：

```text
<wait/>
<think>short semantic update</think>
```

endpoint 后，协议固定要求：

```text
<think>final compact state</think>
<answer>final answer</answer>
```

`WAIT` 只推进音频，不写入文本 memory；pre-endpoint `THINK` 则把短 semantic state 永久追加到后续 prompt。旧 thought 不会自动删除，因此“短、具体、只在证据变化时写入”非常重要。否则 context 会积累噪声，早期错误也会形成 anchoring。

这同时暴露了论文标题与实验协议之间的一条边界：作者称其为 wait-think-answer formulation，但实验中 **answer timing 不是 endpoint 前可学习的 stopping decision**。模型不能听到足够信息就提前回答，也不能自主决定用户长停顿是否已经结束；它只学习 endpoint 前 WAIT/THINK 的分配和 endpoint 后的固定 THINK/ANSWER 合约。

## 4. 数据轨迹是怎样构造的

初始候选库包含 80,000 条：40,000 条可验证问题和 40,000 条开放问题。GPT-4o 为每条记录生成 spoken-friendly surface form、TTS style instruction、answer-relevant lexical anchors、semantic wait-think-answer trace 与 final answer。

经过验证、去重和人工抽查后，剩余 75,723 条：

| 分支 | 数量 | 用途 |
| --- | ---: | --- |
| Verifiable | 38,213 | SFT 与可确定评分的 RL 数据来源 |
| Open-ended | 37,510 | SFT 的协议与语义表达训练 |
| Train | 73,675 | SFT 训练 |
| Validation | 2,048 | 模型和 reward 配置选择 |
| DAPO scorable train | 37,180 | policy optimization |

DAPO mix 还加入少量 ARC-Challenge、ARC-Easy、GSM8K、PIQA 和 SocialIQA 的 training-split 样本，让 reward 接触 benchmark-style reasoning format；占比低于 2%，不包含 held-out evaluation examples。这不构成直接测试泄漏，但意味着 benchmark 家族与格式并非完全 out-of-domain。

所有 spoken input 用 Qwen3-TTS 一次性合成为完整 utterance，而不是按 action segment 分段合成。这个决定是正确的：如果每个 thought boundary 都重新 TTS，会在人为标注点制造异常停顿，controller 很容易通过声学接缝“作弊”。

随后用 CTC-style forced alignment 将 transcript word 对齐到时间戳，再把 lexical anchor 对应的 controller boundary **向上吸附到 0.5 秒 decision grid**。因此 timing label 的根源是：Teacher 先指出哪些词改变答案状态，再由 forced alignment 把这些词定位到音频时间轴。

![Trace construction pipeline](/images/blog/when-to-think-data-pipeline.svg "图 2：从语义问题到完整 TTS 音频、forced alignment、0.5 秒网格和 WAIT/THINK 监督轨迹。完整 utterance 合成避免在动作边界制造声学泄漏。")

这个构造很实用，但要区分两件事：模型确实只接收 audio，不接收 transcript；然而 update timing reward 使用由文本 lexical anchor 和对齐器产生的“何时应想”标签。它学习的是 Teacher 定义的 answer-relevant update，而不是从真实人类互动中直接发现自然认知节奏。

## 5. 为什么先 SFT，再做 DAPO

SFT 使用 Qwen2.5-Omni-7B、MS-Swift 与 LoRA：rank 8、alpha 32、dropout 0.05，目标是 all linear layers；audio encoder 和 aligner 冻结，language model 可训练。训练一轮，学习率 `1e-5`，最大长度 8192，4 张 B200，有效 batch size 32。

SFT 主要教四件事：

1. 三种 action tag 的合法序列；
2. 普通 WAIT 与 answer-relevant THINK 的区别；
3. thought 应写成短 semantic state，而不是泛泛总结；
4. endpoint 后 final think 与 final answer 的格式。

但 imitation learning 会继承 Teacher trace 的局限，也不直接优化任务答案。主表中 SFT 的合成准确率反而从 base 67.6% 降到 66.1%，说明“学会协议”可能暂时损害 base model 的任务能力。它更适合看作 RL 的 cold start，而非最终结果。

DAPO 从 SFT LoRA 初始化，使用自定义 streaming controller trainer。每个 prompt 采样 8 条 rollout，计算 group-relative advantage，再对 controller、thought 与 answer token 做 asymmetric clipped policy update。设置包括：

```text
steps = 1000
warmup = 50
actor learning rate = 4e-7
LoRA rank = 8, alpha = 32
KL coefficient = 0.01
clip low / high = 0.20 / 0.28
think and answer cap = 48 tokens
hardware = 4–5 × NVIDIA B200
```

训练组必须至少包含足够的 format-valid rollout、一个合法 final-think 和一个合法 pre-endpoint thought；若整组退化为 all-wait 或格式错误，则重采样，持续失败的样本会跳过并记录。这种 dynamic sampling 防止零方差 group 无法提供学习信号，但也会改变有效训练分布：最难形成合法轨迹的 prompt 被相对下采样。

## 6. 六项 reward 为什么缺一不可

最终 reward 由四个 rule-based 项和两个 judge-assisted 项组成。

| 项 | 目标 | 主要防止的退化 |
| --- | --- | --- |
| `R_a` Answer | 最终答案正确 | 流畅但错误的答案 |
| `R_f` Format | WAIT/THINK/ANSWER 合约合法 | 提前回答、标签错误、缺 final think |
| `R_s` Sync/Latency | endpoint 后 final think 紧凑 | 把所有推理拖到用户说完后 |
| `R_u` Update timing | thought 靠近关键证据变化 | 漏掉关键更新或每 tick 都想 |
| `R_t` Thought quality | thought 短、具体、支持答案 | 冗长、泛化、meta commentary |
| `R_c` Chain consistency | 正确答案的 thought chain 连贯 | 前后矛盾、无依据跳到答案 |

合法轨迹的 shaped reward 可简化为：

```text
R_valid = λa Ra + λf Rf + λs Rs + λu Ru + λt Rt
          + 1[Ra > 0] λc Ra Rc

λa=1, λf=1, λs=1, λu=3, λt=1, λc=0.45
```

protocol gate 优先：若 `R_f <= 0`，最终 reward 只保留 format penalty，正确答案也无法救回非法轨迹。Consistency bonus 仅在答案正确时开启，避免“自洽地答错”获得额外奖励。

`R_s` 的实现也很具体：final-think 有 6-token free budget，之后线性惩罚系数 0.30，上限 3.0；3–6 token 的紧凑 answer cue 可获小额 bonus。`R_u` 允许 thought 与目标 update tick 相差两个 0.5 秒 tick，并加入 sparsity pressure。

`R_t` 和 `R_c` 由本地 Qwen3.6-35B-A3B judge 评分，输出映射为 `{0, 0.5, 1}`。所以六奖励版本的优势包含 judge preference；它不只是纯规则 RL。Judge 的偏好能改善短 thought，也可能把特定语言风格注入 policy。

![Six-reward trajectory design](/images/blog/when-to-think-rewards.svg "图 3：六项轨迹奖励分别约束答案、协议、残余延迟、更新时间、局部 thought quality 和全链一致性。只优化单一目标会产生 wait-all 或 think-everywhere。")

## 7. Synthetic SRQA：主结果如何读

评测包含 8,959 条 TTS spoken reasoning question，覆盖 ARC-Easy、ARC-Challenge、SocialIQA、PIQA、GSM8K 与 300 条 LLaMA-QS。核心比较应限制在同一 Qwen streaming harness 内。

| Controller | ARC-E | ARC-C | SIQA | PIQA | GSM8K | LLaMA-QS | Avg. | Final think |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 87.8 | 80.8 | 68.6 | 63.5 | 22.8 | 71.0 | 67.6 | 10.44 |
| SFT | 86.3 | 78.1 | 68.6 | 60.5 | 21.6 | 71.7 | 66.1 | 9.82 |
| DAPO 4 rewards | 88.9 | 81.7 | 68.4 | 65.1 | 24.6 | 70.3 | 68.5 | 10.87 |
| DAPO 5 rewards | 89.1 | 81.7 | 69.6 | 66.4 | 24.9 | 71.0 | 69.2 | 10.94 |
| DAPO 6 rewards | 89.6 | 81.7 | 71.0 | 69.2 | 25.9 | 71.0 | **70.3** | **8.99** |

六奖励版本相对 base：平均准确率 `+2.7`，final-think `-1.45 token`，即约 14% 缩短。它在 ARC-E、ARC-C、SIQA、PIQA、GSM8K 上提高，LLaMA-QS 持平。

消融的形状很有信息量：4/5 reward 虽然提升 accuracy，却让 final-think 从 10.44 增加到 10.87/10.94。只有加入 chain consistency 的六奖励版本同时把 accuracy 推到 70.3 并将 final-think 降到 8.99。这可能说明一致的中间 state 让 endpoint 后无需重做推理；也可能是 reward interaction 和 checkpoint selection 的结果。论文没有多随机种子误差，因此不能仅凭单次 ablation 断言 `R_c` 具有独立因果贡献。

完整音频 Qwen2.5-Omni-7B 的平均准确率为 70.8、final-think 10.27。六奖励 streaming controller 达到 70.3，接近 complete-audio upper reference；这比跨模型对比更值得关注。

## 8. Real Audio Bench：迁移成立，排序不成立

作者让 5 位 speaker 各录制 40 条 GPT-4o 生成候选，共 200 条；人工筛除含糊或不可答问题并修正 answer key，最终保留 186 条，还记录了 23 处答案修正。

| Controller | Accuracy | 95% bootstrap CI | Final think |
| --- | ---: | --- | ---: |
| Base | 64.0 | [57.0, 71.0] | 6.52 |
| SFT | **68.8** | [61.8, 75.3] | 6.64 |
| DAPO 4 | 65.6 | [58.6, 72.6] | 7.74 |
| DAPO 5 | 67.7 | [60.8, 74.2] | 7.39 |
| DAPO 6 | 65.1 | [58.1, 72.0] | **6.33** |

SFT 的准确率最高，六奖励 DAPO 的 final-think 最短，而且是唯一低于 base 的 learned controller。但所有 95% CI 大量重叠。样本规模只有 186，speaker 只有 5 位，最可靠结论是：由 TTS 轨迹训练出的 action protocol 在真人录音上仍能运行，并呈现 accuracy-latency operating points；不能声称 DAPO 6 在真实语音上总体优于 SFT。

更值得注意的是 synthetic 与 real 的最优点不同。合成集上 DAPO 6 同时最好；真人录音上 SFT accuracy 最好、DAPO 6 residual 最短。这提示 timing label 与 TTS prosody 之间仍有 domain gap。

![Accuracy and residual reasoning results](/images/blog/when-to-think-results.svg "图 4：合成 SRQA 与 Real Audio Bench 上的 accuracy–final-think trade-off。真人录音置信区间重叠，最优准确率与最短残余推理来自不同模型。")

## 9. “延迟降低 14%”究竟意味着什么

论文主 latency metric 是 **post-endpoint final-think token count**。它回答的是：用户说完后，模型还需要输出多少 reasoning token 才进入 answer。这个指标适合比较同一 Qwen tokenizer、同一 harness 下 reasoning placement，但不是端到端响应毫秒数。

实际 wall-clock 还包括：

- 每 0.5 秒触发一次 controller；
- 重新 replay 从开头到当前的全部音频前缀；
- 生成 WAIT 或 THINK；
- judge/reward 只存在于训练，不在推理；
- endpoint 后生成 final think、answer 与语音。

论文附录报告 full-prefix replay harness 的平均 RTF：

| Lane | Mean RTF |
| --- | ---: |
| Base | 1.137 |
| SFT | 1.353 |
| DAPO 6 | 1.134 |

RTF 1.134 表示控制器 wall-clock 约为 source audio 时长的 1.134 倍。在当前 replay 实现中，它并非一个已经具备充足实时余量的生产服务。DAPO 6 接近 base 且优于 SFT，说明 policy 没让 runtime 更糟，但 final-think 减少不能自动等价为首包语音延迟减少 14%。

作者做了 cache-native prototype：在 persistent Qwen2.5-Omni Thinker KV cache 中追加 audio chunk，controller decision 从临时 fork 生成；WAIT 不写回 cache，THINK 才 commit。4 秒真实音频的 wiring smoke test 表明 2.0s 与 0.5s chunking 下 cache/attention length 对齐，但尚未形成可评分的 batch benchmark。

另一个细节是，decision grid 虽为 0.5 秒，Qwen2.5-Omni 仍要求最少 2.0 秒 audio window，因此 0.5 秒 tick 提高的是 reconsideration cadence，不代表模型从第一个 0.5 秒就拥有完全独立的低延迟声学处理路径。

## 10. 与 FLAIR、SHANKS 和 Question Completeness 的区别

| 方法 | 监听期 state | “何时想” | 可见性 | 主要目标 |
| --- | --- | --- | --- | --- |
| Question Completeness | 完整度分数 | 达阈值后切换 | 可解释标量 | 判断问题是否说完/信息是否足够 |
| SHANKS | chunk 间显式 reasoning | 固定或策略化 chunk | 可见文本 | 提前规划/API 调用 |
| FLAIR | recurrent soft embedding | 每个 listening frame 更新 | 不可见 latent | 利用 silence frame 改善回复 |
| 本文 | WAIT 或短显式 THINK | 0.5 秒网格上的 learned action | 可见文本 memory | 优化 reasoning placement |

FLAIR 更连续、更容易抢占，但 state 难以审计；本文的 thought 可读、可进入后续 context，却会形成离散 commitment 和 revision 负担。二者并不互斥：一个更完整的系统可以高频更新 latent belief，低频输出 compact checkpoint，真正需要行动时再产生 structured action。

Question Completeness 只估计“问题是否已经足够完整”，而本文还决定“虽然不能回答，是否值得更新中间 state”。这是重要差别：一个长问题可能在 30% 处就提供第一个可计算事实，但仍要等到 100% 才能回答。

## 11. 论文真正证明了什么

1. Qwen2.5-Omni 可以被训练成遵守显式 WAIT/THINK/final-THINK/ANSWER action protocol。
2. semantic anchor、forced alignment 与 0.5 秒网格可以构造大规模 audio-grounded controller trace。
3. SFT 适合作为协议 cold start，但单独使用会牺牲合成任务准确率。
4. trajectory-level DAPO 可以把任务能力拉回并超过 base；六奖励配置在合成集上同时改善 accuracy 与 final-think length。
5. controller family 能迁移到少量真人录音，不完全依赖 TTS waveform。
6. full-prefix 信息流原则上可改写为 cache-native controller，作者已有小规模 wiring prototype。

## 12. 论文没有证明什么

**没有证明完整的 ANSWER timing policy。** endpoint 是外部已知边界，模型不能在 endpoint 前 answer。真正的自然对话还需要在停顿、补充和 barge-in 中判断何时开口。

**没有证明端到端毫秒延迟降低 14%。** 14% 指 final-think token 数；replay RTF、TTS 首包和网络调度是另一组成本。

**没有证明 thought 是 faithful cognition。** 可见 thought 可能只是 reward-friendly checkpoint；Qwen judge 偏好的“短、具体、自洽”不保证它是答案的真实因果过程。

**没有验证在线 revision。** thought 一旦写入 memory 就持续存在。用户后续纠正、否定或改变目标时，模型是否会显式 supersede 旧 state，论文没有专门评测。

**没有覆盖真实交互行为。** SRQA 是单 utterance reasoning QA，不包含 backchannel、重叠、用户打断助手、工具事件或多轮 memory。

**真人评测统计功效有限。** 5 speaker、186 item 的 CI 无法区分多数 controller 差异，且问题仍由 GPT-4o 生成，不是自然发生的开放域对话。

**reward attribution 仍不充分。** 4/5/6 reward 是逐步堆叠而非全因子消融，没有多 seed，也没有 shuffled timing anchor 或 judge replacement 对照。

## 13. 从第一性原理设计下一组 toy experiments

### 13.1 Earliest sufficient evidence

为每条问题人工标记最早可确定答案时刻 `t_ready`，而不只标记 lexical update tick。评估 THINK 是否在有价值信息后出现、ANSWER 是否接近 `t_ready`，并区分“用户还没说完”与“答案已充分确定”。

### 13.2 Prefix contradiction and supersession

构造：

```text
“会议在周三……抱歉，是周四下午三点。”
```

要求 controller 输出 `date=Wednesday` 后，必须产生明确 supersession，而不是简单追加 `Thursday`。测旧 state 残留、最终答案与工具参数污染。

### 13.3 Shuffled anchors

将 update tick 随机平移或跨样本交换。如果模型结果几乎不变，说明 `R_u` 可能主要充当 sparsity regularizer；如果明显下降，才说明 answer-relevant timing supervision 真正在起作用。

### 13.4 Fixed-compute comparison

给 wait-all baseline 在 endpoint 后相同总 token budget，给 periodic-think baseline 与 learned controller 相同调用次数。比较 accuracy、总 FLOPs、首答案时间和 energy，分离“更多计算”“更早计算”与“更好调度”。

### 13.5 Cache-native end-to-end latency

把 prototype 完成到可评分 runtime，报告从 audio packet 到 controller decision、从 endpoint 到 text first token、到 speech first packet 的 P50/P95/P99，并加入并发压力。只有这样才能把 residual token claim 转化为用户体验 claim。

### 13.6 Think-to-action transfer

将最终任务从 QA 改为可撤销的工具调用：查询天气、搜索订单、读取 calendar。测试 pre-endpoint thought 是否真的让工具更早启动，以及用户纠正后能否取消 speculative call。这样才能验证 thought 是否具有 agent utility，而不只是提高 benchmark answer。

![From reasoning scheduler to agent policy](/images/blog/when-to-think-agent-policy.svg "图 5：本文学习 reasoning placement；完整实时 Agent 还需学习 endpoint、ANSWER、工具调用、撤销和播放感知，并用可逆性与风险约束动作。")

## 14. 对交互式 Omni Agent 的启示

从产品角度，WAIT 与 THINK 都不是最终用户价值，真正目标是用最低 interaction cost 做出正确行动。一次内部 thought 是否值得执行，应取决于：

- 新音频带来的 information gain；
- 当前 belief uncertainty；
- 下一动作是否可逆；
- tool prefetch 能节省多少时间；
- 错误 commitment 的 repair cost；
- 用户是否正在补充、犹豫或纠正；
- 当前计算是否会阻塞监听或语音输出。

因此更完整的动作空间应是：

```text
WAIT
LATENT_UPDATE
VISIBLE_CHECKPOINT
BACKCHANNEL
CLARIFY
ANSWER
CALL_TOOL / CANCEL_TOOL
PREPARE / COMMIT / ABORT
MEMORY_WRITE / SUPERSEDE
```

高频 latent update 可以持续吸收证据；低频 visible checkpoint 只在状态发生稳定变化时出现；tool action 则根据风险决定能否 speculative execution。`ANSWER` 不应机械绑定 endpoint，而应成为 optimal stopping action。

这也回到我对 full-duplex omni model 的判断：**它不是一个会自然抢话的 Speech-to-Speech Model，而是一个在连续时间中分配感知、推理、表达和行动预算的 Agent Policy。** 这篇论文的重要性，在于把“何时思考”第一次明确变成可训练的 policy trajectory；它的下一步，是把 reasoning scheduler 接到真正的 action scheduler。

## 15. 最终评价

《Learning When to Think While Listening》没有用更大的模型掩盖时序问题，而是设计了明确的 action language、数据轨迹和 reward stack。六奖励 DAPO 在同一 Qwen harness 下从 67.6% 提升到 70.3%，同时把 endpoint 后 final-think 从 10.44 压到 8.99 token，这个结果说明 reasoning placement 可以通过轨迹优化学习，而不必固定为“听完再想”或“每段都想”。

但这项结论应被准确命名：它证明的是 **partial-audio reasoning checkpoint scheduling**，不是完整的实时对话控制。endpoint、ANSWER、barge-in、工具与 memory 仍在问题之外；final-think token 也不能替代真实毫秒延迟。

如果说 FLAIR 让模型在监听时拥有一个持续演化但不可见的 state，那么本文让模型学习何时把 state 写成一个可见 checkpoint。二者组合后的研究问题会更有价值：

> **模型能否持续维护可修正的 latent belief，只在信息增益超过计算与承诺成本时形成显式 checkpoint，并在证据足够时自主选择回答或行动？**

这才是“Learning When to Think”走向 agentic omni interaction 的完整版本。

## 参考资料

- [Learning When to Think While Listening in Large Audio-Language Models](https://arxiv.org/abs/2605.27190)
- [Official code repository](https://github.com/realAllenSong/Learning-When-to-Think-While-Listening)
- [Real Audio Bench](https://huggingface.co/datasets/Oulasong/Real_Audio_Bench)
- [SRQA Audio](https://huggingface.co/datasets/Oulasong/SRQA_Audio)
- [The Silent Thought / FLAIR](https://arxiv.org/abs/2603.17837)
- [Can Speech LLMs Think While Listening?](https://arxiv.org/abs/2510.07497)
- [SHANKS: Simultaneous Hearing and Thinking for Spoken Language Models](https://arxiv.org/abs/2510.06917)
- [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215)

本文基于 arXiv v1（2026-05-26）、论文附录与官方代码仓库交叉核对。文中“延迟”始终区分 residual final-think token、replay-harness RTF 与真实端到端响应时间；跨模型数字不用于方法归因。

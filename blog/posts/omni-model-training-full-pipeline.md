# Omni 模型训练全流程：算力、数据配比、阶段配方与从 SFT 到 Agentic RL

“训练一个 Omni 模型需要多少张卡？”这个问题没有脱离训练目标的固定答案。同一个 7B 模型，做 LoRA 行为微调、冻结 codec 的双工 Mid-training、全参数多模态继续预训练，以及带 reference model 和实时音频 rollout 的 RL，所需显存和吞吐可能相差一个数量级。数据配比也不是“文本 30%、音频 40%”这样一次写死的采购清单，而是一个由能力退化、交互失败和 loss 贡献共同驱动的控制变量。

本文给出一条面向研究团队、可以逐级执行的训练路线：从 codec 与数据 contract 开始，经过文本能力保留、模态对齐、单流音频、双流连续时间建模、Interaction SFT、Talker 稳定化、控制策略 RL，最后进入带工具和环境反馈的 Agentic RL。重点不是复述 Omni 架构，而是回答五个工程问题：**每一阶段到底训练什么、冻结什么、需要多少卡、数据如何混合、满足什么条件才允许进入下一阶段。**

文中数据分成三类：标注为“公开值”的数字来自论文或官方仓库；显存和存储公式是可复核的工程推导；数据百分比与 GPU 档位是本文建议的启动配置，不代表任何论文的秘密配方。资料核对截止到 2026 年 8 月 14 日。

![Omni training stage map](/images/blog/omni-training-stage-map.svg "图 1：Omni 训练不是一次端到端联合优化，而是从表示、对齐、连续时间交互到 Agentic policy 的逐阶段晋级过程。")

## 1. 一页结论

- **先定义训练任务，再谈卡数。** LoRA、partial tuning、全参数 CPT 和在线 RL 的资源模型完全不同；“7B 能不能用 8 卡训练”不是完整问题。
- **默认不要从零预训练。** 对多数研究团队，合理起点是强文本或现有 Omni checkpoint；从零复现多万亿文本 token 和百万小时音频不属于普通论文预算。
- **Codec 先验证后冻结。** 如果重建质量、因果性、chunk 边界和实时系数没有过关，后续模型只会学习补偿一个不稳定表示层。
- **Thinker 与 Talker 分开控制漂移。** 先 Thinker、后 Talker，或像 DuplexOmni 一样交替冻结；不要让内容、韵律、控制时机和工具策略同时改变。
- **保留文本 replay。** Moshi 在单流音频预训练中有 50% 的 batch 继续使用纯文本，并使用独立 optimizer state；只靠“从文本模型热启动”不足以长期保留文本能力。[1]
- **双工能力来自时间结构，不来自音频总量。** 单流百万小时音频可以训练听觉与语音先验，但 overlap、backchannel、纠正、停止和恢复必须由双轨时间线或可验证的模拟轨迹提供。
- **数据配比要按有效 token 与梯度贡献定义。** 文件条数、原始小时和训练中的 loss contribution 不是同一个量。
- **先 SFT 学动作，再 RL 学取舍。** WAIT、SPEAK、YIELD、STOP、REVISE、CALL、CANCEL 等基本动作应先通过 Mid-training/SFT 成形；RL 再优化时机、延迟和长程任务收益。
- **第一版 RL 冻结 codec 与 Talker。** 先优化 control/action policy，避免 reward 噪声同时破坏语音质量和内容能力。
- **晋级由失败指标决定，而不是由训练步数决定。** 每个阶段都要有 holdout、回归集、延迟预算和停止条件。

## 2. 先定义：你到底在训练什么

至少要区分四类任务。

| 训练任务 | 典型可训练参数 | 主要目的 | 资源特征 |
| --- | --- | --- | --- |
| LoRA / adapter 微调 | 少量线性层、control head | 领域、角色、交互动作 | 权重状态小，activation 仍受长音频影响 |
| 双工 Mid-training | projector、LLM 部分层、control、Talker 部分模块 | 连续听说、重叠、打断、流式状态 | 长序列与多流 activation 是主要压力 |
| 全参数 Omni CPT | LLM、aligner、部分 encoder/Talker | 获得新的模态与基础能力 | 参数、梯度、optimizer、activation 全部昂贵 |
| RL / Agentic RL | policy、reference、rollout、reward | 优化时机、工具和长程任务成功 | 训练与推理资源并存，吞吐常由 rollout 限制 |

“Omni”本身也不是一个单一模块。一个可交互模型通常包含：

```text
audio / vision encoder
        ↓
projector / aligner
        ↓
Thinker or multimodal backbone
        ↓
text head + control/action head
        ↓
Talker / codec predictor
        ↓
codec decoder / vocoder
```

如果只训练 Thinker 的 LoRA，框架即使写着“支持 Qwen Omni”，也不等于支持 Talker、MTP、多 codebook loss、流式 mask 和双工 rollout。训练前必须做一次参数审计：哪些参数 `requires_grad=true`，哪些 loss 真正反传，哪些 cache 路径只在推理时存在。

## 3. 为什么不应该一开始全部端到端打开

全开联合训练看起来最“原生”，但它把四类漂移叠在一起：

```text
语义漂移：Thinker 开始忘记文本知识或指令能力
表示漂移：encoder / projector 的尺度改变
声学漂移：Talker 和 codec 的音质、音色、停顿改变
策略漂移：模型开始过早说话、过度打断或滥用工具
```

此时总 loss 下降几乎没有诊断意义。内容变差可能被更容易预测的 silence token 掩盖；Talker 的 acoustic loss 下降可能伴随 WER 上升；reward 上升可能只是模型学会更短、更保守地回答。

更稳妥的原则是：

> **每个阶段只引入一种新的自由度，并保留上一阶段的回归门。**

Qwen2.5-Omni 把 Thinker 与 Talker 分开，Talker 接收 Thinker 的高层表示；其公开后训练将 Talker 依次做 speech/context continuation、基于 WER 与 punctuation pause error 排序的 DPO，以及 speaker fine-tuning。[2] DuplexOmni 则在每一阶段交替优化 Thinker 和 Talker：训练一方时冻结另一方，数据比例 1:1。[3] 两条路线共同说明，**模块统一不等于优化变量必须同时开放。**

## 4. 用显存公式建立下界

BF16 权重的最简单下界是：

```text
inference weights ≈ 2 bytes × parameter count
```

因此 7B 权重约 14 GB，9B 约 18 GB，14B 约 28 GB。全参数 AdamW 训练的常用粗算是：

```text
BF16 parameters        2 bytes
BF16 gradients         2 bytes
FP32 master weights    4 bytes
FP32 first moment      4 bytes
FP32 second moment     4 bytes
--------------------------------
model states          16 bytes / trainable parameter
```

于是仅模型状态的理论量级为：

| 可训练参数 | 未分片 AdamW 状态下界 | 8 路完全分片后的理想下界 |
| --- | ---: | ---: |
| 7B | 112 GB | 14 GB / GPU |
| 9B | 144 GB | 18 GB / GPU |
| 14B | 224 GB | 28 GB / GPU |

这张表**没有**计算 activation、临时 buffer、通信 bucket、视觉和音频 encoder、Talker、MTP、KV/cache、长序列 attention workspace，也没有计算碎片。8 卡下的 14 GB 绝不意味着 7B 全参数 Omni 可以舒适地运行在 8 张 24 GB 卡上。

DeepSpeed ZeRO-1/2/3 分别分片 optimizer state、gradient 和 parameter；FSDP 也通过分片参数与训练状态降低数据并行复制。[4][5] 它们解决的是状态冗余，不会让长音频和长视频的 activation 自动消失。Activation checkpointing 用额外重计算换显存；tensor、pipeline、context 和 sequence parallel 则分别切模型维度、层、长序列与中间表示。选择并行策略必须从 profile 出发，而不是机械地把所有开关同时打开。

## 5. 四档 GPU 配置：分别能做什么

![GPU budget ladder](/images/blog/omni-training-gpu-budget.svg "图 2：卡数不是模型规模的唯一函数。随着训练目标从 LoRA 走向全参数 CPT 与 RL，资源从单一 trainer 扩展为数据、训练、rollout 和评测多个池。")

NVIDIA 官方规格中，H100 SXM 为 80 GB，H200 为 141 GB，A100 也有 80 GB 版本。[6][7][8] 下面是本文建议，不是论文最低配置。

### 5.1 学习与最小实验：1×80 GB

推荐起点：`1×H100 80GB` 或 `1×A100 80GB`；有条件时 H200 141GB 会显著增加长序列余量。

适合：

- Moshi 7B LoRA；
- codec 编解码、双声道数据管线和 cache 一致性测试；
- 10–100 小时数据 smoke test；
- 冻结主体后的 WAIT/SPEAK/STOP 控制 token SFT；
- 小模型或文本化轨迹上的 reward 消融。

Moshi 官方微调仓库给出的配置使用 LoRA rank 128、100 秒 duration、batch size 16；官方实测 `1×H100` 峰值显存 39.6 GB，`8×H100` 每卡峰值 23.7 GB。[9] 这是 LoRA 微调实测，不是 Moshi 基础预训练成本。仓库还明确提醒：OOM 时可先减 batch，再减 duration；过度缩短训练 duration 可能让模型推理时更早沉默。

24 GB 或 48 GB 卡可以做 codec 预处理、短序列 QLoRA、小模型控制实验，但不应作为 7B 原生长音频双工训练的默认承诺。两张消费卡也不会自动变成一张共享显存卡；缺少 NVLink/NVSwitch 时，频繁跨卡通信会进一步损害长序列训练效率。

### 5.2 研究型中等规模：4–8×80/141 GB

推荐：`4–8×H100 80GB`，或 `4–8×H200 141GB`。

适合：

- 7B LoRA、adapter 或 partial fine-tuning；
- Thinker 与 Talker 分阶段训练；
- 数百到数千小时双工数据的多轮实验；
- 冻结 Talker 的 control/action GRPO；
- 较长 duration、较大有效 batch 和系统性消融。

这一档最适合论文研究，因为它允许同时保留 baseline、ablation 与 evaluation，而不是把所有卡只用来勉强装下一个 job。建议先用 1–2 卡做 memory/throughput sweep，再决定 4、8 卡如何分配。

### 5.3 7B–9B 全参数 Mid-training：8 卡能试，16–32 卡更现实

`8×H100 80GB` 在充分分片、checkpointing、合理序列长度下可以尝试 7B–9B 全参数训练，但“能启动”不等于“能在合理周期内完成”。如果要同时训练音频/视觉 adapter、Thinker、Talker，并保持较长连续序列与有效全局 batch，更现实的区间是 `16–32×H100/H200`。

决定卡数的四个变量是：

```text
memory fit      可训练参数 + activation + temporary buffers
global batch    每次更新需要多少有效秒数/帧数/token
throughput      目标 wall-clock 是否允许反复消融
reliability     是否留出验证、转换和故障恢复资源
```

### 5.4 论文级完整 Omni 与双工 RL：32–64+，公开锚点为 128×H20

DuplexOmni 的公开训练配置是 `128×NVIDIA H20`、batch size 128、Thinker LR `1e-5`、Talker LR `1e-4`，两类优化数据 1:1。[3] 这可以作为完整 Qwen3-Omni 双工训练的论文级锚点，但不能反推“所有 Omni 必须 128 卡”。模型、序列、数据和工期不同，资源也会不同。

带实时语音/视频 rollout 的 RL 还要单独预算：

```text
training pool: actor update, optimizer, reference/log-prob
rollout pool: streaming generation, tool environment, simulators
reward pool: ASR, audio quality, judge, verifier, safety checks
data pool: preprocessing, codec extraction, replay storage
```

本文建议：4–8 卡只做决策 head 或小规模 GRPO 原型；16–32 卡做 7B 多轮 agent RL；32–64+ 卡再考虑语音/视频 rollout 与 policy update 并行。真正的瓶颈常常不是 backward，而是低 batch、长时程、不可批处理的环境交互。

## 6. 节点、网络和存储不能最后再补

单节点优先选择带 NVLink/NVSwitch 的 8-GPU 服务器。跨节点训练建议使用 200/400 Gb/s InfiniBand 或等价 RDMA 网络；每节点配置 64–128 个 CPU core、512 GB–1 TB RAM 和本地 NVMe scratch。音视频 decode、resample、codec、Parquet scan 与随机读取会把 CPU 和存储拖成瓶颈，因此数据预处理 GPU 与主训练 GPU 最好分开调度。

未压缩音频的体积可以直接估算：

```text
24,000 samples/s × 2 bytes × 2 channels × 3,600 s
≈ 0.346 GB / hour
```

一万小时双声道 24 kHz PCM 已约 3.46 TB，还未包含原文件、transcript、codec token、多个数据版本和 shuffle shard。DuplexOmni 官方称完整可训练数据约 9 TB，并公开 metadata 与一个完成 shard，而不是直接托管全部产物。[10] 对这种规模，本文建议准备 20–30 TB scratch，以容纳源数据、PCM、codec、Parquet、索引、checkpoint 和临时副本。

## 7. Stage 0：先冻结数据 contract 和 codec

训练前先定义一个不会随实验脚本漂移的数据 contract：

```json
{
  "session_id": "...",
  "streams": {
    "user_audio": "...",
    "assistant_audio": "...",
    "video": "..."
  },
  "timeline": [
    {"t_ms": 1440, "speaker": "user", "text": "..."},
    {"t_ms": 1600, "control": "LISTEN"},
    {"t_ms": 2080, "action": "CALL_TOOL", "args": {}}
  ],
  "playback": {"generated_until_ms": 0, "played_until_ms": 0},
  "codec_version": "...",
  "loss_mask_version": "..."
}
```

必须保存原始时间戳、speaker、采样率、codec 版本、增强参数、合成来源、工具事件和 loss mask。只保存最终拼接 token 会让后续无法重算对齐、无法定位泄漏，也无法区分用户真正听到的内容。

Codec 在进入大模型训练前至少要通过：

- 因果 encode/decode，不读取未来帧；
- chunk 边界无明显 click、音量跳变或状态重置；
- 单独测量重建 intelligibility、音质和说话人一致性；
- 在目标硬件上 `RTF < 1`，否则实时输出会持续欠债；
- 训练 chunk 与推理 cache contract 一致；
- codec/token 版本可追溯，旧 shard 不与新 tokenizer 静默混用。

Codec 没过门时，不要用大模型训练去“修音质”。那会把表示缺陷扩散到 Thinker、Talker 和 reward。

## 8. Stage 1：文本底座与模态对齐

对大多数团队，Stage 1 不是从零训练一个文本 LLM，而是从强 checkpoint 做 Continual Pretraining 与 modality alignment。先冻结 LLM 和 encoder，只训练 projector/aligner，让音频、图像、视频表示进入语言空间；再逐步解冻 LLM 顶层或小范围 LoRA。

本文建议的 **alignment/CPT 启动采样概率**：

| 数据类型 | 起始比例 | 目的 |
| --- | ---: | --- |
| text replay | 30% | 保留知识、指令与句法能力 |
| single-stream audio-text | 35% | ASR、caption、speech QA、语义对齐 |
| image/video-text | 20% | 视觉与时间语义 |
| mixed audio-video-text | 10% | 跨模态绑定与联合证据 |
| early streaming/duplex | 5% | 提前暴露时间轴和控制 token |

可在 `25–35% text`、`30–40% single audio`、`15–25% vision`、`10–15% mixed`、`5–10% streaming` 范围内调节。Moshi 使用 50% 纯文本 batch 是一个强烈证据：音频训练会遗忘文本能力；但它不是所有 Omni 项目必须复制的比例。[1]

晋级门：单模态 benchmark 明显提升，文本回归不超过预先约定阈值；projector 的输出 norm 与 LLM embedding 稳定；随机丢弃任一模态时，模型行为符合预期而不是完全崩溃。

## 9. Stage 2：单流音频/视频 Mid-training

这一阶段让模型在海量、相对容易获得的单流数据上学习声学、语义和时间先验。Moshi 的公开做法很有参考价值：700 万小时单流音频，batch 覆盖 16 小时音频，每个样本 5 分钟；文本 token 30% 随机 mask，文本与音频延迟在 `-0.6s` 到 `+0.6s` 随机化；训练 1M steps，同时 50% 使用纯文本 batch。[1]

需要注意：700 万小时教会的是通用音频建模，不自动产生真实双工行为。单流阶段适合训练：

- speech/audio understanding；
- text-audio inner representation；
- Talker 的基本 continuation；
- 视频和音频真实时间对齐；
- 长流 cache 稳定性。

关键 tricks：

- 按有效秒数、帧数或 token budget batching，不按样本条数；
- duration bucketing，减少 silence/padding；
- 从短 chunk 到长 stream 做 curriculum；
- temporal jitter、随机 chunk 起点和 cache reset；
- semantic codebook 高权重，acoustic codebook 低权重；Moshi 评测中使用 semantic token 权重 100、acoustic token 权重 1。[1]
- text、aligner、Talker 使用不同 optimizer group 与 LR；
- 分头记录 text/audio/codebook loss，不只看总 loss。

## 10. Stage 3：从单流到真正双流

双工 Mid-training 的目标不是“在每个样本里放两个说话人”，而是学习连续时间中的并发关系：沉默、等待、重叠、backchannel、用户纠正、助手停止、重新接续。

Moshi 先用 diarization 将单流音频模拟成多流，训练 100k steps、8 小时 audio batch，并保留 10% text-only batch；随后使用约 2,000 小时、分声道录制的 Fisher 电话对话训练 10k batches，40 分钟 audio batch。论文明确指出模拟流没有自然 overlap，inactive stream 又过于干净，因此真实双人数据不可替代。[1]

本文建议的 **Duplex Mid-training 启动配方**：

| 数据类型 | 采样概率 |
| --- | ---: |
| 真实双声道自然对话 | 30% |
| Writer-Director 合成双工轨迹 | 25% |
| 单流 speech instruction / ASR / QA | 15% |
| 音视频 grounded interaction | 10% |
| text capability replay | 10% |
| interruption / correction / noise / tool-delay hard cases | 10% |

真实双工数据不应被海量 TTS 淹没。合成轨迹擅长控制事件分布，真实录音负责声学、停顿、犹豫、串音和失败模式。最有效的组合往往是 **synthetic timeline + real acoustics**，而不是纯合成语音或未经结构化的真实通话。

![Data mixture controller](/images/blog/omni-training-data-mixture.svg "图 3：数据配比应从固定启动配方向基于失败桶的闭环采样演化；原始文件条数不能代表训练贡献。")

## 11. 配比的单位：不要按文件条数

假设第 i 类数据的采样概率为 `p_i`，平均有效 token 为 `n_i`，loss 权重为 `lambda_i`，其梯度范数为 `g_i`，那么它对更新的粗略贡献更接近：

```text
contribution_i ∝ p_i × E[n_i] × lambda_i × E[g_i]
```

一条 5 分钟双流音频和一条 10 秒文本问答都算“一条”，显然没有意义。文章中的百分比指 **sampler 的启动概率**；训练时还要报告每类数据的有效秒数、有效 token、loss、gradient norm 和吞吐成本。

每轮评测后，可按失败率调整：

```text
new_weight_i = clamp(
  old_weight_i × exp(eta × normalized_failure_i / unit_cost_i),
  lower_i,
  upper_i
)
```

这不是让困难数据无限增权。上下界用于防止训练分布被一个噪声 benchmark 劫持；unit cost 用于避免一个极慢视频桶耗尽全部 step time。

## 12. 关键交互窗口要重采样

长对话中多数时间是普通 speech 或 silence，而真正决定体验的窗口很稀少。可以在进入 batch 前围绕事件裁剪 4–20 秒窗口，使用以下启动分布：

```text
35% normal turn transition
15% silence / wait
15% overlap / backchannel
15% interruption / correction
10% delayed reasoning / tool result
10% noise / multi-speaker / packet loss
```

silence 仍然重要，但不能让它主导梯度。对 LISTEN、WAIT、BACKCHANNEL、START、YIELD、STOP、REVISE 等 control token 做类别重加权，并分别报告 precision/recall。否则模型最容易学到的局部最优是“永远沉默”或“检测到任何声音就停止”。

## 13. Stage 4：Interaction SFT

此时才把产品行为显式写入 action space：

```text
LISTEN  WAIT  BACKCHANNEL  START_SPEAK
CONTINUE  YIELD  STOP  REVISE
CALL_TOOL  CANCEL_TOOL  ASK_CLARIFY
PREPARE  COMMIT  ABORT  MEMORY_WRITE
```

SFT 样本不应只有最终回答，还要有 earliest legal action time、当前 belief、工具状态和 playback state。例如用户在助手说话时把“昨天”纠正为“前天”，标签应同时覆盖 speech stop、slot revision、旧查询取消和新查询发起。

SFT 的目标是让动作空间可用，不是让每个决策都达到最优。晋级门至少包括：

- ordinary turn 不被过度打断；
- backchannel 不被当成抢话；
- 用户强打断后在目标延迟内静音；
- correction 能撤销旧参数；
- 未播放内容不会被当成共同记忆；
- 不可逆操作不会在缺少确认时 COMMIT。

## 14. Stage 5：Talker 稳定性、音色与 DPO

Talker 的目标与 Thinker 不同。内容正确不代表语音可懂，WER 低也不代表停顿、韵律和延迟自然。本文建议的 Talker SFT 启动配方：

```text
60% clean neutral natural speech
15% prosody / emotion / style
10% multi-speaker / timbre
10% channel and noise robustness
 5% pronunciation / number / entity hard cases
```

Qwen2.5-Omni 先用多模态上下文和口语回复做 speech continuation，再根据 WER 与 punctuation pause error 构造好坏语音对做 DPO，最后 speaker fine-tuning；它还使用 timbre disentanglement，避免特定音色与稀有文本模式错误绑定。[2] 这说明 Talker preference 的 reward 应尽量接近可观测语音缺陷，而不是只让一个通用 judge 评价“是否自然”。

Talker 训练 tricks：

- 先冻结 Thinker，确保内容条件不漂移；
- 对第一或 semantic codebook 赋更高权重；
- 数字、实体、缩写和中英混读建立独立 hard set；
- 训练音色与用户音色严格隔离，避免 voice leakage；
- 监控 WER、pause、speaker similarity、UTMOS/人评和 RTF；
- TTS 合成数据使用多样用户音色，但助手音色策略要与产品目标一致；
- 不让高噪声增强永久降低 clean speech 上限。

## 15. Stage 6：先优化控制策略，再优化整条语音

第一版 RL 建议冻结 codec、Talker 和大部分 Thinker，只训练 control/action head 或少量 policy adapter。原因很直接：时机 reward 本来就稀疏且噪声大，如果 Talker 同时更新，reward hacking 会把“说得短”“快速闭嘴”误当成更自然。

一个基础 reward 可以写为：

```text
R = w1 × task_success
  + w2 × semantic_quality
  + w3 × timing_quality
  + w4 × interruption_recovery
  + w5 × safety
  - w6 × response_delay
  - w7 × false_start
  - w8 × premature_action
  - w9 × unnecessary_tool
  - w10 × compute_cost
```

不要从用户结束说话才开始计算延迟。更合理的起点是 `t_ready`：模型已经获得足够语义信息、允许执行该动作的最早时刻。搜索、预取等可撤销动作可以早做；支付、退款、转账等高风险动作只能 PREPARE，经过确认后 COMMIT。

适合 RL 的窗口包括 turn boundary、长停顿、抢话前后、用户纠正、工具返回和 commit 前。长达十分钟的完整语音 rollout 一开始就做 RL，会让信用分配、显存与环境吞吐同时失控。

## 16. Stage 7：Agentic RL 是环境问题，不只是算法问题

Agentic RL 与普通单轮 GRPO 的差别是策略会多次观察、行动、接收工具结果并继续决策。需要记录：

```text
observation: audio prefix, video, playback, memory, tool state
action: control token, speech plan, tool call, cancel, commit
environment: new chunk, user correction, timeout, tool result
reward: task, timing, safety, effort, latency, compute
```

本文建议的 rollout 启动分布：

```text
40% normal tasks
20% user interruption and parameter revision
15% ambiguity and clarification
10% tool latency, failure and cancellation
10% high-risk confirmation and irreversible action
 5% noise, adversarial and multi-speaker interference
```

轨迹必须保存 prompt/token ID、loss mask、模型权重版本、工具版本、环境 seed、音频时间戳和 reward 分项。否则异步 rollout 产生的旧策略轨迹无法判断 staleness，失败也无法重放。

![Agentic RL loop](/images/blog/omni-training-agentic-rl-loop.svg "图 4：Omni RL 需要把流式 rollout、环境事件、分项 reward、策略更新和回归晋级连成闭环，而不是只在最终语音上打一个总分。")

## 17. RL 稳定性的具体技巧

- **场景内归一化 reward。** 不要让简单 QA 的高分压过银行确认或长时工具任务。
- **监控 KL、clip fraction、advantage 分布和 policy staleness。** 只看平均 reward 会错过策略坍缩。
- **保留 supervised replay。** 每个 RL batch 混入稳定的 SFT/control 样本，限制行为漂移。
- **分离 process 与 outcome reward。** 任务成功不代表工具调用时机正确；时机正确也不代表最终任务完成。
- **独立惩罚 false interruption。** “能停止”不能靠见到任何声音都停止来实现。
- **使用 playback-aware rollback。** 已生成但未播放的尾部可以删除，已经播放给用户的承诺必须进入共同状态。
- **环境先做 deterministic toy。** 在可控延迟、固定用户脚本和可重放工具上验证 reward，再接真实电话或视频流。
- **先做可取消只读工具。** 搜索、查询和预加载比退款、删除、支付更适合最早一轮 Agentic RL。

## 18. 冻结、loss 和学习率如何分工

![Freeze and loss map](/images/blog/omni-training-freeze-loss-map.svg "图 5：每一阶段只开放必要模块，并让 loss、学习率和回归门与该阶段的新增能力一一对应。")

下面是一套启动策略，不是通用最优超参数：

| 阶段 | 建议开放 | 建议冻结 | LR 关系 |
| --- | --- | --- | --- |
| alignment | projector/aligner | codec、encoder、LLM | aligner 最高 |
| CPT / single stream | aligner + LLM 部分或全部 | codec，初期冻结 Talker | LLM 低，aligner 中 |
| duplex mid-train | control + Thinker，随后 Talker | 两者交替冻结 | Talker 可高于 Thinker |
| Interaction SFT | control/action + 少量 backbone | codec，大部分 encoder | control 中，LLM 低 |
| Talker DPO | Talker/MTP | Thinker、codec | 小 LR，强 KL/参考约束 |
| control RL | policy adapter/head | codec、Talker、大部分 Thinker | 最小且独立 |
| Agentic RL | action/control，必要时少量 Thinker | 声学生成链 | 依据 KL 和 staleness 调整 |

DuplexOmni 的公开值是 Thinker `1e-5`、Talker `1e-4`，但它基于特定架构、batch 128 和 128×H20，不能直接复制到 1 卡 LoRA。[3] Moshi-finetune 对 LoRA 推荐从 `2e-6` 开始、weight decay 0.1、rank 不超过 128。[9] 超参数必须连同可训练参数、global batch、duration 和 optimizer 一起解释。

## 19. 数据清洗比继续加小时更重要

至少做三层去重：

1. waveform fingerprint，去除重复录音与切片重叠；
2. transcript semantic dedup，避免同一脚本被大量 TTS 音色复制；
3. speaker/voice dedup，防止同一说话人跨 train/val/test。

还要检查：

- transcript 与音频 timestamp 偏移分布；
- 两声道 speaker leakage、串音和声道反转；
- silence、overlap、backchannel、interruption 的真实比例；
- TTS 模型、voice ID 和脚本来源；
- 视频帧与音频时钟漂移；
- 许可、隐私、旁观者和声音克隆授权；
- benchmark 污染和提示模板重复。

训练/验证/测试应按 speaker、session 和原始来源隔离，而不是切完 chunk 后随机分。否则相邻片段或同一个 TTS voice 会让验证结果虚高。

## 20. 工程层面的常见稳定性 tricks

- **prefix-streaming 一致性：** 训练只允许看到当前前缀，不能离线偷看未来；推理 cache 的 mask 与训练完全一致。
- **duration curriculum：** 先短片段收敛动作和对齐，再逐步拉长；长期能力仍要用目标 duration 验证。
- **temporal jitter：** 随机改变音频/文本延迟、chunk 边界和视频采样相位，防止模型记住固定格点。
- **网络扰动：** packet loss、jitter、短暂 cache reset、tool delay 进入训练或模拟器。
- **声学增强分层：** clean、noise、echo、reverb 分桶报告，不用一个平均分掩盖 clean regression。
- **padding/silence 降权：** Moshi 在音频 batch 中把 padding CE 权重降低 50%。[1]
- **optimizer state 分离：** Moshi 为 text-only 与 audio batch 使用独立 optimizer state，并在 audio batch 中把 text embedding/head LR 乘 0.75。[1]
- **梯度监控：** 按 encoder、aligner、Thinker、control、Talker、codebook 记录 norm 与 overflow。
- **断点可复现：** 保存 sampler state、shard cursor、codec 版本和随机增强 seed，不只保存权重。
- **先 profile 后扩卡：** 分别测 dataloader、forward、backward、all-reduce、checkpoint 与 eval 的时间占比。

## 21. 每阶段的晋级门

| 阶段 | 必须通过的核心门 | 不能被平均值掩盖的回归 |
| --- | --- | --- |
| codec | 重建、因果、chunk、RTF | 数字与实体可懂度 |
| alignment | 单模态理解与跨模态绑定 | 文本能力、模态缺失鲁棒性 |
| single stream | speech QA、长流 cache | 文本遗忘、早停沉默 |
| duplex | turn、overlap、interrupt | false stop、永远沉默、抢话 |
| Interaction SFT | action validity、revision | 未确认 commit、错误 memory |
| Talker | WER、pause、voice、RTF | 内容正确性和 clean quality |
| control RL | timing、yield、repair | reward hacking、过短回复 |
| Agentic RL | task success、tool correctness | rollback、审计、高风险误执行 |

最终评测至少报告 P50/P95/P99：首个有效语义动作延迟、`t_ready` 到工具触发、用户打断到助手静音、工具结果到回复修订，以及端到端任务完成时间。质量侧同时报告 content、speech、turn-taking、tool、memory 与 safety；不要合成一个无法诊断的总分。

## 22. 三套可以执行的训练方案

### 22.1 一卡方案：先学会整个闭环

```text
Hardware: 1×H100 80GB or A100 80GB
Backbone: Moshi 7B or existing Omni checkpoint
Train: LoRA/control head
Data: 10–100 h curated dual-channel + text replay
Goal: codec contract, data timeline, LISTEN/SPEAK/STOP baseline
RL: text/state simulator only, no end-to-end audio RL
```

输出应是一套可重放数据管线、一组 failure buckets 和一个不会在真实流式推理中崩溃的 baseline，而不是追求通用模型榜单。

### 22.2 八卡方案：完成一篇严谨的交互训练论文

```text
Hardware: 8×H100 80GB or H200 141GB
Train: LoRA/partial Thinker + alternating Talker
Data: 500–3,000 h real/synthetic duplex + hard windows
Experiments: freeze order, data mixture, text replay, timing policy
RL: frozen Talker, control/action GRPO or DPO
Evaluation: offline + real streaming + product task simulator
```

卡的分配不要始终 8 卡单 job。开发期可用 4 卡训练、2 卡 rollout、1 卡 reward、1 卡评测；稳定后再 8 卡扩大全局 batch。

### 22.3 32 卡以上：全参数 Mid-training 与 Agentic rollout

```text
Hardware: 16–32×H100/H200 for full-parameter 7B–9B mid-training
Scale-up: 32–64+ for parallel speech/video rollout and policy update
Reference: DuplexOmni reports 128×H20 for its complete training setup
Data: multi-TB sharded dataset, independent preprocess and eval pools
System: Megatron/FSDP/ZeRO + high-speed fabric + async rollout runtime
```

这一档必须先有 8 卡 profile 和 scaling curve。没有单节点 tokens/s、audio-seconds/s、MFU、通信占比、checkpoint 时间和 dataloader stall，直接申请 32 或 64 卡只是把未知问题放大。

## 23. 框架如何选

| 目标 | 推荐起点 | 边界 |
| --- | --- | --- |
| 学原生双流与 LoRA | Moshi + moshi-finetune | 主要是 speech-text，不是视觉 Omni；微调库不是完整预训练复现 |
| CPT / SFT / DPO 原型 | ms-swift | 必须核对目标模型 Talker、codec 与 streaming loss 是否真正支持 |
| 简单 SFT/DPO 数据验证 | LLaMA-Factory / TRL | 不适合直接承担自定义多流 codec 与大规模异步语音 rollout |
| 大规模 Mid-training | Megatron-LM / NeMo | 工程复杂，适合从已验证 recipe 扩大 |
| 正式分布式 RL | verl | 适合拆 rollout、reward、advantage、policy update、reference 与同步 |
| 高级大规模 RL | slime | 更接近 Megatron + SGLang 与自定义 rollout，门槛更高 |
| Agent 轨迹与环境 | Agent Lightning / Agent-R1 | 是 trace、环境和 MDP 抽象，不自动替代权重训练后端 |

合理顺序是：用最简单框架验证 loss 和数据，再迁移到扩展性更强的训练栈。不要因为某框架“支持某模型名称”，就默认它覆盖完整 Thinker-Talker、codec 和双工推理 contract。[11][12][13][14][17][18][19]

## 24. 最常见的失败，以及根因

**模型永远沉默。** 通常不是缺少语言能力，而是 silence/padding 数量、loss 权重或错误终点标签压倒了稀有 START/BACKCHANNEL。

**模型一听到声音就停止。** interruption reward 没有区分背景噪声、附和和抢话，false stop 缺少独立惩罚。

**音频 benchmark 提升，文本推理下降。** text replay 太少、共享 optimizer state 被音频梯度主导，或 oral-style SFT 过窄。Moshi 也观察到不保留 50% text-only batch 会明显损害问答能力。[1]

**离线很好，流式崩溃。** 训练看到了完整音频，推理只有 prefix；chunk mask、position、cache reset 或 codec state 不一致。

**Talker 很自然，但说错数字。** acoustic loss 主导，semantic codebook、实体 hard set 和 WER/punctuation preference 不足。

**工具调用很快，但改口后无法恢复。** 训练只有成功轨迹，没有 CANCEL、REVISE、ABORT 和 playback-aware rollback。

**扩卡后吞吐不升反降。** duration bucket 不合理、数据解码慢、跨节点 all-to-all、checkpoint 或 reward service 成为瓶颈。

**平均 reward 上升，用户体验下降。** reward 把短回复、保守沉默或频繁 backchannel 当成捷径，且缺少场景分桶和真实 streaming 回归。

## 25. 最终推荐路线

如果团队刚开始，我会采用：

```text
Moshi / existing Omni checkpoint
→ codec and timeline contract
→ single-GPU LoRA smoke test
→ alignment with text replay
→ synthetic duplex + real dual-channel mid-training
→ Interaction SFT with explicit control/action tokens
→ frozen-Thinker Talker stabilization and DPO
→ frozen-Talker control-policy RL
→ tool simulator and Agentic RL
→ only then consider broader joint optimization
```

真正应被规模化的不是第一次能跑通的配置，而是已经通过 toy experiment 的因果假设。例如：“模型永远沉默是因为 silence 梯度占比过高”，先在可控数据上改变 silence 权重并验证；“用户纠正无法生效是因为轨迹没有 action invalidation”，先增加 CANCEL/REVISE 窗口并验证。只有根因实验成立，才值得扩到千小时和几十张卡。

一句话总结：

> **Omni 训练的核心不是把所有模态、模块和 reward 一次性打开，而是建立一条可诊断的能力阶梯：每一步只增加必要自由度，用数据配比控制遗忘与稀有事件，用冻结和分头 loss 控制漂移，用晋级门决定是否值得投入下一档算力。**

## 参考资料

1. Défossez et al., [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037), 2024.
2. Xu et al., [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215), 2025.
3. Huang et al., [DuplexOmni: A Scalable Framework for Real-Time Omni Interaction](https://arxiv.org/abs/2606.09186), 2026.
4. DeepSpeed, [ZeRO documentation](https://www.deepspeed.ai/tutorials/zero/).
5. PyTorch, [Fully Sharded Data Parallel documentation](https://pytorch.org/docs/stable/fsdp.html).
6. NVIDIA, [H100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h100/).
7. NVIDIA, [H200 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h200/).
8. NVIDIA, [A100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/a100/).
9. Kyutai, [moshi-finetune](https://github.com/kyutai-labs/moshi-finetune).
10. MuyeHuang, [DuplexOmni repository and data pipeline](https://github.com/MuyeHuang/DuplexOmni).
11. ModelScope, [ms-swift](https://github.com/modelscope/ms-swift).
12. NVIDIA, [Megatron-LM](https://github.com/NVIDIA/Megatron-LM).
13. verl project, [verl](https://github.com/verl-project/verl).
14. THUDM, [slime](https://github.com/THUDM/slime).
15. Microsoft, [Agent Lightning](https://github.com/microsoft/agent-lightning).
16. AgentR1, [Agent-R1](https://github.com/AgentR1/Agent-R1).
17. hiyouga, [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
18. Hugging Face, [TRL](https://github.com/huggingface/trl).
19. NVIDIA, [NeMo](https://github.com/NVIDIA/NeMo).

> 证据说明：Moshi、Qwen2.5-Omni、DuplexOmni 和 moshi-finetune 的数字均来自论文或官方仓库；其中 Moshi 未披露 H100 数量，Qwen2.5-Omni 未披露完整训练 GPU 数和数据配比。本文 GPU 档位、20–30 TB scratch、各阶段数据百分比、学习率分工与 RL 卡数属于工程启动建议，必须通过目标模型的显存与吞吐 profile 校准。

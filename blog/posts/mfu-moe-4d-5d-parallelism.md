# MFU 与 MoE 4D/5D 混合并行：从利用率公式到 DP、TP、PP、EP、CP

训练大模型时，人们经常同时讨论两类问题：一类是“这批 GPU 到底跑得好不好”，对应 Model FLOPs Utilization（MFU，模型 FLOPs 利用率）；另一类是“模型和数据应该怎样切到这些 GPU 上”，对应 Data Parallelism（DP）、Tensor Parallelism（TP）、Pipeline Parallelism（PP）、Expert Parallelism（EP）以及 Context Parallelism（CP）。

这两类问题必须放在一起看。并行度不是越多越先进，每增加一个并行维度，都会用新的通信、同步或流水线空泡换取显存容量与可扩展性。MFU 则把这些代价最终压缩成一个效率信号：理想模型计算量占设备理论峰值算力的多少。

对稠密模型，常见组合是 DP、TP、PP，通常被称为 3D parallelism。MoE 在此基础上增加 EP，形成 DP、TP、PP、EP 四种并行机制的组合。长上下文训练再增加独立的 CP，可以形成 DP、TP、PP、EP、CP 五维笛卡尔并行。Sequence Parallelism（SP）、ZeRO/FSDP、Expert Tensor Parallelism（ETP）也会参与系统设计，但它们是否被计作“第五维”，不同框架和文章并不统一。

因此，这篇文章不把“4D/5D”当作固定品牌名，而是先定义坐标，再讨论组合：

```text
本文的 4D：DP × TP × PP × EP
本文的 5D：DP × TP × PP × EP × CP
SP：通常复用 TP group，不单独乘 world size
ZeRO/FSDP：DP 维度上的模型状态分片，不天然是新的正交 rank 维度
```

![MFU accounting](/images/blog/mfu-model-flops-accounting.svg "图 1：MFU 用理想模型计算吞吐除以设备理论峰值，所有并行与系统开销都会通过 step time 反映出来。")

## MFU 到底衡量什么

PaLM 论文把 MFU 定义为：观测到的 token throughput，相对于系统以理论峰值 FLOPs 执行模型必要 forward 和 backward 运算时所能达到的最大 token throughput。等价地，可以写成：

```text
MFU = observed_model_FLOPs_per_second / theoretical_peak_FLOPs_per_second

    = tokens_per_second × model_FLOPs_per_token
      ------------------------------------------------
      number_of_accelerators × peak_FLOPs_per_accelerator
```

其中：

- `tokens_per_second` 应是整个训练 job 的全局有效 token 吞吐；
- `model_FLOPs_per_token` 是模型定义所要求的理想 forward + backward 计算量；
- `peak_FLOPs_per_accelerator` 必须匹配真实使用的数据类型、Tensor Core 模式和硬件 SKU；
- 分母是所有参与训练设备的理论峰值之和。

MFU 为 50%，不表示 GPU 有一半时间完全空闲。它表示：按统一的“模型必要计算量”记账，观测吞吐折算出的计算速率等于理论峰值的 50%。剩余部分可能消耗在通信、内存访问、pipeline bubble、重计算、负载不均、CPU launch gap、数据等待，也可能来自 kernel 无法达到峰值。

PaLM 报告 540B 模型在 6144 个 TPU v4 上达到 46.2% MFU，同时报告 57.8% Hardware FLOPs Utilization（HFU）。这两个数字不同，正好说明 MFU 与“硬件实际做了多少计算”不是一回事。

## MFU 与 HFU 的区别

HFU 试图统计硬件实际执行的 FLOPs。Activation recomputation 会让设备重新执行一部分 forward 运算，硬件确实做了更多计算，因此 HFU 可能上升。但这些重复计算没有改变模型定义，也没有多处理 token。

MFU 只把模型必要的 forward + backward FLOPs 计入分子。重计算增加 step time，却不增加 MFU 的模型 FLOPs，所以会降低 MFU。PaLM 提出 MFU 的动机之一，就是让不同 rematerialization 策略、不同系统实现之间更容易比较。

可以用一个极端例子理解：如果为了“让 GPU 忙起来”而把同一层无意义地重复计算十次，HFU 可能很好看，训练吞吐却会下降。MFU 不会奖励这种工作。

不过 MFU 也不是绝对客观的自然常数。不同报告如果对 embedding、attention、激活函数、router、padding token、loss、共享参数或稀疏计算采用不同 FLOPs 口径，结果仍然不能直接横比。可靠的 MFU 报告必须同时给出公式、模型结构、有效 token 定义、设备峰值口径和 step-time 测量窗口。

## 稠密 Transformer 的 FLOPs 近似

对参数量为 `N` 的 decoder-only dense Transformer，经常使用：

```text
training FLOPs per token ≈ 6N
```

直觉是一次 forward 对每个权重执行大约一次乘加，按乘和加各计一个 FLOP，约为 `2N`；backward 中输入梯度和权重梯度计算约为 forward 的两倍，于是总计约 `6N`。

这是一个方便的主项近似，不是精确公式。长序列下 attention 的 `QK^T` 与 `AV` 计算会变得显著；embedding、输出词表、门控 FFN、参数共享和激活重计算也会改变记账。PaLM 在附录中使用了包含 attention 项的公式。因此，比较同一训练 run 的趋势时可以用 `6N` 快速估计，正式报告则应该采用与架构一致的逐层公式。

## MoE 的 MFU 不能使用总参数量

MoE 最容易犯的错误，是把总参数量 `N_total` 直接代入 `6N`。假设一层有 64 个 experts，每个 token 只路由到 top-2；另外 62 个 experts 的参数没有参与这个 token 的 forward/backward。用全部 64 个 experts 计算模型 FLOPs，会把分子夸大约 32 倍，甚至可能算出超过 100% 的“MFU”。

MoE 应按每 token 实际激活的模型路径计算：

```text
F_model_per_token
  = F_shared_dense_path
  + F_router
  + top_k × F_one_expert
  + F_attention_and_other_terms
```

或者在参数主项近似下，把总参数量拆成 shared parameters 与每 token activated expert parameters：

```text
N_active_per_token = N_shared + top_k × N_one_expert_path
training FLOPs per token ≈ 6 × N_active_per_token + attention correction
```

这里还要区分“模型必要计算”和“实现实际计算”。如果 expert kernel 为了固定 capacity 而给每个 expert padding 大量空 token，这些 padding 计算属于实现开销，通常不应增加理想模型 FLOPs；它会通过更长 step time 拉低 MFU。如果某些 token 因 capacity 溢出而被 drop，有效 token 和必要计算的口径也必须写清楚。

Router、token permutation、All-to-All 和 combine 会消耗时间。通信本身不是模型数学定义里的 FLOPs，所以不会增加 MFU 分子，却会降低吞吐。这正是 MFU 对 MoE 系统效率敏感的原因。

## 一个 MFU 算例

假设一个训练任务使用 64 张加速卡，每张卡在目标精度下的理论峰值按 `1 PFLOP/s` 计；模型每个有效 token 的 forward + backward 理想计算量为 `400 GFLOPs`；全局吞吐为 `80,000 tokens/s`：

```text
observed model throughput
= 80,000 × 400 GFLOPs/s
= 32 PFLOP/s

system theoretical peak
= 64 × 1 PFLOP/s
= 64 PFLOP/s

MFU = 32 / 64 = 50%
```

这是纯示意数字。真实计算时不能把不同 GPU 型号、频率、稀疏峰值、BF16 峰值和 FP8 峰值混用；也不能用包含 padding 的 sequence length 计算 FLOPs，却用去掉 padding 的 tokens/s，或者反过来。

## DP：切 Batch，复制模型

Data Parallelism 的基本做法是：每个 DP rank 保存一份模型副本，处理不同 micro-batch，backward 后对梯度做 All-Reduce 或 Reduce-Scatter，使副本继续保持一致。

DP 的优势是局部计算完整、矩阵规模大，通常最有利于 MFU。只要 global batch 允许，扩展训练时通常应尽可能保留较大的 DP degree，而把 TP、PP、EP、CP 控制在满足显存和模型规模的最小值。

标准 DDP 会复制参数、梯度和 optimizer states。ZeRO 或 FSDP 在 DP group 内进一步分片这些模型状态：ZeRO-1 分 optimizer states，ZeRO-2 再分 gradients，ZeRO-3/FSDP 再分 parameters。它们改变 DP 维度内的内存与通信方式，但通常不额外创造一组独立 rank 坐标。

DP 的主要通信发生在 backward 梯度同步，往往可以与反向计算 overlap。它适合跨节点扩展，因为每步通信频率通常低于 TP 的逐层 collective，也不像 EP 那样搬运每层路由后的 token。

## TP：切单层矩阵

Tensor Parallelism 把同一层的大矩阵沿行或列切到多个 ranks。以 Transformer 为例，QKV projection、attention output projection 和 FFN 的两个线性层可以采用互补的 column/row parallel 切法，使中间张量局部计算，并在必要位置执行 All-Reduce、Reduce-Scatter 或 All-Gather。

TP 直接减少每张卡持有的层参数和部分 activation，解决单层太宽、单卡放不下的问题。但它通常每层都通信，对延迟和带宽非常敏感。TP degree 过大时，每张卡的 GEMM 变小，kernel 效率下降，collective 又变得更频繁，MFU 可能明显下降。

因此 TP 通常优先放在 NVLink/NVSwitch 等高带宽域内。Megatron 的经验原则也是先把 TP 控制在满足显存的最小范围，再用 PP 和 DP 扩到更多节点。

Sequence Parallelism 常与 TP 配套。Megatron 中的 SP 会把 LayerNorm、Dropout 等原本在 TP ranks 上重复保存的 activation 沿 sequence 维分片，并复用 TP process group。它降低 activation memory，但通常没有独立的 `SP_size` 乘到 world size 上。这一点和后文的 CP 不同。

## PP：切模型深度

Pipeline Parallelism 把连续层分到不同 pipeline stages。第 0 个 stage 处理前几层，把 activation 发给第 1 个 stage；最后一个 stage 计算 loss，backward 再反向传递 activation gradients。

PP 适合很深的模型，因为它按层数分摊参数、optimizer states 和 activation。stage 之间主要是 point-to-point send/recv，通信量可能低于跨节点 TP，因此常用来跨节点扩展。

代价是 pipeline bubble。把一个 global batch 切成 `M` 个 micro-batches 后，流水线需要 warmup 和 cooldown。对非交错 schedule，一个常见直觉近似是 bubble fraction 与：

```text
(P - 1) / (M + P - 1)
```

同量级，其中 `P` 是 pipeline stages。`M` 越大，空泡占比越小，但 activation、调度和优化器语义会受到影响。Interleaved/virtual pipeline 可以进一步减少 bubble，却会增加 P2P 次数和调度复杂度。

PP 还存在负载均衡问题。embedding、final norm、loss、dense 层和 MoE 层的成本不同，简单按层数平均切分不一定按时间均衡。某个 stage 慢 10%，整条 pipeline 都会被它限制。

## EP：切 Experts，而不是普通层

Expert Parallelism 把同一个 MoE layer 的 experts 分布到不同 ranks。Router 为每个 token 选择 top-k experts 后，系统必须完成：

1. 根据目标 expert 对 token 做 permutation；
2. 通过 All-to-All 或等价 dispatcher 把 token 发到持有该 expert 的 rank；
3. 各 rank 对本地 experts 执行 grouped GEMM；
4. 再通过 All-to-All 把 expert 输出送回原 rank；
5. 按原 token 顺序 combine，并乘 router weights。

![MoE 4D parallelism](/images/blog/moe-4d-parallelism-flow.svg "图 2：DP、TP、PP、EP 在一次 MoE Transformer 训练中的分工与主要通信。")

EP 可以让总 expert 参数量随设备数增长，而每张卡只保存一部分 experts。但 expert 权重分片不等于 activation 自动缩小：每个 rank 接收多少 token 取决于 router 分布、top-k、capacity 和负载均衡。Hot expert 会让部分 ranks 工作更久，其他 ranks 等待，MFU 随最慢 rank 下降。

EP 的核心通信是 token All-to-All，数据量大致与 token 数、hidden size 和 top-k 成正比。它对拓扑非常敏感，通常希望 EP group 位于节点内或高速互联域。Megatron Core 的 MoE 指南建议尽量让 `EP × TP` 落在 NVLink domain 内，并在跨节点扩展时优先增加 PP 或 DP，而不是盲目扩大 TP/EP。

## Dense DP 与 Expert DP 不是同一组

MoE 混合并行最容易出错的地方，不是公式，而是 process group 语义。

非 expert 参数，例如 attention、norm、router 和 shared dense 层，通常需要在“dense data-parallel group”中同步。Expert 参数只应该在持有同一个 expert replica 的 ranks 之间同步。如果 rank A 持有 expert 0，rank B 持有 expert 1，就不能把它们当作同一 expert 的 DP replicas 去 All-Reduce gradients。

假设一个 dense DP pool 有 16 个 ranks，`EP=4`。一种常见组织方式是把这 16 个 ranks 拆成 4-way EP 和 4-way expert-data replication：

```text
D_dense = EP × D_expert
16      = 4  × 4
```

对 shared parameters，16 个 ranks 都是数据并行副本；对每个具体 expert，它只有 4 份副本，对应 `D_expert=4`。这就是为什么不同框架里的 `data_parallel_size` 可能看起来不同：有人用 `DP` 指 `D_dense`，有人在 4D/5D 公式里用 `DP` 指 `D_expert`。

两种 world-size 写法可以是等价的：

```text
W = TP × PP × CP × D_dense

D_dense = EP × D_expert

所以 W = TP × PP × CP × EP × D_expert
```

如果把最后一式中的 `D_expert` 简写成 `DP`，就得到常见的：

```text
W = TP × PP × CP × EP × DP
```

因此，看到 world-size 公式时，第一件事不是代数，而是确认其中 `DP` 的定义。

## 4D MoE：DP × TP × PP × EP

在本文口径下，4D rank 可以写成坐标：

```text
rank = (d, p, e, t)

d: expert-data replica coordinate
p: pipeline stage coordinate
e: expert-parallel coordinate
t: tensor-parallel coordinate
```

固定其余坐标、只改变某一维，就得到对应 process group。TP group 在一层内部做 collective；EP group 做 token dispatch/combine；PP group 传 activation；DP group 同步对应参数的 gradients 或 optimizer states。

一个 128 GPU 的示意配置是：

```text
TP = 2
EP = 4
PP = 4
D_expert = 4

world size = 2 × 4 × 4 × 4 = 128
```

如果每节点 8 张 GPU，可以把 `TP × EP = 8` 优先映射到节点内高速互联；PP stages 跨节点连接；剩余副本形成 expert-data groups。真实 rank order 要看框架，因为不同线性化顺序会决定哪些 collective 跨节点。

4D MoE 的一次 forward 可以这样理解：DP 给每个 replica 不同数据；PP 把模型深度切成 stages；stage 内的 attention/shared 层使用 TP；进入 MoE FFN 后，router 产生路由，EP 把 token 送到对应 experts；输出返回后继续 TP/PP 流程。Backward 反向执行通信，最后 shared parameters 与 expert parameters 在各自正确的 DP groups 内同步。

## 第五维为什么更适合定义为 CP

Context Parallelism 沿完整 sequence length 切分网络输入和 activation。每个 CP rank 只持有一段上下文；attention 中，本地 Q 仍需要与全序列 K/V 交互，因此 ranks 要用 ring P2P、All-Gather/Reduce-Scatter 或其他 attention-aware 通信交换 KV 与梯度。

CP 和 SP 的关键区别是：

| 机制 | 切分范围 | Process group | 是否通常独立乘 world size |
| --- | --- | --- | --- |
| SP | 主要是 LayerNorm/Dropout 等 activation | 通常复用 TP group | 否 |
| CP | 整个网络输入与全部 activation 的 sequence 维 | 独立 CP group | 是 |

Megatron Core 文档明确区分两者：SP 只分一部分 activation，而 CP 切分所有 activation；attention 额外交换 KV。MQA/GQA 因 KV heads 更少，还可以减少 CP 的 KV 通信量。

因此，对长上下文 MoE，一个清晰的 5D 定义是：

```text
rank = (d, p, c, e, t)

world size = DP × PP × CP × EP × TP
```

![MoE 5D process grid](/images/blog/moe-5d-process-grid.svg "图 3：5D MoE rank grid；SP 通常嵌在 TP group 中，ZeRO/FSDP 则在 DP 维度内分片模型状态。")

沿用前面的 4D 配置，再设置 `CP=2`，world size 变成 256。每个 rank 只处理一半 sequence activation，但 attention 必须完成跨 CP ranks 的 KV 交换。

CP 不是无条件提速。短序列下，它会把本来足够大的计算切碎并引入额外通信；只有 activation memory、attention 长度或 recomputation 已成为瓶颈时，它才更有价值。Megatron 的当前指南把 8K 以上序列作为常见使用场景提示，但实际阈值仍取决于模型宽度、FlashAttention kernel、micro-batch 和硬件。

## 为什么有人把 SP 或 ZeRO 称为第五维

术语并未标准化。Megatron Core 当前列出的可组合策略包括 DP、TP、PP、CP、EP 和 FSDP，并说明 SP 与 TP 配套；它的 MoE README 也写有 `EP + DP + TP + PP + SP` 支持。这里的“五种机制”不一定意味着五个独立 process-group size 相乘。

DeepSpeed MoE 教程则说支持 five different forms of parallelism，并列出 Expert、Data、Model、ZeRO-powered data、ZeRO-Offload 等组合。这里的“five forms”是在描述能力组合，也不是一个固定的五维 rank grid。

所以工程文档里最好不要只写“5D parallel”。更可审计的写法是直接列配置：

```text
TP=2, PP=4, EP=4, CP=2, expert-DP=4, SP=on, ZeRO-1
```

这比“用了 5D 并行”提供的信息多得多，也避免不同团队对第五维的理解不一致。

## Expert Tensor Parallelism 与 Parallel Folding

MoE expert FFN 也可能大到单卡放不下，于是 expert 内部还要做 tensor parallel，这通常称为 Expert Tensor Parallelism（ETP）。问题是 dense attention/shared layers 的最佳 TP degree，不一定等于 expert FFN 的最佳 ETP degree。

例如 attention 可能需要 `TP=4` 才能放下，但每个 expert 收到的 token 已经很少；如果 expert 也做 4-way TP，单个 grouped GEMM 会被切得过小。反过来，专家特别宽时又可能需要 ETP。

Parallel folding 或异构 parallel mapping 允许 dense 路径和 expert 路径使用不同并行映射：

```text
dense path:  TP_dense × DP_dense
expert path: ETP × EP × DP_expert
```

这种做法能改善 GEMM 粒度和通信，但 process groups、checkpoint sharding、optimizer state 和 rank mapping 都更复杂。它已经超出“一个统一 TP size”的简单笛卡尔网格，因此配置与监控必须明确区分 dense TP 和 expert TP。

## 并行维度如何影响 MFU

每个并行维度都有典型的 MFU 损失模式：

| 并行维度 | 主要收益 | 主要通信/等待 | 常见 MFU 风险 |
| --- | --- | --- | --- |
| DP | 扩大数据吞吐 | gradient All-Reduce/Reduce-Scatter | global batch 受限、尾部同步 |
| TP | 分片宽层 | 逐层 collective | GEMM 过小、跨节点带宽不足 |
| PP | 分片模型深度 | activation P2P | pipeline bubble、stage 不均衡 |
| EP | 分片 experts | token All-to-All | hot experts、capacity padding、A2A 拥塞 |
| CP | 分片长序列 | attention KV exchange | 短序列切得过碎、KV 通信暴露 |

这些损失并非简单相加。TP 与 EP 可能争用同一 NVLink；DP gradient reduce 可以和 backward overlap，却可能与 EP All-to-All 同时占用网络；PP bubble 期间某些 ranks 空闲，另一些 ranks 仍在通信；CP 的 KV exchange 又可能与 attention compute overlap。

因此 MFU 低时，不能只看总 GPU utilization。需要把 step timeline 分解到不同 process groups 和不同阶段。

## MoE MFU 的四个特有陷阱

### Router 负载均衡

平均每个 expert 收到相同 token，不代表每个 step 都均衡。应该查看每层、每 expert 的 token count 分布、最大/平均比、p95/p99，以及 dropped/padded tokens。最慢 expert rank 决定 All-to-All 何时结束。

### Capacity 与 Padding

固定 capacity 有利于静态 shape 和 kernel，但 capacity factor 过大会产生空算；过小又可能 drop tokens，影响质量。Dropless MoE 避免 drop，却需要处理不规则 token counts。MFU 计算必须说明 padding 是否算作有效模型 FLOPs。

### Grouped GEMM 粒度

总激活参数量很大，不代表本地 GEMM 大。经过 EP、TP、top-k 和 micro-batch 切分后，每个 local expert 可能只收到少量 token。此时 GPU 忙于很多小 GEMM，峰值算力利用率很低。盲目增加 EP/TP 往往会让问题更严重。

### All-to-All 拓扑

EP token dispatch 不是普通 All-Reduce。流量目的地由 router 决定，跨节点比例随 mapping 改变。即使平均字节数相同，节点内 NVLink、节点间 RDMA、rail 绑定和网络拥塞也会产生完全不同的 step time。

## 一个实用的并行配置顺序

我更倾向于按“先可运行，再提高 MFU”的顺序配置，而不是一开始枚举所有维度。

第一步，建立单卡或最小规模基线。确认每 token FLOPs、loss、tokens/s、显存组成和 kernel 正确，避免把实现 bug 带进大规模集群。

第二步，满足单层显存。只有 attention/FFN 矩阵或 activation 真的放不下时才增加 TP；启用与 TP 配套的 SP，观察 GEMM size 与 collective 时间。

第三步，满足总模型深度显存。增加 PP，把 layers 按实测耗时而不只是层数均衡到 stages；用足够 micro-batches 和必要的 virtual pipeline 降低 bubble。

第四步，为 MoE 引入 EP。优先尝试较小 EP，确保本地 experts 的 GEMM 仍足够大；让 `TP × EP` 尽可能留在高速节点内拓扑，并监控 token imbalance 和 All-to-All。

第五步，仅在长上下文需要时加入 CP。比较 CP 与 activation recomputation 的显存、吞吐和通信代价，不要因为它能组成“5D”就默认启用。

第六步，把剩余设备用于 DP/expert-data replicas。DP 通常提供最干净的 throughput scaling；结合 distributed optimizer、ZeRO/FSDP 和 communication overlap 解决模型状态内存。

第七步，固定测量协议后调 MFU。至少 warm up 若干 steps，排除编译、数据缓存和 checkpoint steps，报告中位数与尾延迟，确认统计的是全局有效 tokens。

## MFU 低时怎样定位

可以按时间线从外到内排查：

1. **Step 间有大空洞**：检查 dataloader、CPU、Python GC、JIT/compile、checkpoint 和 straggler。
2. **GEMM 很短且数量多**：TP/EP 可能过大，micro-batch 或 local expert tokens 太小。
3. **TP collective 暴露**：减小 TP、放回节点内、启用 TP communication overlap，检查 rank mapping。
4. **PP ranks 周期性空闲**：增加 micro-batches、调整 stage layers、使用 virtual/interleaved schedule。
5. **EP All-to-All 占比高**：检查 router balance、top-k、capacity、dispatcher、节点内外流量与 DeepEP/Tutel 等实现。
6. **DP reduce 在 backward 尾部堆积**：调 bucket、overlap、distributed optimizer 和网络并发。
7. **长上下文 OOM 且重计算很多**：评估 CP，比较 KV exchange 与 recomputation 的成本。
8. **HFU 高但 MFU 低**：很可能有大量重计算、padding 或其他非必要工作。

优化时一次只改变一个主要并行度，并保留每次实验的 `TP/PP/EP/CP/DP`、rank mapping、global batch、micro-batch、sequence length、top-k、capacity、dtype、tokens/s、MFU、显存和通信占比。否则多个维度同时变化，很难判断收益来自哪里。

## 一份可审计的实验记录应该包含什么

只报告“MFU 55%”或“采用 5D parallel”都不够。至少应该记录：

```text
Model:
  total params / active params per token
  layers / hidden / heads / experts / top-k
  sequence length / vocabulary / precision

Parallelism:
  TP / PP / EP / CP / dense-DP / expert-DP
  SP on/off / ZeRO or FSDP stage / ETP
  rank order and intra-node mapping

Batching:
  micro batch / gradient accumulation / global batch
  valid tokens per step / padding policy

Performance:
  median step time / p95 step time / global tokens per second
  model FLOPs per token formula / device peak FLOPs assumption / MFU
  TP, PP, EP, CP, DP communication time
  pipeline bubble / recomputation / router imbalance / dropped tokens
```

只有这些数据放在一起，MFU 才能反向指导并行策略，而不是变成一个脱离配置的排行榜数字。

## 总结

MFU 衡量的是模型必要计算吞吐相对于设备理论峰值的比例。它不会奖励 activation recomputation、capacity padding 或无意义的额外计算，因此比单纯硬件忙碌程度更适合比较大模型训练系统。但 MFU 的可信度依赖统一的 FLOPs 与有效 token 口径；MoE 尤其必须使用每 token 激活参数，而不是总 expert 参数。

DP、TP、PP、EP、CP 分别切 batch、层内张量、模型深度、experts 和完整 sequence。它们解决不同显存约束，也引入不同通信：DP gradient collectives、TP 逐层 collectives、PP activation P2P、EP token All-to-All、CP attention KV exchange。

本文把 MoE 4D 定义为 `DP × TP × PP × EP`，把长上下文 5D 定义为再乘独立 `CP`。SP 通常复用 TP group，ZeRO/FSDP 通常在 DP 内分片状态；它们是重要机制，却不一定构成新的 rank 维度。Dense DP 与 expert DP 的定义也必须分开，否则 world-size、gradient groups 和 checkpoint 都可能配错。

真正有效的配置原则并不复杂：**先用最少的模型并行满足显存，再尽可能扩大 DP；让高频 TP/EP 通信留在最快拓扑内；用 PP 扩展深度和节点；只在长上下文确有需要时增加 CP；最后通过时间线和 MFU 找到暴露的通信、空泡与负载不均。** 4D/5D 的价值不在维度数量，而在每个维度都有明确职责、正确 process group 和可验证收益。

## 参考资料

- [Aakanksha Chowdhery et al., PaLM: Scaling Language Modeling with Pathways, 2022](https://arxiv.org/abs/2204.02311)
- [Deepak Narayanan et al., Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM, 2021](https://arxiv.org/abs/2104.04473)
- [Dmitry Lepikhin et al., GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding, 2020](https://arxiv.org/abs/2006.16668)
- [Samyam Rajbhandari et al., DeepSpeed-MoE, 2022](https://arxiv.org/abs/2201.05596)
- [Megatron-LM Parallelism Strategies Guide](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/parallelism-guide.md)
- [Megatron Core MoE Guide](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)
- [DeepSpeed Mixture-of-Experts Tutorial](https://www.deepspeed.ai/tutorials/mixture-of-experts/)

# Megatron-LM、DeepSpeed 与 Colossal-AI：大模型分布式训练框架怎么选

Megatron-LM、DeepSpeed 和 Colossal-AI 经常出现在同一张“大模型训练框架”列表里。它们都能做分布式训练，都谈 tensor parallel、pipeline parallel、ZeRO、mixed precision 和 checkpoint，却不是三个完全对位的产品。

最容易理解它们的方法，不是比较谁的功能勾选更多，而是先问：**谁在控制模型结构，谁在控制训练状态，谁在控制并行进程组，谁在控制执行引擎？**

一句话概括：

- **Megatron-LM / Megatron Core** 是 performance-first 的 Transformer 并行内核与参考训练栈，擅长在 NVIDIA GPU 集群上把 TP、PP、DP、CP、EP 和高性能 kernel 深度组合。
- **DeepSpeed** 是 runtime/memory-first 的训练引擎，核心优势是 ZeRO、CPU/NVMe offload、分布式 optimizer states、配置驱动接入，以及与 PyTorch/Hugging Face 生态的广泛整合。
- **Colossal-AI** 是 plugin/transformation-first 的组合式系统，用 Booster、Plugin 和 ShardFormer 把 DDP、ZeRO、TP、PP、SP、mixed precision 和模型替换策略注入现有训练代码。

它们不是简单的“高性能、中性能、易用”三级，也不存在对所有模型都成立的速度排名。模型架构、GPU 拓扑、并行度、sequence length、checkpoint 约束和团队维护能力，往往比框架名字更决定结果。

本文依据 2026 年 8 月 24 日可访问的三方主分支与官方文档撰写。分布式训练框架变化很快，具体模型支持、checkpoint 兼容性和 backend 状态应在部署时固定 revision 重新验证。

![Three distributed training control surfaces](/images/blog/training-framework-control-surfaces.svg "图 1：Megatron-LM、DeepSpeed 和 Colossal-AI 的默认控制面不同。")

## 先区分三个层次：Library、Runtime 与 Training Stack

一个完整大模型训练系统至少包含：

1. 模型结构与 layer implementation；
2. 参数、gradient、optimizer states 和 activation 的分片；
3. DP、TP、PP、EP、CP 等 process groups；
4. forward/backward 与 pipeline schedule；
5. mixed precision、fused kernels 和 communication overlap；
6. dataloader、loss、optimizer、scheduler 与训练 loop；
7. distributed checkpoint、故障恢复和拓扑变更；
8. profiling、logging 和作业运维。

三个项目覆盖这些层次的比例不同。

Megatron Core 更接近专为 Transformer 训练打造的并行计算 library：它提供 tensor-parallel layer、pipeline schedules、distributed optimizer、context/expert parallel、Transformer model components 和 fused execution 路径。Megatron-LM 则在其上提供参考训练脚本、数据处理和完整例子。

DeepSpeed 更接近包裹模型、optimizer 和训练 loop 的 runtime。调用 `deepspeed.initialize()` 后，返回的 engine 接管 backward、step、gradient accumulation、ZeRO state management 和 checkpoint。某些模型并行能力由 DeepSpeed 自己提供，另一些历史上会与 Megatron 等 model-parallel implementation 组合。

Colossal-AI 的 Booster 位于两者之间。它通过一个 Plugin 选择分布式策略，再用 `booster.boost()` 同时转换 model、optimizer、criterion、dataloader 和 scheduler。TP/PP 等模型结构变换由 ShardFormer policy 完成，ZeRO、Gemini、DDP/FSDP 则由不同 plugins 提供。

因此，同样写着“支持 TP+PP+DP”，真正的模型改写方式、schedule ownership 和 checkpoint schema 可能完全不同。

## Megatron-LM 与 Megatron Core 是什么关系

当前 Megatron-LM 仓库明确区分两个组件：

- **Megatron-LM**：包含 Megatron Core 和预配置训练脚本的 reference example；
- **Megatron Core**：面向自定义训练框架的可组合 GPU-optimized building blocks。

这一区分很重要。很多团队说“我们用了 Megatron”，可能指直接运行 `pretrain_gpt.py`，也可能只把 Megatron Core 的 tensor parallel、pipeline schedule、distributed checkpoint 或 transformer layer 接进自己的平台。

Megatron 的起点是模型并行。2019 年的原始论文重点展示了在原生 PyTorch 中给 Transformer 插入少量 collective，实现高效 intra-layer tensor parallel。后续工作把 tensor、pipeline 和 data parallel 组合起来，现代 Megatron Core 又加入 context parallel、expert parallel、sequence parallel、FSDP/distributed optimizer、distributed checkpoint 和 MoE stack。

它的优势来自纵向整合：模型结构知道自己如何切分，parallel state 知道 rank groups，kernel 知道 tensor layout，pipeline scheduler 知道 micro-batches，checkpoint 知道 sharded state。代价也是同一个：自定义模型必须进入这套结构契约，不能只把任意 PyTorch module 套一层 wrapper 就自动获得所有高性能路径。

## DeepSpeed 的核心不是“另一个 Megatron”，而是 ZeRO Runtime

DeepSpeed 最具辨识度的能力是 ZeRO。标准 data parallel 在每张 GPU 上复制 optimizer states、gradients 和 parameters；ZeRO 分阶段消除这些副本：

```text
ZeRO-1: shard optimizer states
ZeRO-2: shard optimizer states + gradients
ZeRO-3: shard optimizer states + gradients + parameters
```

ZeRO-3 在 forward/backward 需要时 gather 参数，再重新 partition。ZeRO-Infinity 进一步把状态 offload 到 CPU 或 NVMe。它解决的第一问题是“现有模型如何装进有限显存”，而不是先要求用户把每个 Linear 改成 tensor-parallel layer。

这也是 DeepSpeed 对研究代码友好的原因。对于只需要 ZeRO 的普通 PyTorch/Hugging Face 模型，主要改动通常集中在初始化、engine 调用和 JSON config；ZeRO 教程强调不需要改模型代码。但一旦进入 pipeline parallel、自定义 tensor parallel、MoE 或复杂 checkpoint 拓扑，模型结构和 runtime 约束仍然会出现，不能把“几行配置”泛化到所有场景。

DeepSpeed 还提供 PipelineModule、AutoTP、AutoEP、Ulysses/long-sequence training、MoE、communication logging、Flops Profiler、offload 和 Universal Checkpointing。它覆盖的硬件与集成范围也更广；当前仓库 CI 列出了 NVIDIA、AMD、CPU、Intel Gaudi、Intel XPU、Ascend 和 Hugging Face/Accelerate 等 integration。但具体 op、性能路径和 feature parity 必须逐 backend 核对，CI 列表不等于所有功能在所有设备上完全一致。

## Colossal-AI 的核心是 Booster、Plugin 与 ShardFormer

Colossal-AI 希望让用户保留熟悉的 PyTorch/Hugging Face 模型与训练组件，再通过统一入口注入分布式能力：

```python
plugin = HybridParallelPlugin(
    tp_size=2,
    pp_size=4,
    zero_stage=1,
    precision="bf16",
)

booster = Booster(plugin=plugin)
model, optimizer, criterion, dataloader, scheduler = booster.boost(
    model, optimizer, criterion, dataloader, scheduler
)
```

Booster 是统一 facade，Plugin 决定底层策略。官方文档列出的主要 plugins 包括 TorchDDPPlugin、TorchFSDPPlugin、LowLevelZeroPlugin、GeminiPlugin 和 HybridParallelPlugin。

其中：

- LowLevelZeroPlugin 提供 ZeRO-1/2；
- GeminiPlugin 提供 chunk-based ZeRO-3 与 heterogeneous memory management；
- HybridParallelPlugin 组合 TP、PP、DP/DDP/ZeRO、mixed precision 和多种 kernel 优化；
- ShardFormer 按 policy 把原生 modules 替换成 distributed modules，并改写必要的 forward 行为。

这套设计的优点是入口统一、Python 组合感强。用户可以按场景更换 plugin，而不必重写整个上层训练框架。限制则来自 transformation coverage：如果目标 Hugging Face 模型已有 ShardFormer policy，接入会比较顺；如果是新的自定义架构，就需要写 policy、定义 module replacements，并验证 tied weights、forward signature、checkpoint 和 pipeline semantics。

## 三种默认工作流

三者的典型代码形态可以简化为：

### Megatron-LM：先进入并行模型规范

```text
define TransformerConfig and model provider
initialize Megatron parallel state
construct tensor/pipeline/expert-parallel modules
run Megatron training loop and schedules
save distributed checkpoint
```

用户的主要工作是把模型架构、数据和训练目标放入 Megatron contract。并行策略从一开始就是模型定义的一部分。

### DeepSpeed：用 Engine 接管训练状态

```python
engine, optimizer, dataloader, scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json",
)

loss = engine(batch)
engine.backward(loss)
engine.step()
```

用户保留 PyTorch model，engine 接管 optimizer state、gradient accumulation、ZeRO、mixed precision 和 checkpoint 等 runtime 行为。

### Colossal-AI：用 Plugin 选择注入策略

```python
plugin = GeminiPlugin(...)  # or HybridParallelPlugin / FSDP / ZeRO
booster = Booster(plugin=plugin)
model, optimizer, criterion, dataloader, scheduler = booster.boost(...)

loss = criterion(model(batch))
booster.backward(loss, optimizer)
optimizer.step()
```

用户以 Plugin 作为策略边界。Booster 对外统一 API，Plugin 与 ShardFormer 决定内部改写。

![Framework training paths](/images/blog/training-framework-execution-paths.svg "图 2：三套框架从用户模型到分布式执行的典型路径。")

## 并行能力不能只看勾选表

下面这张表描述“主路径”，不是声明某框架绝对不支持其他方式：

| 能力 | Megatron-LM / Core | DeepSpeed | Colossal-AI |
| --- | --- | --- | --- |
| DP / sharded DP | DDP、distributed optimizer、Megatron-FSDP | ZeRO-1/2/3 是核心路径 | DDP、FSDP、LowLevelZero、Gemini |
| TP | 原生 Transformer TP，深度集成 | AutoTP 或与外部 model-parallel stack 组合 | ShardFormer policy 注入 TP |
| PP | 原生 schedules、virtual/interleaved pipeline | PipelineModule / PipelineEngine | HybridParallelPlugin + pipeline manager |
| CP / 长序列 | 原生 Context Parallel | Ulysses、ALST 等路径 | Sequence/hybrid parallel，具体模型需核对 |
| EP / MoE | 原生 Megatron Core MoE 与 EP stack | DeepSpeed-MoE、AutoEP | ColossalMoE/相关模块，集成覆盖需按模型验证 |
| Offload | activation/optimizer 等路径，偏 NVIDIA stack | CPU/NVMe offload 是核心优势 | Gemini heterogeneous memory、ZeRO offload |
| HF 模型接入 | Megatron Bridge 转换与 recipes | Transformers/Accelerate integrations | Booster + ShardFormer policies |

“支持 TP”可能表示模型本身已经用 parallel linear 实现，也可能表示 runtime 自动替换模块；“支持 checkpoint resharding”可能只覆盖 weights，也可能覆盖 optimizer states；“支持 Hugging Face”可能只表示能加载权重，不表示任意模型都能做 pipeline parallel。

所以选型时应该把功能问题改写成具体 contract：目标模型的哪些 modules 会被替换？tied parameters 怎样处理？optimizer states 怎样 shard？从 `TP=8` 改成 `TP=4` 能否带 optimizer 恢复？MoE expert weights 能否改变 EP size？这些问题比官网首页的功能名更有区分度。

## Megatron 的 Parallelism 优势

Megatron Core 当前官方并行指南把 DP、TP、PP、CP、EP 和 FSDP 列为可组合策略。它最强的地方不是策略数量，而是这些策略共享统一 parallel state 和 Transformer execution model。

TP 能深入到 QKV、attention output 和 MLP；SP 与 TP process group 配套；PP 有 schedules 与 virtual stages；CP 围绕 attention KV exchange；EP 有 router、token dispatcher、grouped GEMM、load balancing 和通信 overlap；distributed optimizer/FSDP 处理 data-parallel state sharding。

在大规模 NVIDIA 集群上，这种统一性有利于：

- 把 TP/EP 放进 NVLink/NVSwitch 域；
- 对 DP、TP、PP、EP 通信做细粒度 overlap；
- 使用 Transformer Engine 的 BF16/FP8/FP4 与 fused kernels；
- 对 MoE、long context 和异构 parallel mapping 做专门优化；
- 让 profiling 指标直接对应 Megatron process groups。

但 Megatron 的学习曲线也来自这里。模型不是黑盒。新增 attention、Mamba-like block、multimodal connector、特殊 loss 或非标准 pipeline boundary 时，可能需要理解 ModuleSpec、parallel layers、tensor layouts、pipeline IO、RNG tracking、distributed optimizer 和 checkpoint mapping。

## DeepSpeed 的 Memory 优势

DeepSpeed 最稳固的选型理由通常不是“它也支持 TP”，而是：现有模型太大，需要用 ZeRO 和 offload 尽快跑起来。

ZeRO-1/2 对训练代码侵入较低，适合保留较大 compute granularity；ZeRO-3 进一步分 parameters，显存节省更大，却在每层引入 parameter gather/partition，对 bucket、prefetch、overlap、module granularity 和网络更敏感。CPU/NVMe offload 用更慢介质换 GPU 显存，能让模型可运行，但不保证高 MFU。

JSON config 让 memory strategy 很容易实验：

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "offload_optimizer": {"device": "cpu"}
  }
}
```

它也带来一类特有风险：配置文件、launcher arguments、Transformers TrainingArguments 和代码默认值可能同时决定同一参数。`auto`、global batch、micro-batch、gradient accumulation 和 scheduler step 如果在多层配置中不一致，训练可能能启动却语义错误。

因此 DeepSpeed 项目需要在启动时打印并归档 resolved config，而不是只保存输入 JSON。

## Colossal-AI 的 Composability 优势

Colossal-AI 的吸引力在于它把策略选择做成 Plugin。对一个现有模型，可以先用 TorchDDPPlugin 建立基线，再切 LowLevelZeroPlugin、GeminiPlugin 或 HybridParallelPlugin，而训练 loop 的高层结构相对稳定。

ShardFormer 进一步把“模型如何并行化”显式封装成 policy。它会替换原生 modules、parameters 或 methods，并可注入 fused normalization、FlashAttention、sequence parallel 等优化。相比完全手写 parallel model，policy 更适合复用；相比纯 runtime 黑盒，它又保留了明确的模型变换层。

这种设计特别适合研究系统原型：团队想试验新的 sharding policy、pipeline split、sequence parallel 或 heterogeneous memory，而不想完全进入 Megatron 训练栈。它同样适合已经围绕 Hugging Face/PyTorch 组织模型代码的项目。

不过，policy coverage 是实际边界。官方文档本身提醒 HybridParallelPlugin 的 TP/PP 只对 ShardFormer 支持的模型子集有效。模型家族、具体 revision、custom remote code 和新 attention implementation 都可能让静态“支持列表”失效。上线前必须用目标 checkpoint 和目标 forward path 做小规模收敛与 save/load 测试。

## Pipeline Parallel 的差异尤其大

PP 是最不能仅凭开关比较的功能。

Megatron 把 pipeline stage 看作其模型结构和 schedule 的原生组成，支持 1F1B、interleaving、virtual pipeline、stage layout 和 distributed loss 语义。

DeepSpeed PipelineModule 要求模型表达为 layer sequence；训练由 PipelineEngine 的 `train_batch()` 推进，因为 forward、backward 和 step 已被 schedule 交错。复杂模型可能需要重构 input/output，把 mask 等状态沿 stages 传递。

Colossal-AI HybridParallelPlugin 通过 ShardFormer 和 pipeline manager 切模型；启用 PP 时，训练 loop 需要调用 `booster.execute_pipeline()`，并明确 criterion 与 last-stage 行为。

这三个 API 都说明一个事实：pipeline parallel 必然改变 execution ownership。把已有训练 loop 里普通的 `loss.backward()` 换成一个配置开关，通常不足以正确处理 micro-batches、stage IO、loss、gradient accumulation 和 evaluation。

## Checkpoint 才是长期成本中心

训练第一次跑通只是开始。真正决定系统能否持续维护的，是 checkpoint 是否支持故障恢复、并行拓扑变化、模型转换和版本升级。

Megatron Core 的 distributed checkpoint 使用 sharded state dict；官方文档说明 weights 可在不同 tensor/pipeline/data parallel 配置间加载。但 optimizer state 还有格式和版本边界，例如某些快速格式不支持任意 model-parallel reshard，需要切换 fully reshardable format；旧 optimizer checkpoint 也可能需要特定版本中转转换。

DeepSpeed 常规 checkpoint 与 ZeRO partition 强相关。Universal Checkpointing 的流程通常是先保存常规 ZeRO checkpoint，再运行转换脚本得到 Universal format，然后在目标拓扑加载。当前文档还分别说明 AutoTP/AutoEP 的 metadata、ZeRO stage 与 EP size 限制。它不是“任意 checkpoint 自动兼容任意模型”。

Colossal-AI Booster 提供统一 save/load API，支持 sharded model checkpoint，并说明其 sharded weight 格式兼容 Hugging Face `from_pretrained`。但 model weights 可加载，不自动意味着 optimizer states、PP layout 和所有 plugin 切换都可无损恢复。

一个生产训练项目至少要提前验证四条路径：

1. 同一配置中断后恢复；
2. 改变 DP size 后恢复；
3. 改变 TP/PP/EP 后恢复；
4. 导出成标准 Hugging Face 或目标推理引擎格式。

没有这些演练，框架迁移成本会在训练跑了几周后才暴露。

## Hugging Face 兼容意味着什么

“兼容 Hugging Face”至少有四个层次：

```text
L1: 能读取 config 和 weights
L2: 能包装原生 HF model 做 DP/ZeRO
L3: 能自动把目标 model 变成 TP/PP/EP model
L4: 能把分布式 checkpoint 稳定导回标准 HF 格式
```

DeepSpeed 在 L2 上通常最直接，Transformers Trainer 与 Accelerate 都有集成。AutoTP/AutoEP 尝试覆盖 L3，但仍有 model list 与结构要求。

Colossal-AI 的 Booster 很适合 L2，ShardFormer policies 负责 L3；policy 不存在时需要自己实现。Booster sharded model checkpoint 的 HF 兼容有利于 L4，但复杂 optimizer/topology 仍需单独验证。

Megatron Bridge 负责 Hugging Face 与 Megatron checkpoint 的双向转换和 recipes。进入 Megatron Core 后，模型通常不再是未经修改的 HF module；L3 更接近显式 Megatron implementation，而不是运行时自动包装。

所以“我们代码基于 Transformers”并不能直接决定框架。还要问：只做 ZeRO finetuning，还是要把模型结构真正切成 TP/PP/EP？

## 三者可以混用吗

可以，但需要明确 ownership。历史上 Megatron-DeepSpeed 就把 Megatron model parallel 与 DeepSpeed ZeRO/runtime 组合，DeepSpeed pipeline 文档也明确提到可以与 Megatron-LM model parallel 结合。

组合的合理方式是让不同系统各自拥有清晰边界，例如：

```text
Megatron owns: TP model layers and model-parallel groups
DeepSpeed owns: ZeRO optimizer states and data-parallel runtime
```

危险方式是两个系统同时初始化相同 process groups、同时 wrap optimizer、同时管理 gradient accumulation、同时保存 checkpoint。此时最常见的问题不是立刻 crash，而是 batch 语义、loss scaling、scheduler steps 或 checkpoint states 重复处理。

Colossal-AI 也可以复用 PyTorch、Transformers、FlashAttention 等组件，但不应再把同一 model 同时交给另一个会重写 module graph 的 runtime，除非集成路径有明确支持和测试。

现代项目里，优先选择一套训练 control plane，再通过它官方支持的桥接层接入其他组件，通常比自由叠加三个框架更可维护。

## 调试体验的差异

### Megatron-LM

常见故障集中在 rank mapping、parallel group、tensor shape、pipeline stage、distributed optimizer 和 checkpoint metadata。错误可能非常底层，但一旦理解 parallel state，性能问题通常能映射到明确通信维度。

### DeepSpeed

常见故障集中在 JIT ops、PyTorch/CUDA/ROCm 版本、ZeRO bucket、offload、launcher environment、JSON resolved values 和外部集成覆盖。它容易让模型先跑起来，但 performance tuning 需要把 runtime 自动行为重新展开。

### Colossal-AI

常见故障集中在 Plugin 约束、ShardFormer policy coverage、module replacement、pipeline IO 和 checkpoint plugin compatibility。抽象清楚时调试很顺；遇到新模型时，需要从 policy 到 distributed module 追踪变换。

无论使用哪套框架，都应该保存：完整 resolved config、package revisions、NCCL/environment、rank mapping、模型 config、global/micro batch、并行度、checkpoint schema、首个 step profiler 和小规模 loss 对齐结果。

## 不要用官网 Benchmark 直接决定框架

三个项目都会展示高吞吐、大模型和大集群 benchmark。这些数字证明框架在特定配置上可以工作，但不能直接构成横向排名，因为它们往往使用不同：

- 模型大小、层数、hidden size 和 vocabulary；
- sequence length、global batch 和 micro-batch；
- GPU 型号、节点互联和集群规模；
- BF16、FP8 或其他精度；
- activation recomputation 与 offload；
- 是否包括 dataloader、optimizer、logging 和 checkpoint；
- strong scaling 或 weak scaling口径；
- 收敛训练或只做性能压测。

选型 benchmark 应该在自己的模型、数据 shape 和集群上做。至少建立同 loss、同有效 token、同 global batch、同 precision 和同 checkpoint 要求的对照，再比较 tokens/s、MFU、peak memory、step p95 和恢复时间。

## 选型矩阵

![Distributed training framework selection map](/images/blog/training-framework-selection-map.svg "图 3：根据模型控制权、显存压力、硬件与并行复杂度选择起点。")

| 场景 | 更自然的起点 | 原因 |
| --- | --- | --- |
| NVIDIA 集群上从头预训练 70B+ dense Transformer | Megatron Core | TP/PP/CP/DP 与 kernel 深度整合 |
| NVIDIA 集群上训练大规模 MoE | Megatron Core | EP、dispatcher、grouped GEMM、parallel folding 路径完整 |
| 现有 Hugging Face 模型做 ZeRO finetuning | DeepSpeed | runtime 包装直接，Transformers/Accelerate 集成成熟 |
| 单机或小集群显存不足，需要 CPU/NVMe | DeepSpeed | ZeRO-Offload/Infinity 是核心设计路径 |
| 多种 accelerator backend 候选 | DeepSpeed 优先评估 | 官方 CI 覆盖更广，但需验证具体 ops |
| 想在 HF 模型上研究新的 sharding/plugin 策略 | Colossal-AI | Booster、Plugin、ShardFormer 的策略边界清晰 |
| 想在统一 API 下切换 DDP、FSDP、ZeRO、Gemini | Colossal-AI | Plugin 模型适合快速实验 |
| 高度自定义非 Transformer 网络 | DeepSpeed ZeRO 或 Colossal Booster | 比进入 Megatron model contract 更自然 |
| 已有成熟 Megatron 模型，但 optimizer memory 不足 | Megatron distributed optimizer/FSDP，或验证官方 DeepSpeed 集成 | 先避免重复 control plane |

这张表只给“起点”，不是最终结论。比如 DeepSpeed 也能做大规模预训练，Colossal-AI 也能做 hybrid parallel，Megatron Core 也已有 FSDP 和 Hugging Face bridge。真正的选择取决于哪条路径对目标模型是第一方、经过持续测试的主路径。

## 一个分阶段选型方法

### 阶段一：定义不可妥协约束

先写清楚模型是否为 dense/MoE、总参数和 active parameters、sequence length、目标 GPU、节点拓扑、最大训练规模、必须支持的 checkpoint 拓扑变化、是否依赖 Hugging Face custom code。

### 阶段二：选择 control plane

需要模型并行性能与 NVIDIA 深度优化，优先用 Megatron Core；需要低侵入 ZeRO/offload，优先用 DeepSpeed；需要 plugin 化实验与模型 transformation，优先用 Colossal-AI。不要一开始就组合两套 engine。

### 阶段三：做 1 卡与 8 卡正确性基线

对齐初始 weights、单步 loss、gradient norm、optimizer update 和固定 batch 的若干 step loss。然后验证 save/load 后继续训练与不中断 run 一致。

### 阶段四：做最小 hybrid-parallel 原型

只启用必需并行维度。例如先 DP/ZeRO，再 TP，再 PP，最后 EP/CP。每增加一维就记录 peak memory、tokens/s、MFU、通信占比和 loss 对齐。

### 阶段五：验证拓扑变化与导出

在投入长训练前，主动从一个并行配置保存，再用另一个配置加载；完成标准格式导出和目标推理引擎读取。Checkpoint 不通过，就不能把框架称为生产可用。

### 阶段六：固定版本

记录 Git revision、container digest、PyTorch/CUDA/ROCm、Transformer Engine、FlashAttention、NCCL 和编译 flags。三个项目都在快速演进，只写包名而不写 revision，无法复现实验。

## 我会怎样选择

如果目标是在 H100/H20/B200 等 NVIDIA 集群上训练一个标准或主流变体的大型 dense/MoE Transformer，并且团队愿意维护并行模型代码，我会从 Megatron Core 开始。它把最关键的高频通信和 Transformer kernel 放在同一个设计中，最适合追求规模与 MFU。

如果目标是尽快把已有 PyTorch/Hugging Face 模型做多卡 finetuning，主要矛盾是 optimizer/parameter memory，或者必须用 CPU/NVMe 扩展容量，我会先用 DeepSpeed ZeRO。它把最常见的显存问题变成 runtime 配置，验证成本低于重写模型。

如果目标是研究并行策略本身、需要在多种 plugins 之间切换、希望对 Hugging Face model 做显式但可复用的 module transformation，我会认真评估 Colossal-AI。Booster 与 ShardFormer 的抽象比“一堆训练 flags”更适合组织实验代码。

我不会因为某个框架官网展示了最高 MFU 就直接迁移，也不会因为另一个框架只需几行代码就假设它适合千卡预训练。选型应该由目标模型的 first-class path、checkpoint contract、硬件 topology 和团队调试能力共同决定。

## 总结

Megatron-LM、DeepSpeed 与 Colossal-AI 的核心差异，是默认控制面不同。

Megatron Core 从 Transformer 模型结构和 process groups 出发，把 TP、PP、CP、EP、DP、FSDP、MoE 与 kernel 深度整合，适合 NVIDIA 大规模预训练和追求高 MFU 的团队；代价是模型与训练代码要进入 Megatron contract。

DeepSpeed 从训练 runtime 和模型状态内存出发，用 ZeRO、offload、Engine、PipelineModule 和配置系统扩展现有 PyTorch 模型，适合 Hugging Face finetuning、显存受限训练和更广硬件生态；代价是自动 runtime 行为与多层配置需要严格审计。

Colossal-AI 从 Booster、Plugin 与 ShardFormer transformation 出发，强调在统一 API 下组合 DDP、FSDP、ZeRO、Gemini、TP、PP 和优化策略，适合研究型框架、Hugging Face 模型改写与快速策略实验；代价是实际能力取决于目标模型的 policy 与 plugin coverage。

最实用的判断不是“哪个框架最好”，而是：**哪套系统应该拥有你的模型图、optimizer、parallel groups、pipeline schedule 和 checkpoint。** 先选唯一 control plane，再验证目标模型的正确性、性能和恢复路径，才是可维护的大模型训练工程。

## 参考资料

- [NVIDIA Megatron-LM / Megatron Core](https://github.com/NVIDIA/Megatron-LM)
- [Megatron Core Parallelism Strategies Guide](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/parallelism-guide.md)
- [Megatron Core Distributed Checkpointing](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/core/dist_checkpointing.md)
- [Mohammad Shoeybi et al., Megatron-LM, 2019](https://arxiv.org/abs/1909.08053)
- [Microsoft DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/)
- [DeepSpeed Pipeline Parallelism](https://www.deepspeed.ai/tutorials/pipeline/)
- [DeepSpeed Universal Checkpointing](https://www.deepspeed.ai/tutorials/universal-checkpointing/)
- [Samyam Rajbhandari et al., ZeRO, 2019](https://arxiv.org/abs/1910.02054)
- [HPC-AI Tech Colossal-AI](https://github.com/hpcaitech/ColossalAI)
- [Colossal-AI Booster API](https://colossalai.org/docs/basics/booster_api/)
- [Colossal-AI Booster Plugins](https://colossalai.org/docs/basics/booster_plugins/)
- [Colossal-AI ShardFormer](https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/shardformer)
- [Shenggui Li et al., Colossal-AI, 2021](https://arxiv.org/abs/2110.14883)

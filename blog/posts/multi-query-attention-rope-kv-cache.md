# Multi-Query Attention 与 RoPE：从 KV Cache 带宽到旋转位置编码

Multi-Query Attention（MQA，多查询注意力）和 Rotary Position Embedding（RoPE，旋转位置编码）经常同时出现在现代大语言模型的架构说明里。它们都作用于 attention，却解决完全不同的问题：MQA 改变 query head 与 key/value head 的数量关系，目标是减少自回归解码时的 KV Cache 和内存带宽；RoPE 改变 query 与 key 携带位置的方式，目标是让注意力分数自然依赖 token 之间的相对距离。

一句话概括：**MQA 决定“要缓存多少组 K/V”，RoPE 决定“Q/K 如何知道自己处在什么位置”。** 一个模型可以同时使用两者，也可以只使用其中一个。

这篇文章从标准 Multi-Head Attention（MHA）出发，逐步解释 MQA 为什么能加速逐 token 解码、它没有减少什么、Grouped-Query Attention（GQA）为什么成为常见折中，以及 RoPE 如何通过二维旋转把相对位置带入点积。最后再把二者放回真实推理系统，讨论 KV Cache、长上下文和实现边界。

这里还需要先厘清原始引文的证据边界。题述段落直接来自 PaLM 论文的架构部分：PaLM 作者写的是“我们发现”MQA 对其模型质量和训练速度影响中性，并在括号中引用 Shazeer 2019。Shazeer 的 MQA 原论文则报告解码明显加速、相对 MHA 只有轻微质量退化。因此，更准确的读法是：**PaLM 在自己的规模和训练配方上观察到近似质量中性；MQA 本身并不保证对所有模型和任务都完全无损。**

![MHA, GQA, and MQA head sharing](/images/blog/mqa-gqa-head-sharing.svg "图 1：MHA、GQA 与 MQA 的 query head 和 KV head 对应关系。")

## 先统一符号与张量形状

设输入隐藏状态为 `X`，它的形状是：

```text
X: [B, T, d_model]
```

其中：

- `B` 是 batch size；
- `T` 是当前参与计算的序列长度；
- `d_model` 是模型隐藏维度；
- `H_q` 是 query head 数；
- `H_kv` 是 key/value head 数；
- `d_h` 是每个 head 的维度。

线性投影以后，更完整的张量形状应该写成：

```text
Q: [B, H_q,  T, d_h]
K: [B, H_kv, T, d_h]
V: [B, H_kv, T, d_h]
```

很多架构介绍把单个时间步、batch 维和序列维省略，只写成 `Q:[k,h]`、`K/V:[1,h]`。这种简写可以表达 head 共享关系，但容易让人误以为 K/V 只有一个向量。实际上，每一层、每个样本、每个历史位置仍然有自己的 K/V；MQA 共享的是**同一个样本、同一层中不同 query heads 所使用的 KV head**，不是让 batch 里的不同样本共享内容。

三个常见变体只是在 `H_kv` 上不同：

```text
MHA: H_kv = H_q
GQA: 1 < H_kv < H_q
MQA: H_kv = 1
```

如果一个模型有 32 个 query heads：标准 MHA 通常有 32 组 K/V；GQA 可以只有 8 组或 4 组 K/V；MQA 则只有 1 组 K/V。Query 仍然保留 32 个 heads。

## 标准 Multi-Head Attention 做了什么

在标准 MHA 中，输入分别乘以三组投影矩阵：

```text
Q = X W_Q
K = X W_K
V = X W_V
```

每个 query head 都有与之对应的 key head 和 value head。第 `i` 个 head 的输出可以写成：

```text
head_i = softmax(Q_i K_i^T / sqrt(d_h)) V_i
```

多个 head 可以在不同子空间里学习不同关系：某些 head 更关注局部语法，某些关注指代，某些关注跨段依赖。完成各 head 的 attention 后，结果被拼接并经过输出投影。

训练或 prompt prefill 时，序列中的很多 token 可以并行计算，矩阵乘法规模也足够大，GPU/TPU 容易获得较高利用率。可是自回归 decode 不一样：模型每一步只产生一个新 token，新的 query 长度为 1，却要读取从第一个 token 到当前位置的全部历史 K/V。

在第 `t` 步，单层张量大致变为：

```text
Q_new:   [B, H_q,  1, d_h]
K_cache: [B, H_kv, t, d_h]
V_cache: [B, H_kv, t, d_h]
```

新的 Q 很小，历史 KV Cache 很大。计算设备做的算术量相对有限，但必须从高带宽内存反复搬运越来越长的 K/V。这正是增量解码经常受内存带宽约束，而不是只受峰值 FLOPs 约束的原因。

## MQA 的核心：多个 Query Heads 共用一组 K/V

Shazeer 在 2019 年提出 MQA 时，保留多组 query heads，只把 key 和 value 收缩成单组：

```text
Q: [B, H_q, T, d_h]
K: [B, 1,   T, d_h]
V: [B, 1,   T, d_h]
```

计算第 `i` 个 query head 时，它不再读取专属的 `K_i` 和 `V_i`，而是与其他所有 query heads 共用 `K_shared` 和 `V_shared`：

```text
head_i = softmax(Q_i K_shared^T / sqrt(d_h)) V_shared
```

不同 query heads 仍然可以产生不同 attention distribution，因为 `Q_i` 不同；但它们检索的是同一套 key 表示，并从同一套 value 表示中聚合信息。

这里的“multi-query”不是一次输入多条自然语言查询，而是**多个 attention query heads 对同一组 K/V 发起查询**。名字描述的是 head 拓扑，不是产品层面的多查询搜索。

## 为什么 MQA 主要加速 Decode，而不是神奇地消除 Attention 计算

MQA 最直接的收益有三个。

第一，K/V 投影参数和投影计算更少。标准 MHA 的 `W_K`、`W_V` 通常输出 `H_q × d_h` 个通道；MQA 只输出 `1 × d_h` 个通道。不过在完整 Transformer 中，FFN 和其他投影仍占大量参数，因此这不等于整个模型缩小 `H_q` 倍。

第二，KV Cache 显著缩小。对 decoder-only Transformer，忽略额外元数据时，KV Cache 字节数近似为：

```text
KV bytes = 2 × L × B × S × H_kv × d_h × bytes_per_element
```

`2` 代表 K 和 V，`L` 是层数，`S` 是已缓存序列长度。其他条件不变时，缓存大小与 `H_kv` 成正比。因此从 32 个 KV heads 减少到 1 个，理论上的 K/V 主体容量也减少到三十二分之一。

第三，每个 decode step 读取的历史 K/V 更少。因为逐 token 解码常常是 memory-bandwidth-bound，减少缓存读取可以明显提高 token throughput，也能让同一设备容纳更多并发请求或更长上下文。

但 MQA **没有**把 query heads 合成一个，也没有让所有 attention score 只算一次。每个 query head 仍要形成自己的 `Q_i K^T` 和 softmax，attention score 的主要计算仍与 `H_q × S × d_h` 有关。它优化得最彻底的是 K/V 的生成、存储和读取，而不是把 attention 的全部算术复杂度从根本上消掉。

这也是为什么 MQA 在训练和 prefill 阶段未必带来同等幅度的提速。长序列 prefill 中仍然要计算大量 query-key 配对，通常依赖大规模并行矩阵乘法；decode 中则只有一个新 query，KV Cache 带宽更容易成为主瓶颈。真实收益还取决于 kernel、batching、量化、张量并行和硬件内存系统。

## 一个 KV Cache 数量级例子

考虑一个纯示意配置：

```text
层数 L = 32
Query heads H_q = 32
Head dim d_h = 128
上下文 S = 32,768 tokens
Batch B = 1
缓存类型 = BF16/FP16，每个元素 2 bytes
```

按前面的公式计算：

| Attention 类型 | `H_kv` | 单序列 KV Cache 主体 |
| --- | ---: | ---: |
| MHA | 32 | 16 GiB |
| GQA | 8 | 4 GiB |
| GQA | 4 | 2 GiB |
| MQA | 1 | 0.5 GiB |

![KV Cache scaling with KV heads](/images/blog/mqa-kv-cache-scaling.svg "图 2：在固定层数、上下文和 head dimension 时，KV Cache 随 KV head 数线性变化。")

这个例子不是任何具体模型的实测显存报告。真实服务还会受到 KV Cache 分页、对齐、碎片、量化、跨卡切分、并发调度和框架元数据影响。但它准确展示了 MQA 的主要缩放规律：**当长上下文和并发数把 KV Cache 推成系统瓶颈时，减少 `H_kv` 的价值会迅速放大。**

## “不同样本之间不共享 K/V”该怎么理解

有些介绍会说标准 MHA 效率低，是因为 K/V 在 examples 之间不共享。这个表述需要拆开。

MQA 带来的结构共享发生在 **attention heads 之间**：同一序列中的多个 query heads 共用一组 K/V。不同 batch 样本通常仍然有不同 token 历史，所以不能因为使用 MQA 就共享语义内容。只有当多个请求拥有完全相同的前缀，并且服务系统实现了 prefix caching 或 KV reuse 时，才可能额外共享那部分前缀缓存；这是 serving 层优化，不是 MQA 定义本身。

逐 token 解码效率低的更准确原因是：每一步的新 query 很短，而每个请求的历史 K/V 很长；标准 MHA 还为每个 query head 保留独立的 K/V，导致每一步需要搬运大量缓存。MQA 用 head-level sharing 缩小了这批数据。

## 质量为什么可能下降

减少 KV heads 会形成信息瓶颈。在 MHA 中，每个 head 不仅有自己的 query 投影，还有自己的 key/value 表示；在 MQA 中，所有 query heads 必须共用同一套 K/V 子空间。虽然不同 Q 仍能选择不同位置，但 value 表示的多样性减少了。

因此，“MQA 对模型质量完全中性”不应该被当作无条件结论。Shazeer 2019 的原始论文报告的是明显解码加速和较小的质量退化；后续不同模型、不同规模和不同训练配方可能观察到接近中性的结果，也可能看到更明显差距。质量影响需要在目标模型和任务上验证。

这个问题促成了 GQA：不要在 `H_kv = H_q` 和 `H_kv = 1` 之间二选一，而是保留少量 KV groups。

## GQA：MHA 与 MQA 之间的工程折中

Grouped-Query Attention 让多个 query heads 组成一组，每组共享一个 KV head。例如：

```text
H_q = 32
H_kv = 8
每 4 个 query heads 共用 1 个 KV head
```

对第 `i` 个 query head，可以把它映射到：

```text
kv_group(i) = floor(i / (H_q / H_kv))
```

GQA 保留了比 MQA 更丰富的 K/V 表示，同时仍然显著减少缓存。Ainslie 等人在 2023 年系统讨论了如何把已有 MHA checkpoint 用少量额外预训练计算 uptrain 成 MQA/GQA，并报告 GQA 能取得接近 MHA 的质量和接近 MQA 的推理速度。

因此，现代架构里经常看到的不是纯 MQA，而是 `H_q > H_kv > 1` 的 GQA。它并不否定 MQA，而是把 MQA 的 head sharing 从“所有 query heads 一组”推广成“若干 query heads 一组”。

## 实现时不要真的复制 K/V

概念上，我们可以把单组 K/V 扩展到所有 query heads；实现上却不应该无条件执行物理 `repeat`，否则刚节省的内存带宽又可能被复制操作吃掉。

简化的伪代码如下：

```python
# x_new: [B, 1, d_model]
q = project_q(x_new).view(B, H_q, 1, d_h)
k = project_k(x_new).view(B, H_kv, 1, d_h)
v = project_v(x_new).view(B, H_kv, 1, d_h)

q, k = apply_rope(q, k, position_id)
kv_cache.append(k, v)

# Kernel 在逻辑上把每个 KV group 广播给对应 query heads，
# 不一定物理复制完整的历史 K/V。
output = grouped_attention(q, kv_cache, H_q, H_kv)
```

高效 attention kernel 需要直接理解 `H_q/H_kv` 的分组关系。框架兼容性、张量布局和并行切分都会影响 MQA/GQA 的真实速度；只在高层代码中把 K/V `repeat_interleave` 成 MHA 形状，不足以证明获得了预期的带宽收益。

## RoPE 解决的是另一个问题：Attention 如何感知位置

Self-attention 本身对 token 排列是置换等变的。如果不给位置信息，模型只知道有哪些 token，不知道谁在前、谁在后。早期 Transformer 直接把绝对位置向量加到 token embedding 上；相对位置方法则修改 attention score，使它显式依赖两个 token 的距离。

RoPE 采取了一个很漂亮的中间路线：它根据绝对位置分别旋转 Q 和 K，但两者做点积以后，结果只通过相对位置差进入。

注意，RoPE 通常作用于 **Q 和 K**，而不是 V。它改变的是“当前位置如何匹配历史位置”，不是被聚合内容本身。

![RoPE rotation and relative positions](/images/blog/rope-relative-rotation.svg "图 3：RoPE 对 Q/K 分别施加位置旋转，点积中的旋转角最终只依赖相对位置差。")

## 从二维旋转理解 RoPE

把一个 head 的偶数维和奇数维两两配对。对其中一对数值 `[a, b]`，在位置 `m` 上施加角度 `mθ` 的二维旋转：

```text
[a']   [ cos(mθ)  -sin(mθ) ] [a]
[b'] = [ sin(mθ)   cos(mθ) ] [b]
```

对不同维度对使用不同频率 `θ_i`，常见形式是：

```text
θ_i = base^(-2i / d_h)
```

低频维度旋转得慢，适合表达较长尺度；高频维度旋转得快，适合区分较近位置。整个 head dimension 被拆成许多二维平面，每个平面按自己的频率旋转。

设原始 query 位于位置 `m`，原始 key 位于位置 `n`：

```text
q_m' = R_m q
k_n' = R_n k
```

旋转后的点积满足：

```text
(q_m')^T k_n'
= q^T R_m^T R_n k
= q^T R_(n-m) k
```

关键就在最后一行：虽然 Q 和 K 分别使用绝对位置 `m`、`n` 做旋转，attention score 中出现的却是 `n - m`。这使 RoPE 同时具备绝对位置操作形式和相对位置交互性质。

## 一个直觉例子

假设 query 在位置 100，要比较位置 90 和位置 99 的两个 key。对同一个频率维度：

```text
query 与 key_90 的相对旋转：90 - 100 = -10
query 与 key_99 的相对旋转：99 - 100 = -1
```

模型不需要额外查一张“距离为 10”的 embedding 表。Q/K 的旋转点积已经让不同相对距离产生不同匹配模式。多组频率共同工作，就像用不同刻度的表盘描述距离。

RoPE 还有几个工程优点：没有随最大长度增长的位置 embedding 参数表；cos/sin 可以预计算；它直接融入 Q/K，容易与常规 attention kernel 结合；在训练长度附近和适当扩展策略下，通常比简单绝对位置 embedding 更适合长序列。

## RoPE 不等于无限长上下文

“RoPE 在长序列上表现更好”不意味着把 position id 任意增大，模型就会可靠理解无限上下文。

模型训练时只见过有限的位置与相对距离。超过训练窗口后，旋转相位分布会发生变化，不同频率可能出现混叠，attention pattern 也会落到训练分布之外。因此实践中会出现 position interpolation、NTK-aware scaling、YaRN 等 RoPE scaling 方法，用不同方式调整位置或频率。

这些方法是在改善长度外推，不是免费创造长上下文能力。真实长上下文还依赖训练数据长度、attention kernel、KV Cache 容量、检索行为和评测。一个模型能“接收”128K token，不代表它能在所有距离上稳定使用信息。

## MQA 与 RoPE 如何一起工作

二者组合时，典型 decode 流程是：

1. 当前 token 经过投影，得到多组 Q 和少量 K/V。
2. 根据当前 `position_id` 对 Q 和新 K 应用 RoPE。
3. 把旋转后的 K 与 V 追加到 KV Cache。
4. 每个 query head 读取对应 KV group 的历史缓存，计算 attention。
5. 得到当前 token 的输出，然后进入下一层和下一解码步。

在常见实现里，KV Cache 保存的是已经应用 RoPE 的 K，因此历史 token 不必在每一步重复旋转。V 不参与 RoPE。MQA 中的单组 K 在当前位置旋转一次，然后被所有 query heads 逻辑共享。

这也带来一个容易忽略的系统约束：KV Cache 不只是内容缓存，还隐含位置变换。进行 prefix reuse、滑动窗口、缓存搬移或 position scaling 时，必须保证 position id 和所采用的 RoPE 规则一致；不能把来自不同位置约定的 K 直接拼在一起。

## 两者对长上下文的贡献不同

RoPE 与 MQA 都常被宣传为“支持长上下文”，但它们贡献的维度不同：

| 机制 | 主要改变 | 主要收益 | 没有单独解决的问题 |
| --- | --- | --- | --- |
| MQA | KV head 数 | 缩小 KV Cache、降低 decode 带宽 | 位置建模、全部 attention score 计算 |
| GQA | KV head 分组 | 在质量与缓存之间折中 | 位置建模、无限长度外推 |
| RoPE | Q/K 的位置变换 | 让点积自然依赖相对位置 | KV Cache 容量、可靠无限外推 |

如果目标是更长上下文，RoPE 让模型拥有更合适的位置归纳偏置，MQA/GQA 让缓存更有机会装进显存并被高效读取。前者偏模型表示，后者偏模型结构与服务系统。二者互补，却不能互相替代。

## 常见误解

### 误解一：MQA 只有一个 Attention Head

不对。MQA 仍有多个 query heads，只有 key/value head 数为 1。不同 query heads 仍然生成不同 attention distribution。

### 误解二：MQA 把 Attention 计算量降低了 `H_q` 倍

不准确。KV Cache 容量和 K/V 读取量可以按 `H_kv` 大幅下降，但各 query heads 的 score、softmax 和 value aggregation 仍然存在。实际加速主要来自缓解增量解码的内存带宽瓶颈。

### 误解三：MQA 会在不同用户请求之间共享 K/V

不对。它默认只在同一样本的 heads 之间共享。跨请求的相同前缀复用属于 serving 系统的 prefix caching。

### 误解四：RoPE 是把一个位置向量加到 Token Embedding

不对。RoPE 通常不把位置向量直接加到输入，而是对 Q/K 的成对维度施加与位置有关的旋转。

### 误解五：RoPE 天然支持任意长序列

不对。它有良好的相对位置结构，但超出训练长度仍属于外推问题，通常需要 scaling 方法和长上下文训练共同支持。

### 误解六：使用 MQA 一定不损失质量

不对。质量变化依赖模型规模、任务、训练配方和 KV bottleneck。原始 MQA 工作报告的是较小质量退化；GQA 正是为了提供更稳妥的折中。

## 如何阅读一份模型配置

看到某个模型宣称使用 MQA、GQA 或 RoPE，可以按下面顺序检查：

1. `num_attention_heads` 与 `num_key_value_heads` 分别是多少？两者相等是 MHA，后者为 1 是 MQA，介于两者之间是 GQA。
2. `head_dim`、层数、缓存 dtype 和最大上下文是多少？这些量共同决定 KV Cache 数量级。
3. 推理框架是否有原生 grouped-query kernel？如果先物理复制 K/V，再调用普通 MHA，收益可能打折。
4. RoPE 的 `base/theta`、训练长度和 scaling 配置是什么？只知道“用了 RoPE”不足以判断长度外推。
5. 速度数据来自 prefill 还是 decode？batch size、上下文长度、生成长度和硬件是否相同？
6. 质量结论是否覆盖了目标任务，还是只来自平均 benchmark？

这种读法可以避免把架构名词直接等同于真实系统能力。MQA/GQA 的效果要看 KV Cache 和 kernel，RoPE 的效果要看训练与 scaling；最终还要回到端到端 latency、throughput、并发容量和任务质量。

## 总结

标准 MHA 为每个 query head 配置独立的 K/V heads，表达能力强，却让自回归 decode 必须反复读取庞大的 KV Cache。MQA 保留多组 Q，只让所有 query heads 共用一组 K/V，从而显著减少缓存容量和内存带宽；GQA 则用少量 KV groups 在效率和质量之间提供连续折中。

RoPE 没有改变 KV head 数。它把 Q/K 的维度两两配对，并按 token 位置进行多频率旋转，使两个位置的点积最终依赖相对距离 `n-m`。它提供了适合序列建模的位置结构，但不等于无限长度外推。

把二者放在一起看，现代 Transformer 的长上下文效率就更容易理解了：**RoPE 负责让 attention 知道历史 token 在哪里，MQA/GQA 负责让这些历史 token 的 K/V 更便宜地留在缓存里。** 前者处理坐标，后者处理带宽。真正高效、可靠的长上下文模型，需要两类设计与训练数据、kernel、缓存管理和服务调度共同配合。

## 参考资料

- [Noam Shazeer, Fast Transformer Decoding: One Write-Head is All You Need, 2019](https://arxiv.org/abs/1911.02150)
- [Jianlin Su et al., RoFormer: Enhanced Transformer with Rotary Position Embedding, 2021](https://arxiv.org/abs/2104.09864)
- [Joshua Ainslie et al., GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, 2023](https://arxiv.org/abs/2305.13245)
- [Aakanksha Chowdhery et al., PaLM: Scaling Language Modeling with Pathways, 2022](https://arxiv.org/abs/2204.02311)

# VeRL-Omni: 面向多模态生成模型的 RL 后训练基础设施

`verl-project/verl-omni` 是一个很值得关注的开源项目。它表面上是 `verl` 生态里的一个多模态 RL 训练仓库，但如果从研究系统的角度看，它真正重要的地方不只是“又支持了几个模型”，而是把多模态生成模型的后训练问题拆成了几个可以组合的工程层：高吞吐 rollout、异步 reward、分布式 actor 训练、模型族适配、奖励服务、稳定性修正和端到端 recipe。对于 diffusion、统一多模态理解生成模型、以及 Qwen3-Omni 这样的 omni model，这种拆分比单独讨论某个算法更关键。

我更愿意把 VeRL-Omni 看作一个面向 multimodal generative RL 的系统框架，而不是单一算法库。传统 RLHF 或 PPO 框架主要围绕文本 LLM 展开：action 是离散 token，rollout 是自回归生成，reward 多数是规则、RM 或 LLM judge，训练瓶颈也主要在序列长度、KV cache 和 actor/reference/critic 的资源编排。多模态生成模型则完全不同：diffusion model 的 action 可以是连续去噪轨迹，video generation 的样本很重，reward 可能是 OCR、VLM judge、HPSv3 或外部 HTTP scorer，rollout 后端还要能处理图像、视频、音频和复杂 pipeline。因此，直接把文本 RLHF 框架套到多模态生成上，通常会在 I/O、显存、吞吐、reward latency 和数值稳定性上遇到问题。

![VeRL-Omni architecture](/images/blog/verl-omni-architecture.svg "图 1：VeRL-Omni 将 rollout、reward loop 和 trainer 解耦成可组合的多模态 RL 后训练栈。")

## 项目定位：从 verl 里长出来的多模态 RL 栈

官方 README 对 VeRL-Omni 的定位很清楚：它是 built on top of `verl` 的通用 RL 训练框架，聚焦 multimodal generative models。它起源于 `verl` 中的多模态生成 RL 工作，后来独立出来，是为了让这一方向能围绕自己的约束快速演进。这句话背后有一个重要判断：多模态生成 RL 不是文本 RL 的小改动，而是一个有独立系统瓶颈的训练范式。

官方文档把目标模型分成三类。第一类是 diffusion generative models，包括图像、视频和音频生成，例如 Qwen-Image、Stable Diffusion 3.5、Wan2.2。第二类是 unified multimodal understanding + generation models，例如 BAGEL、HunyuanImage-3.0。第三类是 omni-modality models，也就是同时处理文本、图像、音频和视频的模型，例如 Qwen3-Omni。这个分类很有价值，因为它不是按“模型名字”罗列功能，而是按训练形态和系统负载划分问题。

从工程架构看，VeRL-Omni 的核心抓手包括：用 `vLLM-Omni` 做高吞吐多模态 rollout；用 flexible reward pipeline 支持规则奖励、模型奖励、HTTP scorer 和多 reward 组合；用 FSDP2 或 VeOmni 做 actor/ref 的分布式训练；用 async reward 把 reward latency 从关键路径里部分隐藏；用 rollout correction、deterministic rollout/reward/trainer 等工具提高稳定性。它不是把所有功能塞进一个大 trainer，而是保持 `verl` 式的角色拆分和 resource pool 设计，让 rollout、reward、trainer 可以按任务重新摆放。

## 支持模型和算法：不是单点 demo，而是模型族矩阵

VeRL-Omni 官方模型表里，Qwen-Image 是当前最完整的 diffusion image 样例，支持 FlowGRPO、Flow-DPPO、MixGRPO、GRPO-Guard、DiffusionNFT 和 Diffusion DPO 等训练方式。SD3.5 Medium 支持 FlowGRPO 和 DPO。Wan2.2-TI2V-5B 作为 text-to-video diffusion model，示例使用 DanceGRPO 和 HPSv3 reward。BAGEL 则代表 unified multimodal understanding + generation，用 FlowGRPO 做 OCR 或 PickScore 类 reward。Qwen3-Omni-30B-A3B Thinker 代表 omni-modality，使用 GSPO + LoRA 做数学推理后训练。

![VeRL-Omni model map](/images/blog/verl-omni-model-map.svg "图 2：VeRL-Omni 按 diffusion、unified multimodal 和 omni-modality 三类模型组织算法支持。")

这个矩阵说明两件事。第一，VeRL-Omni 的目标并不是只服务 diffusion，也不是只服务 Qwen3-Omni，而是把“多模态生成模型 RL 后训练”作为统一问题来处理。第二，算法和模型不是任意组合的。FlowGRPO 更适合 flow matching / diffusion；DPO 和 DiffusionNFT 面向 diffusion preference 或 forward process；GSPO 则更接近 verl 原生 PPO-style 的 LLM/omni Thinker 后训练。一个好的框架必须承认这些差异，而不是用一个抽象把所有细节压平。

从研究视角看，这种矩阵化组织有助于比较不同算法的边界。例如，FlowGRPO 的关键在于把 flow matching 原本确定性的 ODE rollout 转成带随机性的 SDE rollout，从而获得 group sampling 和 policy gradient 所需的探索；Flow-DPPO 则把 PPO-style ratio clipping 替换为基于 flow 模型结构的 divergence mask；DiffusionNFT 不直接在 reverse sampling chain 上做 policy gradient，而是优化 forward diffusion process，把 reward 信号折进 supervised flow-matching objective。这些方法都叫“diffusion RL”，但优化对象、log-prob 来源、采样轨迹和稳定性问题完全不同。

## FlowGRPO：把在线 RL 接到 flow matching 上

FlowGRPO 是 VeRL-Omni diffusion 侧最核心的算法入口之一。对于文本 LLM，GRPO 的 group sampling 很直观：对同一个 prompt 采样多个回答，根据相对 reward 计算 advantage，再更新策略。对于 flow matching model，这件事不直接成立，因为标准 flow matching 采样通常是确定性 ODE 过程，不天然提供 RL 需要的随机轨迹和 log-prob。

FlowGRPO 的做法是把 ODE 转成等价 SDE，在保持边际分布的前提下引入随机性。这样同一个 prompt 可以生成一组不同图像轨迹，reward model 给每张图打分，训练再对去噪轨迹中的若干步骤计算 policy gradient。为了避免训练所有 denoising steps 成本太高，FlowGRPO 还使用 denoising reduction，只在一个较小的 SDE window 上训练，同时 rollout/inference 仍然可以保持完整采样步骤。

这里的系统难点远比文本 GRPO 多。文本里的 action 是 token，log-prob 是 next-token probability；diffusion 里的 action 更像连续 denoising transition，log-prob 与 SDE 噪声预测相关。文本里一个 batch 是若干 token 序列；Qwen-Image OCR recipe 中，一个 step 可以是 32 个 prompt，每个 prompt rollout.n=16，也就是 512 张图像样本，还要把这些样本按 prompt group 保持在同一个 mini-batch 中，才能正确做 group-relative advantage。再叠加 FSDP、LoRA、micro-batch 和 SDE window，训练维度会变得非常复杂。

VeRL-Omni 的价值在这里体现出来：它不是只给一个公式，而是把这些维度变成明确的配置和 recipe。例如 `data.train_batch_size` 表示 prompt 数，`actor_rollout_ref.rollout.n` 表示每个 prompt 的采样数，`ppo_mini_batch_size` 以 prompt 为单位切 actor mini-batch，`ppo_micro_batch_size_per_gpu` 再控制每卡 micro-batch。对于真正跑实验的人，这些比抽象算法描述更重要，因为一个配置单位理解错，就可能导致 advantage 组被拆散、显存爆掉或训练信号不对。

## vLLM-Omni rollout：多模态 RL 的吞吐核心

多模态生成 RL 的大部分时间不一定花在反向传播上。对于 image/video/diffusion 任务，rollout 可能非常重；对于 Qwen3-Omni 这种模型，rollout 还要处理复杂 tokenizer、processor、stage config、TP 和多模态输入。VeRL-Omni 把 `vLLM-Omni` 作为重点 rollout backend，就是为了解决高吞吐生成问题。

这件事的意义在于，RL 后训练不是离线 supervised fine-tuning。每一步都要从当前或旧策略采样，再打 reward，再更新 actor。如果 rollout 很慢，训练吞吐会直接被锁死。官方 README 提到，在参考 Qwen-Image FlowGRPO setup 上，VeRL-Omni 相比 diffusers-based `flow_grpo` 实现有大约 25% 的端到端吞吐提升，来源包括 `vLLM-Omni` rollout、FSDP2 trainer 和异步 reward 等组合优化。这里不能把数字孤立理解为某个 kernel 快了 25%，更合理的理解是：多模态 RL 的吞吐来自整条 pipeline 的重排。

Qwen3-Omni GSPO recipe 也能看出 rollout 后端的重要性。示例用 4 张 H100/H200 80GB，同一卡上 colocate FSDP actor 和 `vLLM-Omni` rollout，rollout TP=4，actor 是 30B total、3B active 的 MoE Thinker，LoRA rank 64，并开启参数/优化器 CPU offload。由于 rollout engine 和 actor 共享 GPU，stage YAML 中的 `gpu_memory_utilization` 被设得比较保守。这种细节说明，omni model RL 不是“把模型加载起来跑 PPO”这么简单，而是持续在 rollout memory、actor memory、batch size、response length 和 offload 之间做工程平衡。

## Async Reward：把 reward latency 从关键路径挪出去

VeRL-Omni 另一个非常实用的设计是 async reward。多模态 reward 往往很贵：OCR reward 可能需要 VLM judge；图像偏好可能要 PickScore、HPSv3 或 aesthetic model；外部业务 reward 可能跑在单独服务里；视频 reward 更可能是慢模型。如果 reward scoring 必须等完整 rollout batch 结束后才开始，那么 rollout worker 和 trainer 都会出现明显 idle time。

Async reward 的核心是 sample-level streaming reward computation。每个 rollout sample 一完成，agent loop 就把它送到 reward-loop worker，其他 rollout sample 继续生成。训练仍然要等完整 scored batch 准备好以后才开始 actor update，所以它没有破坏 on-policy training step 的语义；它只是把 reward 计算和后续 rollout 重叠起来。官方文档强调，这不是用 partial 或 stale batch 提前更新 actor，而是在 rollout/reward 阶段减少等待。

![VeRL-Omni async reward](/images/blog/verl-omni-async-reward.svg "图 3：Async reward 将已完成样本提前送到 reward worker，使 reward latency 与剩余 rollout 重叠。")

这个设计对多模态尤其关键。文本数学题的 rule reward 可能很快，但图像 OCR reward、VLM judge 或 HTTP scorer 可能很慢。VeRL-Omni 支持把 reward model 放到独立 resource pool，例如 4 张 GPU 给 actor/rollout，额外 1 张 GPU 给 reward inference。这样 reward model 不再和 actor/rollout 抢同一批 GPU。官方 FlowGRPO example 里也提供了 LoRA + async reward 版本，体现的就是这种资源分层思想。

HTTP scorer 是 async reward 的自然延伸。VeRL-Omni 的 HTTP reward client 会把生成图像序列化成 JPEG bytes，和 prompt 一起通过 pickle payload POST 到外部 scorer service；scorer 可以是 Flask/Gunicorn，也可以封装 PaddleOCR、HPSv3、CLIP-based scorer 或其他服务。这样 reward runtime 可以和训练环境解耦，避免一个大型 reward 模型或特殊依赖污染训练环境。对真实实验平台来说，这种解耦很重要，因为 reward 服务经常会被多个训练 job 共享，也可能需要单独扩缩容。

## Rollout Correction：在速度和偏差之间做显式管理

多模态 RL 里还有一个容易被低估的问题：rollout backend 和 training graph 对同一条轨迹的 log-prob 可能不完全一致。以 diffusion FlowGRPO 为例，rollout policy 可能由 vLLM/vLLM-Omni 用低精度 kernel、tensor parallelism 或 fp8/bf16 采样；old policy recompute 则由 actor 在训练图里重新计算 old log-probs；current policy 又在每个 actor mini-step 里计算。为了省时间，可以跳过 old policy recompute，直接复用 rollout backend 的 log-probs，但这会引入 off-policy bias。

VeRL-Omni 的 rollout correction 就是为这个速度-偏差权衡提供显式工具。它用 importance sampling 和 rejection sampling 处理 rollout_logp 与 old_logp 的差异：IS 通过 log-ratio 估计权重，RS 则把落在阈值外的样本或 step 从 loss 中屏蔽。对于 diffusion 来说，因为 active SDE window 通常很短，比如 window size 可能只有 2，统计性质和文本 token 序列不一样，所以文档专门给了 diffusion-specific tuning guide。

我觉得这类工具的研究价值在于，它承认系统优化会改变训练分布。很多论文里会默认 rollout policy 和 training policy 计算完全一致，但实际高吞吐系统中，rollout 和 actor 可能跑在不同 kernel、不同精度、不同并行布局上。VeRL-Omni 把这种 drift 显式监控出来，例如 rollout_is_mean、rejected fraction、KL、log-PPL diff 等指标，能帮助研究者判断“省掉 recompute”是不是正在损害训练。

## Qwen3-Omni GSPO：omni Thinker 后训练的现实样例

对我来说，VeRL-Omni 最值得关注的部分之一是 Qwen3-Omni Thinker GSPO + LoRA trainer。这个例子表明，框架并不只停留在 image diffusion，而是开始覆盖真正的 omni model。示例训练的是 `Qwen/Qwen3-Omni-30B-A3B-Instruct` 的 Thinker，模型是 30B total、3B active 的 MoE，输入模态覆盖文本、图像、视频和音频；训练方式是 GSPO + LoRA，任务是 math reasoning。

这个 recipe 的工程细节很有启发。它只训练 Thinker：LoRA rank 64，target modules 是 all-linear，同时排除 talker、code2wav、code_predictor、visual、audio_tower 等模块，并冻结 vision tower。非 Thinker heads 会在 FSDP wrap 时被 strip。这样做的意图是明确的：先把 RL 信号集中到推理核心，而不是同时更新语音生成、视觉编码和其他 tower。对于 omni model 后训练，这种模块边界非常关键，否则训练成本、显存和不稳定性都会急剧上升。

该 recipe 使用 `dapo` reward manager 进行数学答案准确性奖励，健康信号包括 rollout_actor_probs_pearson_corr 大于 0.95、actor loss 在合理范围、grad norm 不爆、max memory 不超过 60GB 等。官方 README 里给出的参考性能显示，4 张 H100/H200 上一个 full step 大约 22 分钟，主要时间花在 rollout generation。这说明即使只是 Thinker-only + LoRA，omni model RL 仍然非常重。未来如果扩展到音视频交互 reward、长视频理解 reward 或 social interaction reward，rollout 和 reward 的系统设计会更加重要。

## 和我的研究方向的关系

VeRL-Omni 和我关注的 multimodal systems、long-video understanding、efficient omni-modal orchestration 有明显联系。过去很多多模态研究关注模型结构和 benchmark，但当我们真正想优化一个 omni model 的行为时，问题会变成：如何生成高质量多模态 rollout？如何定义 reward？如何把 reward 模型放进训练 pipeline？如何让昂贵 reward 不拖死训练？如何在有限 GPU 下同时放 actor、rollout、reward 和 reference？这些问题都不是单纯模型架构能解决的。

从 long-video understanding 看，VeRL-Omni 的 async reward 和 external scorer 机制很有启发。长视频任务的 reward 往往不是一个简单 scalar，它可能来自 temporal grounding、event consistency、audio-visual alignment、OCR、object tracking 或 human preference。如果未来要做 long-video RL 后训练，reward 计算大概率需要独立服务化，并且要和 rollout 重叠，否则每个 step 的等待时间会非常长。VeRL-Omni 已经给出了一个可扩展的 reward-loop 抽象。

从 omni-modal orchestration 看，VeRL-Omni 可以被理解为训练侧的 orchestration。推理时 orchestration 关注如何选择模型、工具和专家；训练时 orchestration 关注如何组织 actor、rollout、reward、reference、data 和 distributed backend。二者都在回答同一个问题：多模态系统如何在复杂模块之间保持高效、稳定和可控。VeRL-Omni 的价值正是在训练侧把这些模块显式化。

## 局限与风险

当然，VeRL-Omni 仍然是一个快速演进中的工程框架，不能把它理解成已经完全成熟的黑盒平台。首先，很多 recipe 对环境版本非常敏感。例如 Qwen3-Omni GSPO README 明确 pin 了 vLLM、vLLM-Omni、transformers、torch、flash-attn 和 verl commit，并说明 transformers 5.x 会触发权重加载问题。这说明 omni RL 的软件栈还处在快速变化阶段，复现实验时需要严格记录版本。

其次，支持矩阵里的“可跑”不等于“已调优到最优”。官方 Qwen3-Omni GSPO 示例也把初步结果定位为 plumbing-correctness signal，而不是最终性能结论。对于研究者来说，能跑通、loss 正常、actor-rollout 相关性高、没有 OOM，只是第一步。真正的 research contribution 仍然需要设计任务、reward、数据和评测。

再次，多模态 reward 本身仍然是核心难点。OCR reward、HPSv3、PickScore、VLM judge 都有偏差。RL 会放大奖励模型的偏差，特别是在生成模型里，模型可能学会 exploit reward 而不是提升真实质量。因此，VeRL-Omni 解决的是“如何训练”的基础设施问题，不自动解决“reward 是否正确”的研究问题。

## 我对 VeRL-Omni 的理解

我倾向于把 VeRL-Omni 看作多模态生成 RL 的基础设施原型。它不是某一个算法的附属代码，而是把多模态 RL 训练中最难绕开的部分系统化：rollout 要快，reward 要灵活，资源要可分池，trainer 要能支撑 diffusion 和 omni model，稳定性问题要有监控和修正。对于一个研究者来说，这类框架的意义在于降低“把想法跑起来”的工程门槛，让更多精力可以放在 reward design、benchmark construction、long-video reasoning 和 omni interaction 上。

如果说 TMRoPE 解决的是 omni token 的时间坐标问题，Hybrid Attention MoE 解决的是长上下文计算结构问题，那么 VeRL-Omni 解决的是后训练阶段的系统组织问题。一个完整的 omni AI 系统不只需要会看、会听、会说，还需要能通过 reward 持续改进。而要让这种改进发生，必须有稳定、高吞吐、可扩展的训练基础设施。VeRL-Omni 正是在这个方向上迈出的一步。

## 参考资料

- [verl-project/verl-omni GitHub Repository](https://github.com/verl-project/verl-omni)
- [VeRL-Omni Documentation](https://verl-omni.readthedocs.io/en/latest/index.html)
- [Supported Models](https://verl-omni.readthedocs.io/en/latest/start/models.html)
- [Async Reward for Diffusion Training](https://verl-omni.readthedocs.io/en/latest/algo/async_reward.html)
- [Flow-GRPO Documentation](https://verl-omni.readthedocs.io/en/latest/algo/flowgrpo.html)
- [Qwen3-Omni GSPO Trainer README](https://github.com/verl-project/verl-omni/tree/main/examples/gspo_trainer)

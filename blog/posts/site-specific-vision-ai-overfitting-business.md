# 当 Overfitting 变成商业模式：从 ESP32 车库门到一座超市一个模型

> 读完 [《10小时攻关，半小时 Dev Time：在指甲盖大小的单片机上跑神经网络识别车库门》](https://yage.ai/esp32-garage-door-ai.html)，我认为它展示的不只是一个 TinyML 项目，而是一种非常直球的商业机会：通用机器视觉需要处理整个世界，现场专用视觉只需要处理一个摄像头。过去定制成本吃掉了这条路线的商业价值；现在 Agent 可以把数据挖掘、标注、训练、量化、烧录、测试和迭代自动化，定制本身第一次可能成为规模化产品。

机器视觉过去最难的地方是什么？不是识别“门”这个概念，而是一个模型必须面对无数门、无数车库、无数相机、无数光线、无数遮挡和无数安装角度。为了卖给所有人，模型必须交一笔巨大的 **generalization tax**。

但一个真实客户通常并不需要“理解世界”。他只需要知道：

```text
这个车库门现在关没关？
这条消防通道有没有被堵住？
这个冷柜门是否超过两分钟没有关闭？
这个固定工位的工人有没有佩戴指定护具？
这个卸货口是否出现了不该出现的人或物？
```

如果摄像头固定、任务固定、动作固定，问题空间会骤然缩小。模型不需要对所有车库泛化，只要对**这个机位未来会出现的变化**泛化。

我的商业判断是：

> **AI Agent 正在把机器视觉从“卖一个通用模型”改造成“自动为每个现场生产一个专用传感器”。真正可规模化的产品不是某个识别算法，而是 Site-Specific Model Factory。**

这篇文章从这个车库门实验出发，讨论为什么这个机会现在才出现、哪些任务真的适合、商业模型怎样成立、护城河在哪里，以及怎样用一个最小试点验证它是不是生意，而不只是一个漂亮 Demo。

![Generalization tax versus site-specific scope](/images/blog/site-vision-generalization-tax.svg "图 1：固定现场主动缩小部署分布，把通用模型的泛化税换成可管理的现场校准与漂移维护。")

## 1. 先看原项目到底做了什么

原文作者要解决的是一个具体问题：家里的车库门没有磁控传感器，曾经因为忘关而整夜敞开。作者使用一块带摄像头、Wi-Fi 和有限算力的 ESP32-CAM，希望通过拍照和片上神经网络，把它变成一个只回答 `open / closed` 的视觉传感器。

这套流程中最重要的不是模型名称，而是 AI Agent 获得了一个可自行迭代的闭环。根据作者公开记录，Agent 完成了：

1. 使用本地 Vision LLM 初步标注历史监控帧，并通过 contact sheet 复查；
2. 发现车库门绝大多数时间关闭，正样本极度稀缺；
3. 先训练初始视觉模型，扫描过去几天的几十万帧，主动挖掘更多“门开着”的稀有样本；
4. 经过多轮数据回收后，微调一个约 90 万参数的 MobileNet V4；
5. 评估 ESP32 端到端机会，作者报告推理约 200 ms，而拍照本身约 400 ms；
6. 发现普通 INT8 量化造成明显预测翻转，于是进行 LSQ-based quantization-aware training；
7. 编写测试固件，通过 HTTP POST 向单片机发送图片，批量验证精度、内存、延迟和温度；
8. 最终固件每十分钟拍照推理，短时唤醒 Wi-Fi 上报，其余时间深度睡眠；
9. 连供电设备的低电流自动断电问题也通过实际硬件测试纳入交付。

作者将整个过程描述为约 10 小时，其中自己的主动开发时间约半小时。这里的时间、精度和硬件结果是**作者报告**，不是本文独立复现实验；但即使不把 10 小时当作普遍 SLA，这个案例仍揭示了关键变化：AI 不只是写一段分类代码，而是在代理完整的现场机器学习生命周期。

## 2. 为什么“只管一个车库”会让问题突然简单

一个通用车库门分类器需要处理：

- 平开门、卷帘门、分段提升门和不同材质；
- 室内、室外、逆光、夜视、红外和雨雪；
- 相机在顶角、正面、侧面或车内；
- 门被车、人、纸箱和工具遮挡；
- 开门一半、维修状态、掉线画面和相机移动；
- 不同国家、建筑、安装标准和用户行为。

而一个固定机位通常拥有极强的先验：

```text
相机外参基本不变
目标区域基本不变
背景结构基本不变
类别数量很少
状态转移有限
业务规则清楚
历史视频天然存在
错误可以由现场反馈纠正
```

因此，模型可以利用背景、位置、颜色、边缘和局部纹理等在通用任务里会被视为 shortcut 的信息。在这个部署合同里，它们反而是合法特征。

### 2.1 这不是新发现，NoScope 已经证明了 specialization 的价值

[NoScope](https://arxiv.org/abs/1703.02529) 在 2017 年就研究了固定角度 webcam 和 surveillance video。它给定目标视频、要查询的对象和一个昂贵 reference network，自动搜索由 difference detector 与 specialized model 组成的 cascade。论文报告，在二分类视频查询上可实现两到三个数量级的加速，同时将准确度维持在 reference network 的 1–5% 范围内。

NoScope 的关键结论不是某个加速数字，而是：**最优 cascade 会随视频和对象变化。** 换句话说，不存在一个对所有摄像头都最优的廉价模型；但如果允许逐视频流专用化，小模型就可以忠实模拟大模型在这一小块世界里的行为。

### 2.2 Chameleon 与 Ekya 又补上了配置和漂移

[Chameleon](https://doi.org/10.1145/3230543.3230574) 研究视频分析配置如何随着摄像头和时间变化。分辨率、帧率、模型等配置并非一套参数通吃，系统需要利用摄像头之间的相关性，低成本地重新选择配置。

[Ekya](https://arxiv.org/abs/2012.10557) 则直接面对现场数据漂移：边缘端压缩模型会因为 live video 与训练数据逐渐分离而掉点。Ekya 用 continuous learning 周期性重训，并用 micro-profiler 找出最值得重训的模型。论文报告，相比基线调度器，其 accuracy gain 高 29%，而基线需要 4 倍 GPU 资源才能达到相同准确度。

三项工作连起来，几乎就是这个商业机会的技术历史：

```text
NoScope: 每路视频值得拥有专用小模型
Chameleon: 每路视频的最优配置不同且会变化
Ekya: 每路视频需要现场持续学习来应对漂移
今天的 Agent: 把上述定制和维护流程自动执行
```

## 3. “Overfit 就完了”是对的，但要说得更精确

商业直觉上，`overfit` 是一个非常有力量的表达：不要承担无关场景的泛化成本。但工程上如果真的让模型记住训练帧，它会在第二天清晨、一次灯泡更换或相机被碰歪后失效。

应该区分两件事。

### 3.1 坏的 overfitting

```text
训练集和测试集来自相邻视频帧
同一事件被切成数百张近重复图，随机分到两边
模型记住时间戳、水印或曝光模式
没有覆盖夜晚、逆光、雨雪、遮挡和设备故障
离线准确率很高，未来事件召回率很差
```

### 3.2 好的 specialization

```text
部署范围明确限定为 camera_id + ROI + task + action
按时间和事件分割训练、验证、测试集
保留不同昼夜、天气、季节和运营状态
允许模型使用稳定的现场 shortcut
对分布外输入 abstain，而不是武断判断
相机移动或漂移触发重新校准
```

所以更准确的产品承诺不是“我们故意过拟合”，而是：

> **我们不追求跨现场泛化，只承诺经过验收的 site-specific generalization envelope。**

这会把一个含糊的 AI 能力，改造成可写进合同的范围：哪一路相机、哪个 ROI、哪些状态、何种光照、允许多少误报、多久发现一次漏报、发生相机位移后系统怎样降级。

## 4. 为什么以前大家不这么做：不是算法不知道，而是服务成本太高

逐现场定制过去一直存在，但通常是昂贵的项目制交付：

```text
销售去现场定义需求
工程师导出视频
标注团队制定规则
算法工程师清洗和训练
嵌入式工程师压缩部署
现场工程师安装调参
客户反馈误报
所有团队重新排期
```

如果换一个超市就要重复一个月，毛利会被人力和沟通吞掉。每增加一个客户，不是复制软件，而是新增一个小型咨询项目。

AI Agent 改变的不是识别上限，而是**定制成本曲线**。它可以把大量重复但跨工具的工作变成机器时间：

- 解析客户用自然语言描述的事件；
- 从 NVR 历史录像中采样并去重；
- 用 VLM 产生弱标签和标签解释；
- 制作 contact sheet 供人快速确认；
- 用 embedding、聚类和初始模型寻找少数类；
- 自动执行训练、蒸馏、剪枝和量化搜索；
- 编译固件或边缘容器；
- 调用测试设备、串口和摄像头完成硬件闭环；
- 生成验收报告与失败样本；
- 上线后监测分布漂移并提出重训候选。

这就是为什么机会现在变得“直球”：原来不可规模化的最后一公里，被 Agent 变成了可重复工作流。

## 5. 真正的产品不是模型，而是 Site-Specific Model Factory

![Site-specific micro-model factory](/images/blog/site-vision-model-factory.svg "图 2：通用 VLM 负责理解和启动，现场小模型负责稳定、廉价地执行，Agent 负责把两者变成闭环。")

一个可卖的系统不应让工程师为每个客户手工开 notebook。它应当接受一个现场任务合同，自动产出可验收的传感器：

```text
Input
  camera stream / historical video
  region of interest
  event description
  acceptable false alarms and misses
  response action
  privacy and retention policy

Compiler
  candidate mining
  weak labeling
  human review
  specialist training
  compression / quantization
  temporal rule synthesis
  acceptance testing

Output
  edge model
  confidence and abstention thresholds
  event policy
  monitoring dashboard
  drift trigger
  rollback package
```

这个架构里，大模型和小模型的职责不同。

### 5.1 Foundation model 是教师和开发工具

VLM 适合处理长尾语义、自然语言需求、零样本初筛、难例解释和自动标注。它强，但贵、慢，而且输出不够稳定。

### 5.2 Site specialist 是生产传感器

小型 CNN、轻量 ViT、change detector 或传统规则负责高频、低延迟、可离线的持续推理。它不理解世界，但对一个固定任务足够稳定。

### 5.3 Agent 是模型工厂的操作系统

Agent 决定何时调用教师、何时让人复核、何时重训、哪个模型满足硬件限制、怎样运行 shadow test，以及失败时如何回滚。商业价值大多聚集在这里，而不是最后那个 90 万参数网络。

## 6. 最适合的商业任务长什么样

不是所有视觉问题都适合“一场景一模型”。可以用七个条件筛选。

| 条件 | 好机会 | 差机会 |
| --- | --- | --- |
| 机位 | 固定、可防碰撞 | 移动相机、频繁改造 |
| 标签 | 少量、互斥、可解释 | 开放世界复杂行为 |
| ROI | 目标区域稳定 | 目标可出现在任何位置 |
| 反馈 | 事件可被确认 | 无法知道模型是否错 |
| 价值 | 漏报或人工巡检成本高 | 看见也没有动作 |
| 频率 | 有足够历史数据或正常样本 | 罕见到无法验收 |
| 合规 | 可本地处理、目的明确 | 涉及敏感身份与高风险判断 |

### 6.1 第一批最直球的任务

- 门、闸、阀、盖板的开关状态；
- 消防通道、卸货口、工位和安全区域是否被占用；
- 托盘、料箱、货架、垃圾桶是否达到阈值；
- 设备指示灯、仪表盘或固定显示器是否异常；
- 固定产线的缺件、错位、卡料和停线；
- 冷柜、仓门或泳池门长时间未关闭；
- 固定停车位、充电位或装卸位占用状态；
- 特定区域在特定时段是否有人进入。

它们的共同点是：状态明确、位置固定、事件发生后存在清晰动作。

### 6.2 “超市有没有人抽烟”为什么不能一句话说简单

用户提出的超市例子方向是对的：一个超市的机位、吸烟高发位置和运营环境比“所有超市”简单得多。但抽烟仍可能涉及：

- 烟很小，常被手、脸和货架遮挡；
- 电子烟与普通动作外观接近；
- 真正证据可能是时序动作或烟雾，不是单帧物体；
- 远距离像素不足；
- 误报会直接影响顾客体验；
- 持续人员监控涉及隐私、告知和用途限制。

因此 specialization 会降低难度，但不会让信息论上不可见的信号凭空出现。更现实的设计可能是：先以区域、时段和多帧行为做高召回候选，再由值班人员确认，而不是自动给顾客定性。

## 7. 商业机会矩阵：不要从“最酷”开始，要从“最容易验收”开始

![Opportunity matrix for site-specific vision](/images/blog/site-vision-opportunity-matrix.svg "图 3：优先选择现场稳定、动作价值高且容易取得反馈的任务，而不是主观性强的人类行为判断。")

可以用两个轴选择首批垂直场景：

- **现场可压缩性**：固定机位、固定 ROI 和少状态能否显著减少问题空间；
- **事件经济价值**：发现事件后是否能节省人工、避免损失或触发明确流程。

优先象限是“高可压缩、高价值”：冷柜门、危险区域侵入、消防通道、产线卡料、固定资产状态。低可压缩、主观性强的行为判断，例如“员工是否积极”“顾客是否可疑”，不应成为早期产品。

一个事件只有在下面的链路完整时才有价值：

```text
看见 → 判断 → 通知 → 有人处理 → 结果被记录 → 反馈进入模型
```

只卖 dashboard 而不连接行动，客户很快会得到另一块没人看的屏幕。

## 8. 商业模式：卖结果，不卖“一个模型”

最自然的包装是 `hardware + setup + recurring service`。

### 8.1 一次性费用

- 设备、支架、电源与安装；
- 现场任务定义和隐私配置；
- 历史数据接入；
- 初次模型生成与验收；
- 与现有 NVR、告警、工单或门禁系统集成。

### 8.2 经常性收入

- 每摄像头/月或每事件类型/月订阅；
- 设备健康监控；
- 漂移检测和模型更新；
- 事件存储、报表和审计；
- SLA、人工复核或异常升级；
- 多站点管理与角色权限。

### 8.3 结果型定价

对价值容易计算的任务，可以按节省或风险定价：

- 减少多少巡检工时；
- 避免多少冷链损耗；
- 缩短多少停线时间；
- 降低多少未关门时长；
- 提高多少装卸位周转。

但结果型合同需要强审计能力，否则“事件是否本来就会发生”“告警后为何无人处理”会变成责任争议。

## 9. Unit Economics：Agent 只降低训练成本，不会自动消灭现场成本

![Unit economics of a site-specific vision service](/images/blog/site-vision-unit-economics.svg "图 4：真正决定毛利的是激活时间、人工复核、现场上门、误报支持和持续维护，而不只是训练 GPU 成本。")

一个现场的贡献毛利可以粗略写成：

```text
Site Contribution Margin
= subscription revenue
− hardware amortization
− installation and truck rolls
− data transfer and storage
− teacher-model and training compute
− human review and escalation
− customer support
− warranty and replacement
```

AI 最可能大幅压缩的是：数据筛选、初标、实验、模型压缩、测试报告和一部分远程诊断。它很难自动消除支架松动、镜头脏污、Wi-Fi 死角、断电和客户流程不执行。

因此最重要的商业指标不是 benchmark accuracy，而是：

| 指标 | 为什么重要 |
| --- | --- |
| Time to first accepted alert | 决定销售承诺和现金回收速度 |
| Human review minutes per camera-week | 决定服务毛利能否随站点增长 |
| False alerts per camera-day | 决定客户多久开始忽略系统 |
| Missed critical events | 决定产品是否真正承担业务价值 |
| Remote resolution rate | 决定是否需要昂贵上门 |
| Retrain frequency and cost | 决定“专用模型”是不是维护陷阱 |
| Camera-month retention | 决定它是功能还是长期基础设施 |

如果一个模型每月只花 2 美元推理，但每周需要工程师看 30 分钟误报，它仍然是一个低毛利服务业务。

## 10. 最关键的产品指标不是准确率，而是每摄像头每天误报数

假设某事件每天真实发生一次，模型 precision 为 90%，听起来不错。但如果它扫描数十万帧并产生 20 个告警，运营人员仍会被淹没。

现场产品至少应同时报告：

```text
event-level recall
false alerts per camera-day
median and P95 detection delay
abstention rate
human review time
alert-to-action completion rate
days since last verified positive
performance by day / night / weather / shift
```

数据集切分也必须按事件和时间，而不是随机帧。相邻 30 fps 视频中的两张图几乎相同，把它们随机放进训练集和测试集会制造虚假高分。

推荐的验收顺序是：

1. 用历史时间段做 retrospective test；
2. 用完全后置的时间段做 prospective holdout；
3. 在线 shadow mode 运行，不触发真实动作；
4. 人工核验所有告警，并抽样检查未告警区间；
5. 达到 event-level 合同后再进入 active alert；
6. 对高风险动作始终保留人工确认或双传感器验证。

## 11. 数据飞轮应该逐现场闭环，而不是把所有客户视频混成一锅

现场专用模型天然形成两层数据资产。

### 11.1 Site-private loop

```text
现场新视频
→ 低置信度与分布外片段
→ 本地或授权范围内复核
→ 更新该 site specialist
→ shadow comparison
→ 通过后替换
```

这层数据可能包含敏感运营信息，应默认隔离、最小保留并支持删除。

### 11.2 Cross-site reusable priors

可以跨客户复用的不是原始画面，而是：

- 任务 schema；
- ROI 模板；
- 标签规范；
- 负样本生成方法；
- 漂移检测器；
- 压缩和量化 recipe；
- 评测协议；
- 与设备和业务系统的 connector；
- 经授权、去标识或合成的 generic representation。

这能避免两个极端：既不把每个客户当完全从零的项目，也不把所有数据混在一起制造隐私和合同风险。

## 12. 护城河在哪里：不是视觉模型，而是自动交付和运营数据

一个 MobileNet 分类器很容易复制，甚至客户可以让另一个 Agent 重新训练。真正的护城河更可能来自五层。

### 12.1 任务定义资产

什么算“门没有关好”？开 5 厘米算不算？维修时是否暂停？持续多久才告警？任务 schema 越成熟，销售到验收越快。

### 12.2 自动化闭环

从 NVR 抽帧到 edge rollout 是否真正无人值守，能不能自动发现 class imbalance、label leakage、量化翻转和硬件异常。这是原车库案例最有价值的部分。

### 12.3 运维和漂移数据

哪些光照变化会使哪个模型失败、哪类摄像头多久会移动、怎样用最少复核恢复性能。这些跨站点经验会不断改善成本模型。

### 12.4 业务集成

客户不购买一个概率值，而是购买短信、工单、停机、巡检、门禁联动和审计记录。连接器和责任链比模型权重更粘。

### 12.5 信任与验收

能否提供版本、数据来源、阈值、回滚、事件证据和 SLA。现场 AI 一旦参与安全与运营，可信交付本身就是产品。

![Lifecycle and moat](/images/blog/site-vision-lifecycle-moat.svg "图 5：专用模型必须经历 shadow、验收、监控、漂移检测和可回滚更新；持续积累的是交付系统与运营知识。")

## 13. 产品架构：Policy shared, perception specialized

每个现场都训练完全独立的软件栈会失控。更合理的边界是：

```text
Shared control plane
  device registry
  task schema
  training orchestrator
  model registry
  policy engine
  alert routing
  monitoring / audit / rollback

Site-specific artifacts
  camera calibration
  ROI and masks
  event thresholds
  specialist weights
  temporal rules
  local drift baseline
```

也就是共享平台和治理，专用感知与阈值。这样既保留 specialization 的性能优势，又不把公司变成维护几千套手工代码的外包团队。

模型还需要 `abstain / degraded / offline` 三种非正常输出：

- `abstain`：当前画面超出验收分布，交给人或大模型；
- `degraded`：镜头污损、遮挡或机位移动，暂停自动动作；
- `offline`：设备或网络故障，触发设备告警而不是业务告警。

不会拒答的现场模型，会把 camera shift 误判成业务事件。

## 14. Edge 部署为什么重要，但不是所有场景都必须 ESP32

ESP32-CAM 证明极低成本硬件也能承担特定二分类任务。边缘部署有明确价值：

- 原始视频不离开现场；
- 带宽成本低；
- 断网仍可判断；
- 延迟稳定；
- 小模型功耗和 BOM 低。

但产品不应为了“端侧”而端侧。硬件层应按任务选择：

```text
MCU: 低频拍照、少类别、简单状态、极低功耗
NPU camera: 持续视频、检测或轻量时序
Edge GPU: 多路视频、复杂行为、现场 VLM fallback
Cloud: 低频复核、跨站点训练、管理和报告
```

现代工具链已经把这条路线产品化了一部分。例如 [NVIDIA TAO Toolkit](https://docs.nvidia.com/tao/tao-toolkit/text/overview.html) 支持用自有数据 fine-tune 预训练视觉模型、蒸馏到更小 backbone、进行量化感知训练并导出 ONNX。工具不是护城河，但它降低了搭建模型工厂的基础成本。

## 15. 失败模式：这条生意最容易死在哪里

### 15.1 Demo accuracy 很高，真实 positive 太少

车库门案例主动挖掘稀有开门帧，说明 class imbalance 是核心。商业验收不能只看正常状态；如果三个月只有一次危险事件，就需要事件模拟、历史回放或合成数据，但必须明确它们与真实事件的差距。

### 15.2 每个站点都需要专家救火

如果 Agent 只能完成 80%，剩余 20% 每次都需要 senior scientist，规模化仍然失败。应该记录每次人工介入的原因，并优先自动化最高频阻塞。

### 15.3 相机不是传感器级安装

普通监控摄像头可能被保洁碰动、被货物遮挡、自动曝光改变或被 IT 重新编码。产品要提供 tamper detection、安装基准图和一键重校准。

### 15.4 误报摧毁行为闭环

连续几天错误告警后，客户会静音。降低 false alerts per camera-day 往往比再提高 1 个离线 AP 点更有价值。

### 15.5 把人的行为当作机器状态

门的开关具有客观状态；“员工偷懒”“顾客可疑”“情绪异常”具有主观性、歧视和权力风险。早期公司应优先检测物理状态和安全流程，不要靠敏感的人类判断制造收入。

### 15.6 销售承诺超出 site envelope

客户买了“烟火检测”，之后希望同一摄像头顺便识别打架、跌倒、盗窃和库存。产品必须把每个 task 当独立验收单元，而不是默认一个摄像头接入后就获得通用智能。

## 16. 一个可以真正验证商业机会的 30 天试点

不要一开始建大平台，也不要直接训练所有任务。选择一个垂直行业、两个事件、五个现场。

### 16.1 场景选择

例如便利店或小型仓库：

```text
事件 A: 后门持续打开超过 2 分钟
事件 B: 指定消防区域被货物占用
```

两者都有固定 ROI、明确状态和可执行动作，且不需要识别人脸或推断人的意图。

### 16.2 第一周：只验证数据和标签

- 每站点导出 7–14 天历史录像；
- 测量真实 positive 数量；
- 定义 event-level label 与忽略区间；
- 用 VLM 弱标注和 change detection 挖候选；
- 人工核验 contact sheets；
- 估算每站点人工分钟数。

如果连 gold event 都无法稳定定义，立即停止，而不是换更大模型。

### 16.3 第二周：训练 specialist，做时间后置测试

- 每站点独立训练；
- 最后 20% 时间段完全 hold out；
- 对同一事件的连续帧按事件分组；
- 设置 abstention；
- 同时跑通用 VLM、共享垂直模型和 site specialist 三个基线。

核心不是证明 site model 一定最高，而是测量它用多小的算力换来多少现场收益。

### 16.4 第三周：Shadow deployment

- 告警不通知一线人员；
- 记录所有候选和设备状态；
- 每天复核误报；
- 抽样未告警时段寻找漏报；
- 模拟相机移动、遮挡、断网和夜间条件；
- 测量远程恢复能力。

### 16.5 第四周：有限真实联动

- 只对低风险事件发通知；
- 要求接收者确认 `true / false / ignored`；
- 记录告警到处理的时间；
- 计算客户节省的巡检和事件暴露时长；
- 汇总每摄像头的人力、算力和支持成本。

### 16.6 Go / No-Go 标准

在试点前写死阈值，例如：

```text
P95 time-to-activation < 1 business day
human onboarding < 30 minutes per camera
false alerts < 0.2 per camera-day
critical event recall > 95% in accepted envelope
remote issue resolution > 90%
monthly gross margin after support > target
customer confirms measurable action value
```

具体数字应随事件风险调整，但必须提前定义，防止在 Demo 后用另一个指标解释成功。

## 17. 更大的商业含义：AI 把“非标”重新变成一种标准化能力

传统软件喜欢标准化，因为复制的边际成本接近零；传统 AI 也努力训练一个统一模型服务所有客户。但现实世界高度非标：每家店布局不同、每台机器角度不同、每条产线规则不同。

过去，非标意味着人力；现在，非标可能变成 Agent 的输入。

```text
过去的标准化：所有客户使用同一个模型

新的标准化：所有客户使用同一条自动定制流水线，
             每个客户得到不同的模型和阈值
```

这是一种更高层的标准化。就像云平台不要求所有应用运行同一份二进制，而是标准化构建、部署、监控和扩缩容；Site-Specific AI 不要求所有摄像头使用同一模型，而是标准化模型生产和运维协议。

因此，“一座超市一个模型”不是反 SaaS。只要定制过程足够自动化、可观测、可回滚，它反而可能是下一种 SaaS：**Specialization as a Service。**

## 18. 与 Omni 和 Agent 研究的关系

这个方向表面上是 TinyML，内核却非常 agentic。系统需要：

- 理解客户自然语言中的任务和例外；
- 规划如何获得正负样本；
- 调用 VLM、训练器、编译器、设备和 NVR；
- 观察实验结果并修改假设；
- 在量化翻转、类别不平衡和设备故障时诊断；
- 维护每个 site 的模型、数据、硬件和验收状态；
- 在漂移时决定继续观察、重训、回滚还是请求人工。

如果加入实时音视频和自然语言反馈，Omni Agent 还可以在现场安装时与人协作：

```text
“请把摄像头向左转五度。”
“现在冷柜门区域仍被货架遮挡。”
“我需要你模拟一次门半开的状态。”
“这三张属于正常补货，不应告警，对吗？”
```

它不是一个聊天助手，而是一个持续把物理世界编译成可靠数字传感器的工程 Agent。

## 19. 最终判断

车库门案例最值得重视的，不是 ESP32-CAM 便宜，也不是 MobileNet V4 小，而是它把三个原本分离的事实连到了一起：

1. 固定现场可以主动放弃无关的跨场景泛化；
2. 小模型能在这个收缩后的分布上提供便宜、稳定、隐私友好的执行；
3. AI Agent 能够自动完成过去让逐现场定制无法规模化的数据和工程工作。

所以这个商业模式确实非常直球：

> **找到一个状态明确、机位固定、事件有价值的现实问题；用通用 AI 自动理解和构建数据；为每个现场生成一个极度专用的小模型；在边缘持续执行；再用反馈和漂移检测维护它。**

但最后决定它是不是生意的，不是“能不能做出 99% accuracy”，而是四个更朴素的问题：

```text
多久能让一个新摄像头通过验收？
每路摄像头每周还需要多少人工？
误报是否低到客户愿意长期保持告警？
发现事件后，是否真的产生了可计价的动作和结果？
```

如果 Agent 把这四个数字压下来，机器视觉就可能从昂贵的通用算法项目，变成数量巨大的现场专用数字传感器。那时最重要的公司，不一定拥有世界上最强的视觉模型，而是拥有世界上最高效、最可靠的微模型工厂。

## 参考资料

- grapeot. [10小时攻关，半小时 Dev Time：在指甲盖大小的单片机上跑神经网络识别车库门](https://yage.ai/esp32-garage-door-ai.html), 2026.
- Daniel Kang et al. [NoScope: Optimizing Neural Network Queries over Video at Scale](https://arxiv.org/abs/1703.02529), PVLDB 2017.
- Junchen Jiang et al. [Chameleon: Scalable Adaptation of Video Analytics](https://doi.org/10.1145/3230543.3230574), SIGCOMM 2018.
- Romil Bhardwaj et al. [Ekya: Continuous Learning of Video Analytics Models on Edge Compute Servers](https://arxiv.org/abs/2012.10557), NSDI 2022.
- NVIDIA. [TAO Toolkit Documentation](https://docs.nvidia.com/tao/tao-toolkit/text/overview.html).

> 事实边界：车库门项目的开发时长、模型参数、延迟、量化结果和部署表现均来自原作者公开记录，本文未独立复现；NoScope 与 Ekya 的数字来自论文摘要。商业架构、机会矩阵、unit economics、试点阈值和“Site-Specific Model Factory”是本文基于这些案例提出的产品判断，不是上述作者的商业结论。

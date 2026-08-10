# SeedRealtime 音视频全双工大模型：从轮次问答走向连续世界中的自然交互

2026 年 8 月 5 日，字节跳动 Seed 团队发布 [SeedRealtime](https://seed.bytedance.com/zh/SeedRealtime)，将其定义为原生音视频全双工大模型（native audio-visual full-duplex LLM）。与“摄像头拍一张图，再围绕图片问答”不同，它面向持续到来的音频和视频流：模型在用户说话、镜头移动、旁人插话和环境变化时继续更新状态，并决定何时沉默、何时回答、何时主动提醒。

官方演示覆盖多人聚餐、餐馆点菜、博物馆导览、咖啡机操作、论文翻页、机场问路和家庭陪学。把这些演示放在一起看，SeedRealtime 真正有价值的命题并不是“语音聊天增加了视频输入”，而是：**交互对象从一个个离散 turn，扩展为连续变化的 audio-visual world state。**

这篇文章不把发布演示当作已经复现的实验结论。我会分别说明官方披露了什么、演示支持什么判断、哪些技术细节仍未知，并提出一组可以证伪核心能力的 toy experiments。本文依据截至 2026 年 8 月 6 日可获得的官方产品页与技术博客；目前没有公开技术报告、模型权重、训练代码或完整 benchmark 表格。

![SeedRealtime system view](/images/blog/seedrealtime-system-view.svg "图 1：SeedRealtime 的研究对象不是单次音视频问答，而是连续观测、统一状态、交互控制与回复生成构成的闭环。虚线框表示官方尚未公开的内部实现。")

## 1. 一页结论

- **核心变化是从 turn state 到 world state。** 模型不仅要理解用户刚说了什么，还要维护谁在说话、用户在看什么、先前交代的目标是否仍有效，以及现在是否应该介入。
- **音视频联合理解的难点是绑定，不只是识别。** 看见四个人、转写四段语音并不等于知道“哪张脸、哪个名字、哪段声音、哪个偏好”属于同一实体。
- **主动交互可以写成 future-condition trigger。** 用户先给出目标，模型持续观察；当视觉条件满足且介入收益高于打扰成本时，才主动开口。
- **全双工的本质是持续决策。** 听、等、说、让出话权和忽略旁路对话只是可见动作；内部需要不断更新任务状态、指代关系和不确定性。
- **官方披露仍是高层级的。** 已知关键词包括端到端统一音视频建模、chunked input、streaming generation、efficient quantization 和 inference optimization；参数量、tokenizer、视觉帧率、绝对延迟、训练规模与具体模块边界均未公开。
- **“节奏问题减少一半”不能当作完整 benchmark。** 官方称相较级联系统，对话节奏问题减少约一半，打断、延迟和误触发明显下降，但没有给出样本数、对照模型、绝对值、置信区间和标注协议。
- **它展示了 Agent policy 的外形，但尚未证明完整 Agent 能力。** 主动提醒、纠错、联网查询和目标保持已经超出被动问答；可靠工具调用、取消、确认、长期记忆和可逆执行仍缺少公开评测。
- **产品价值与隐私风险同时放大。** 持续感知能显著减少显式 prompt，却也意味着系统更长时间地观察用户和旁观者。部署必须把感知边界、保留周期、主动程度和退出机制设计成一等产品能力。

我的总体评价是：**SeedRealtime 是一次重要的产品级系统展示，它将“原生实时音频”推进到“持续音视频环境中的交互 policy”；但由于缺少可复现实验和绝对时延数据，目前更适合被视为研究方向与产品能力的强证据，而不是已经完成技术定论的开放基线。**

## 2. 它发布的究竟是什么

官方把 SeedRealtime 描述为已经全面上线并规模化部署的原生音视频全双工模型。公开页面给出的系统目标可以归纳为四点：

1. 连续接收音频、视频和文本，而不是等待完整回合结束；
2. 在统一模型中完成感知、理解、决策和回复生成；
3. 一边输出语音，一边继续处理新的用户和环境证据；
4. 基于持续状态主动提醒、纠错或连接外部信息。

这里需要避免两个误解。

第一，**full duplex 不等于两个 WebSocket 同时传输。** 工程上同时上传麦克风和下载音频并不难；困难的是模型在自己说话时仍能识别新的用户意图，并在打断发生后修正正在生成的内容。

第二，**audio-visual 不等于把每帧字幕化。** 如果系统先对图像做离散描述，再让文本 LLM 处理，细粒度时序、指向、说话人与对象关系会在中间接口损失。官方强调音频、视频和文本在统一架构中对齐，意图正是减少这种级联损失。不过“统一”具体发生在 tokenizer、表示层、backbone 还是训练目标层，目前没有公开结构图可以确认。

## 3. 原生音视频全双工到底意味着什么

可以把普通多模态问答写成：

```text
完整图像/视频 + 完整问题 -> 一次答案
```

而音视频全双工系统面对的是时间序列：

```text
o_t = audio_chunk_t + video_chunk_t + playback_t + tool_event_t
h_t = update(h_(t-1), o_t)
a_t = policy(h_t)
```

其中 `h_t` 不应只是文字 transcript，而至少包含五类状态：

| 状态 | 要回答的问题 | 失败表现 |
| --- | --- | --- |
| Perceptual | 当前看见、听见了什么 | 漏掉目标、把背景声当用户 |
| Referential | 谁在说、看谁、指什么 | 人脸、声纹、名字和偏好错绑 |
| Task | 用户此前交代的目标是否有效 | 忘记提醒、重复旧任务 |
| Interaction | 该听、等、说、让出还是提醒 | 抢话、迟答、误触发 |
| Action | 是否查询、纠错或执行 | 工具调用过早、结果与场景脱节 |

输出也不是单一答案，而是一个持续 policy：

```text
LISTEN | HOLD | BACKCHANNEL | SPEAK | YIELD | ALERT | QUERY | CORRECT
```

这解释了为什么“自然接话”不是孤立的 TTS 功能。模型若不知道对话是否在指向自己、任务是否已经满足、用户是否正在自我修正，就不可能稳定决定什么时候开口。

## 4. 官方公开的技术边界

官方博客披露了以下高层信息：

- 端到端统一音视频建模；
- 感知、理解、决策与回复生成由一个模型完成；
- 原生支持 full-duplex；
- 连续建模 conversation state 与 timing；
- 使用分块音视频输入和流式生成；
- 通过高效量化和推理优化服务在线场景。

这些信息足以确认设计目标，却不足以复现系统。当前仍不能确定：

- 音频与视频使用什么 tokenizer，速率分别是多少；
- 是否存在独立的 speech encoder、vision encoder、speaker diarization 或 VAD；
- 模型是单一自回归流、多流 head，还是异步控制器；
- 视频是固定 FPS、事件触发采样，还是动态分辨率；
- 语音输出期间如何抑制回声并区分用户与自身播放；
- 长时间任务状态保存在 KV cache、显式 memory 还是外部状态机；
- 工具调用是否进入同一 policy，还是由外部 router 决定；
- 参数规模、训练数据规模、后训练目标和安全策略。

因此，文章中不能把它补写成 Thinker-Talker、MoE、多 codebook 或某一种已知开源架构。**相同的产品行为可以由多种内部实现产生；没有技术报告时，最稳健的研究方法是分析可观察能力和待验证假设。**

## 5. 最难的不是“看见”和“听见”，而是绑定

多人聚餐演示中，用户逐一介绍姓名，模型根据外形识别人物，并在后续交谈中把声音、身份和个人偏好关联起来。旅行计划又要求模型记住“谁怕热、谁怕累、谁对海鲜过敏”。这不是四个独立分类器的简单相加，而是一个动态实体图：

```text
name "七七" <-> face track #2 <-> voice cluster B
              <-> utterance history <-> preference: photography
```

当人转身、离开画面、换座、交叉说话或被遮挡时，绑定必须保持；新证据与旧证据冲突时，还要允许修正。真正的难点包括：

- face track 在镜头移动后能否重识别；
- 声音重叠时是否仍能归属正确说话人；
- 名字是指当前画面中的人，还是谈话中未出现的人；
- 手指方向、视线和口头“这个”指向哪个对象；
- 身份不确定时，模型是否会保留多个假设而不是强行绑定。

![Binding state](/images/blog/seedrealtime-binding-state.svg "图 2：多人音视频交互需要持续维护 face、voice、name、utterance、gaze 与 preference 的绑定图，并对冲突证据进行修正。")

川菜馆演示强调另一类 grounding：服务员说“很下饭”时，模型要知道这句话指向刚端上来的菜；用户问皮蛋时，又要把当前画面、菜单语义、文化知识和英文解释合并。真正要验证的是联合证据是否改变了答案，而不是系统能否分别识别菜品和转写语音。

## 6. 主动交互：把未来条件保存在状态里

博物馆和论文翻页是最能体现系统变化的两个案例。用户不是在目标出现后提问，而是预先声明：

```text
“看到错金银铜虎噬鹿屏座就提醒我。”
“翻到训练参数那部分提醒我。”
```

这类任务可写成一个持久触发器：

```text
goal g = detect(target) and notify(user)
for each observation o_t:
  belief_t = P(target present | o_1:t)
  if belief_t > threshold and utility(alert) > interruption_cost:
      ALERT
      mark g completed
```

它至少需要目标编码、持续感知、跨时间记忆、阈值决策和完成状态。若只做逐帧视觉问答，模型不会知道数分钟前的口头要求仍然有效；若每帧都重复查询，又会产生高成本和大量误报。

咖啡机演示进一步增加了过程状态。模型先识别“整粒咖啡豆直接进入萃取手柄”这个错误，再根据萃取后的液量和油脂给建议。它需要区分：当前观察是准备、错误操作、纠正、执行还是完成，而不是只识别静态物体。

![Proactive loop](/images/blog/seedrealtime-proactive-loop.svg "图 3：主动交互不是看到目标就立即说话，而是目标保持、连续证据、置信度更新、打扰成本和完成状态共同构成的闭环。")

主动性也带来一个容易被忽略的目标冲突：更高 recall 往往意味着更多无必要提醒。产品不能只优化“该提醒时提醒”，还必须同时控制：

```text
helpful intervention
- unnecessary interruption
- missed intervention
- late intervention
- repeated intervention
```

一名一直插话的助手，即使视觉识别很准，也不会带来更自然的体验。

## 7. 自然时机与旁路误触发

官方称，SeedRealtime 不再完全依赖外部 VAD 规则决定轮次，而是根据多模态上下文持续判断何时接话、沉默和响应。在机场演示中，同伴聊天提到航班没有触发模型；用户随后正式询问时，模型才使用此前看到的大屏信息回答。家庭学习演示中，父亲电话声也没有让模型偏离女孩的学习任务。

这要求系统估计的不只是 speech activity，而是 addressee 与 interaction relevance：

```text
P(this speech addresses assistant)
P(this speech changes current task)
P(user has yielded the floor)
P(speaking now is more useful than waiting)
```

外部 VAD 只能告诉模型“有人在发声”，不能告诉它“这句话是不是对我说的”。视频中的视线、身体朝向、手势、人与设备距离，以及历史任务归属都可能改变判断。

但“由模型决定”也不等于彻底没有外部组件。生产系统仍可能使用回声消除、声学 VAD、降噪或安全中断机制；合理目标不是消灭所有模块，而是避免由一个硬编码 endpoint 独占交互决策。官方没有披露这些工程边界。

## 8. 如何读“节奏问题减少一半”

产品页表示，相比 cascaded models，SeedRealtime 的 conversational pacing issues 减少一半，interruptions、latency 和 false triggers 显著下降，单轮对话的流畅性与完整性 usability 更高。这是值得关注的人评信号，但还不是可独立审计的 benchmark。

一项完整声明至少需要：

| 缺失信息 | 为什么重要 |
| --- | --- |
| 对照系统 | 强级联和简单级联差异很大 |
| 样本与场景数 | demo 场景不能代表开放世界分布 |
| pacing issue 定义 | 抢话、长停顿、重复、误停应分别统计 |
| 绝对值 | 从 2% 到 1% 与从 40% 到 20% 都是“减半” |
| 标注者和协议 | 主观流畅性高度依赖说明与语言 |
| 置信区间 | 小样本人评可能有很大波动 |
| 延迟分位数 | 平均值会掩盖 P95/P99 尾部卡顿 |

在没有这些信息前，正确写法是“官方人评宣称改善”，而不是“SeedRealtime 已证明延迟领先”。特别是官方没有给出绝对毫秒级延迟，所以无法与 Moshi 的公开数字或其他系统的首包时间直接横比。

![Evidence map](/images/blog/seedrealtime-evidence-map.svg "图 4：当前证据由官方披露、产品演示和研究推断三层组成。越接近可复现 benchmark，公开证据越少。")

## 9. 与 GPT-4o、Moshi、Qwen-Omni 等系统怎么比较

不同系统的产品接口和论文协议不一致，不能用一张“谁最快”的表制造虚假精度。更合理的是比较公开能力与证据边界：

| 系统 | 原生实时音频 | 连续视频重点 | 主动视觉触发 | 公开绝对延迟 | 开放权重/代码 | 主要价值 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-4o / Realtime | 是 | 产品支持视觉，但协议随产品演进 | 有产品潜力，公开系统评测有限 | 部分产品指标 | 否 | 大规模通用实时交互 |
| Moshi | 是，全双工 | 否，重点为 speech | 非主要目标 | 论文与仓库给出约 200 ms 级系统数据 | 是 | 可研究的双流语音基线 |
| Qwen-Omni 系列 | 音视频输入与实时语音 | 是 | 取决于版本和部署 | 论文披露更具体 | 部分开放 | 结构、训练与 benchmark 透明度较高 |
| FLAIR | 全双工 speech research | 否 | 否 | 研究协议内评估 | 以论文状态为准 | 监听阶段 latent cognition |
| DuplexSLA | speech-language-action 时间轴 | 非主要目标 | action 可并行 | 论文协议 | 当前公开状态有限 | 把工具动作纳入全双工 policy |
| SeedRealtime | 是 | 是，持续视觉是核心卖点 | 官方 demo 明确展示 | 未公开 | 未公开 | 产品级音视频持续状态与主动交互 |

SeedRealtime 相比语音全双工工作的差异，不只是多一个 vision encoder，而是视觉信息长期参与 **addressee detection、指代绑定、future trigger 和 action timing**。相比结构公开更详细的学术模型，它的优势是官方宣称已经大规模部署；弱点是外部研究者无法核对结构和指标。

## 10. 它真正证明了什么，又没有证明什么

官方演示至少支持三个重要判断。

第一，**持续视频可以成为交互控制信号。** 视频不只是回答内容的证据，还决定是否应主动叫停、纠错或继续沉默。

第二，**自然交互需要跨模态 common ground。** 机场屏幕已离开画面，用户随后才提问；模型必须保存“我们共同看过什么”，而不是只保留当前帧。

第三，**全双工与 Agentic 能力正在合流。** 当模型持续维护目标并依据环境行动时，它已经不再是 turn-based chatbot。

但演示没有证明：

- 在随机、多样、长时间场景中的成功率；
- 多人绑定在换位、遮挡和重叠语音下的稳定性；
- 主动提醒的误报率与用户容忍度；
- 语音打断后的停止延迟和状态回滚；
- 工具调用的正确率、取消能力和不可逆操作安全；
- 长时间运行的成本、热稳定性与 P95/P99 延迟；
- 不同语言、口音、设备和网络条件下的鲁棒性。

发布视频是寻找研究假设的入口，不是替代测试集的证据。

## 11. 七个可以立即做的 toy experiments

与其一开始就讨论大规模训练，不如先把每个核心主张拆成最小可证伪实验。

### 11.1 Speaker-face permutation

两个人先自我介绍，随后交换座位；再用后处理交换两人的音轨或声线。分别测姓名、脸、声音和偏好的绑定准确率。若交换音轨后系统仍按脸回答，说明 voice evidence 没有真正进入联合推断。

### 11.2 Counterfactual modality

构造四组：原始音视频、静音视频、黑屏音频、错配音视频。问题必须只有联合证据才能回答。若错配时答案不变，所谓 audio-visual understanding 可能只是单模态捷径。

### 11.3 Distractor speech injection

在主任务中加入旁人、电视、电话和机场广播，逐级提高信噪比和语义相似度。报告 false trigger rate、addressee error、任务状态漂移和恢复时间，而不是只报 ASR WER。

### 11.4 Persistent visual trigger

用户先交代目标，目标在 30 秒、2 分钟、5 分钟后出现，中途加入相似干扰物。测命中率、误报率、提醒延迟和重复提醒率，以检验目标保持和视觉 memory。

### 11.5 Visual correction and supersession

用户先做错、随后撤销并做对。模型应停止旧建议、承认状态已更新，且不在之后继续引用旧错误。这直接测试 state supersession，而不是单帧识别。

### 11.6 Proactivity cost curve

改变主动阈值，联合绘制 helpful intervention、unnecessary interruption 和 missed intervention。最优点取决于场景：学习陪伴可以多提醒，会议和驾驶场景则应更保守。

### 11.7 Playback-aware barge-in

让用户在助手说到不同位置时打断。记录从用户起声到助手静音的延迟、模型之后是否引用用户从未听见的内容，以及被打断计划能否正确取消。

这七个实验不要求先训练新模型，却能快速判断系统是否真的维护了联合状态，以及失败发生在感知、绑定、任务保持还是 interaction policy。

## 12. 从实际产品看，还缺少哪些能力

连续感知产品的门槛不只在模型分数，还在“是否值得一直开着”。至少要解决以下系统问题。

### 可控的主动程度

用户需要按场景选择安静、普通和主动模式，并能对单个任务设置“只提醒一次”“仅高置信度提醒”或“先震动再说话”。主动性若不可控，会很快变成打扰。

### 明确的感知与记忆边界

界面应让用户知道摄像头、麦克风和短期记忆何时开启；哪些内容只在设备上处理，哪些上传；旁观者数据是否保留；如何一键删除当前会话状态。

### 可检查的当前任务

对于“看到某物提醒我”这类持续任务，产品应提供简洁任务列表：正在观察什么、何时过期、是否已完成。纯隐式 memory 很难纠错，也容易造成意外持续监控。

### 失败时的降级

网络拥塞、摄像头遮挡、强噪声或多人绑定不确定时，模型应表达不确定性、请求确认或退化为普通语音模式，而不是用流畅语音掩盖状态错误。

### 工具动作的权限层级

查询和预加载可以提前做；发送、购买、删除、预订等不可逆动作需要 prepare-confirm-commit。实时不意味着越早执行越好，而是更早准备、在证据充分时安全提交。

## 13. 隐私和社会边界不是附录

持续音视频模型天然会捕获旁观者。多人聚餐、机场和家庭演示也正是隐私最复杂的场景：并非所有进入镜头或麦克风的人都同意被识别、绑定身份和建立偏好记录。

需要区分至少四类数据：

```text
ephemeral perception  只用于当前时刻推断
session state         当前会话内保持
user-approved memory  用户明确选择长期保存
tool record           外部查询与业务操作日志
```

默认把一切写入长期 memory 会让主动交互变成持续画像。更合理的原则是数据最小化、默认短暂、敏感绑定需确认、用户可查看与撤销，并对旁观者使用更严格的保留规则。

另一个风险是“好心但错误的介入”。咖啡机建议的代价较低；医疗、驾驶、工业设备中的主动纠错则可能影响安全。系统需要把视觉置信度、行动风险和人类确认合并，而不是用单一识别阈值决定是否开口。

## 14. 从 SeedRealtime 到 continuous-time Agent Policy

把官方展示抽象后，可以得到一个更一般的交互式 Omni policy：

```text
observations:
  audio, video, playback, tool results, memory reads

latent state:
  entities, beliefs, goals, common ground, uncertainty, action status

actions:
  listen, speak, yield, alert, query, prepare, commit, cancel, remember
```

![Agent gap](/images/blog/seedrealtime-agent-gap.svg "图 5：SeedRealtime 已展示连续感知、状态保持和主动开口；走向完整 Agent 还需要显式工具生命周期、安全提交、可撤销 memory 和可审计评测。")

这一视角比“语音输入、语音输出的聊天模型”更准确。自然停顿、附和和打断是 policy 的表现层；内部真正困难的是：

- 在信息不完整时维护多个假设；
- 判断等待的价值与提前行动的收益；
- 在用户修正后撤销旧绑定和旧计划；
- 区分已生成、已播放和用户实际听到的内容；
- 在持续视觉中保留任务，但不过度保存个人信息；
- 将主动帮助的收益与打扰、误报和安全风险统一优化。

SeedRealtime 的博物馆与机场案例已经显示出前三步的产品形态，官方未来方向也明确提到连接查询、预订与任务办理。下一阶段的关键不只是更低首包延迟，而是把工具、memory、权限和 rollback 纳入同一连续时间决策过程。

## 15. 我希望下一份技术报告回答什么

如果 Seed 团队后续发布技术报告，最有价值的信息不是再增加几个 demo，而是给出可复现的系统 contract：

1. 音频与视频 token rate、chunk size 和跨模态对齐方式；
2. 端到端 P50/P95/P99 响应、打断停止和主动提醒延迟；
3. 多人绑定、旁路误触发、长期视觉触发与 counterfactual modality benchmark；
4. “节奏问题减半”的完整人评协议和绝对值；
5. 语音输出期间的持续监听、回声处理和 playback state；
6. 短时状态、显式 memory 与工具结果的系统边界；
7. 主动交互的误报成本、安全阈值和用户控制；
8. 不同设备、语言、噪声和网络条件下的部署结果；
9. 与强级联基线在质量、延迟、成本和可维护性上的同条件比较。

这些指标会让“自然交互”从演示感受变成研究共同体可以验证的对象。

## 16. 最终评价

SeedRealtime 最值得关注的地方，不是把视频接入语音模型，而是让视频成为持续交互 policy 的一部分：它影响指代、目标保持、开口时机、主动提醒和工具上下文。模型不再只回答“这一帧里有什么”，而要处理“我们刚才共同看过什么、用户让我继续留意什么、现在出现的变化是否值得打断”。

这也是 Omni Model 从 multimodal model 走向 embodied interaction system 的关键一步。未来竞争不会只看识别准确率或语音自然度，而会看：

```text
是否理解当前共同情境
是否在正确时间做正确动作
是否能在错误后低成本修复
是否让用户持续掌握控制权
```

在公开证据层面，SeedRealtime 已经给出了很强的产品方向信号，却尚未给出足以比较和复现的技术细节。对研究者而言，最好的回应不是猜测其内部结构，而是把这些演示拆成 speaker-object binding、persistent trigger、false-trigger suppression、proactivity cost 和 playback-aware repair 等可证伪问题。只有这些问题被系统测量，全模态自然交互才会从“看起来很自然”走向“在复杂现实中可靠”。

## 参考资料

1. ByteDance Seed, [SeedRealtime 产品页](https://seed.bytedance.com/zh/SeedRealtime), 2026-08-05.
2. ByteDance Seed, [SeedRealtime 音视频全双工大模型发布：走向全模态自然交互](https://seed.bytedance.com/zh/blog/seedrealtime-audio-visual-full-duplex-llm-released-toward-omni-modal-natural-interaction), 2026-08-05.
3. ByteDance Seed, [SeedRealtime: An Audio-Visual Full-Duplex LLM](https://seed.bytedance.com/en/SeedRealtime), 2026-08-05.
4. OpenAI, [Hello GPT-4o](https://openai.com/index/hello-gpt-4o/), 2024.
5. Défossez et al., [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037), 2024.
6. Chu et al., [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215), 2025.
7. [The Silent Thought: Modeling Internal Cognition in Full-Duplex Spoken Dialogue Models via Latent Reasoning](https://arxiv.org/abs/2603.17837), 2026.
8. [Learning When to Think While Listening in Large Audio-Language Models](https://arxiv.org/abs/2605.27190), 2026.

> 说明：截至本文写作时，SeedRealtime 尚未公开论文、权重、训练代码或完整 benchmark。文中的架构抽象、状态分解和实验设计属于基于官方可观察行为提出的研究分析，不代表官方技术实现。

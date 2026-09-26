# InferPort 首版设计研究

记录日期：2026-09-26。此处保存 InferPort 首版选型研究和实施阶段，供追溯设计原因。
协议库首版实现已完成；这些内容不是当前待办，也不代表外部模型接入已经完成。
当前接口以 [协议与接口参考](../inferport-design.md) 为准；下一阶段见
[真实推理验证计划](../integration-validation.md)，实测范围见 [验证概览](../validation.md)。

## 首版实施阶段（历史）

| 阶段 | 工作 | 可检查的交付物 |
|---|---|---|
| A：包与数据契约 | 新包骨架、依赖、错误、codec、v1 消息校验 | 可构建 wheel；codec/protocol 正反例通过 |
| B：通信与状态 | Client、serve、ready、独占、工作线程、超时与关闭 | 独立进程调用和生命周期故障测试通过 |
| C：安装与示例 | 文档、无状态与有状态示例、跨版本与安装矩阵 | 干净环境可安装；无模型依赖调用成功 |
| D：性能与首版复核 | 固定 payload 基准、限制检查、打包内容审阅 | 有可复现结果，明确仍存在的验证缺口 |
| E：外部真实适配 | 在模型/执行仓库接入 PI05 等选定后端 | 模型、processor 和环境语义由所属仓库验证 |

A–D 构成独立库首版实现。E 是独立的集成验证工作，已有适配器可作为首个候选；
真机、权重、设备或第二个实际模型未指定时，不能用 mock 测试宣称已完成这些集成。
不要求先接入多种真实机器人才能完成协议库，但也不以协议库测试代替闭环验证。
下一阶段按 [真实推理接入与验证计划](../integration-validation.md) 执行 E，记录实际适配与运行结果。

## 设计阶段诊断与参考项目

以下保留设计阶段的临时诊断作为选型依据；实际实现结果以 [验证记录](../validation.md) 为准：

- OpenPI 编码在 Python 3.10 / NumPy 1.26.4 与 Python 3.12 / NumPy 2.4.4 上通过 20 类数组往返，
  做过跨环境文件解码；其解码存在接受多余字节、-1 维度、收发 dtype 规则不一致和标记字典碰撞的问题。
- 临时 WebSocket 示例完成约 1.84 MB 数组回环、接收超时与显式重连。这些测试未覆盖本设计的
  完整客户端实现、TLS、状态清理、新数组扩展格式或所有 Python 版本。
- 本轮新增“不再读取数据的对端”诊断：16 MiB 发送使用 0.2 秒期限。直接依赖库的 close_timeout
  时关闭等待未结束；采用公开连接扩展点保留 transport、额外 0.1 秒关闭预算并 abort 后，
  Python 3.10 / websockets 16.1.1 和 Python 3.12 / websockets 17.1 都在约 0.30 秒完成退出。
  这验证了有界网络回收的实现路径，不能替代最终 SDK 的全链路故障测试。
- 候选依赖做过部分 Linux x86_64 / ARM64 二进制解析；这不代替目标 wheel 的安装和运行矩阵。
- 当前没有真实模型、仿真闭环或真机性能结果，也没有原生 TCP 与 WebSocket 的公平网络性能比较。

设计借鉴与适用范围：

| 参考 | 采用的经验 | 留在原项目/暂缓的部分 |
|---|---|---|
| OpenPI | 独立轻客户端、二进制数组表示 | 不照搬无限消息大小、模型共享状态或 codec 的宽松解码 |
| GR00T | 薄模型包装、明确 reset 与错误返回 | 不采用其模型类耦合、任意 endpoint 注册或 ZMQ 传输 |
| LeRobot | 模型/执行分离、processor 状态需要被重置 | Robot 类、执行队列、RTC、gRPC 服务结构 |
| RLinf | 输出数据归属、训练与执行数据分离 | Ray/Worker 调度、动态 batch 聚合、NCCL/Gloo 通信体系 |
| APXinf-robo（RLinf 组织的独立项目） | 线程绑定资源初始化、预处理归属和数值一致性检查 | 引擎、机器人预设、OpenPI 协议兼容层 |

### RLinf 补充对照

2026-09-26 查阅官方 main 分支相关源码与 latest 文档；以下是静态对照，未运行 RLinf 或 APXinf 集成。
这些链接会随上游更新，不表示已经固定或验证某个完整上游运行环境。

RLinf 的 Channel 提供 Worker 间队列、路由和 batch 聚合，配合 Ray 与 PyTorch distributed 通信。
其 NCCL/Gloo 和 CUDA IPC 路径服务于分布式训练工作负载；另有面向 SGLang 的 HTTP 客户端。
同一组织的独立仓库 APXinf-robo 提供 OpenPI 兼容 WebSocket 服务。
这几种用途不改变 InferPort 首版的一模型、一活动连接范围。
参见 [Channel 源码](https://github.com/RLinf/RLinf/blob/main/rlinf/scheduler/channel/channel.py)、
[通信说明](https://rlinf.readthedocs.io/en/latest/rst_source/concepts/collective.html)、
[HTTP 客户端](https://rlinf.readthedocs.io/en/latest/rst_source/guides/inference_http_client.html) 和
[APXinf-robo 服务](https://github.com/RLinf/APXinf-robo#openpi-compatible-serving)。

RLinf 的 [rollout 动作发送实现](https://github.com/RLinf/RLinf/blob/main/rlinf/workers/rollout/hf/huggingface_worker.py)
显式处理 CPU 数据交付，源码说明其目的包括避免 CUDA IPC 共享的输出缓冲区被后续推理覆盖。
InferPort 在后端串行工作线程中完成返回数组的编码，再执行下一次调用；接收端重建独立可写数组。
适配器仍须保证编码期间没有外部线程修改返回缓冲区。补充测试检查数组复用、非连续视图、
显式重置和断连清理后，之前的客户端结果保持不变。

APXinf-robo 的 [bare 模型加载接口](https://github.com/RLinf/APXinf-robo/blob/main/src/apxinf_robo/engine.py)
注明模型句柄绑定创建线程。适配时 Backend 构造函数只保存配置或工厂，在工作线程首次 reset 时加载，
后续 reset 清理会话状态，服务结束时在同一线程释放。NumPy 示例与测试验证这一生命周期用法，
不代替特定 CUDA 引擎的集成验证。

RLinf 的 [具身数据接口](https://rlinf.readthedocs.io/en/latest/rst_source/reference/api/embodied_data.html)
将环境执行需要的动作与训练所需的 log-probability、value、版本和轨迹数据放在不同路径中。
InferPort 保持通用 Payload，由适配器约定字段、batch 顺序和状态归属，不引入这些训练数据类。
首次接入真实模型时，按相同输入、模型状态和可控随机性比较原生接口与远程调用的输出，
验证图像布局、预处理/归一化、动作后处理和 reset，数值容差由具体模型精度与推理特性决定。

### 其他参考源码

[OpenPI 数组编码](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/msgpack_numpy.py)、
[OpenPI 服务端](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/serving/websocket_policy_server.py)、
[GR00T 服务接口](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/gr00t/policy/server_client.py)、
[LeRobot 异步客户端](https://github.com/huggingface/lerobot/blob/1bc0bdfb20ad4f4f76dc8a68a0e0746d54f50de9/src/lerobot/async_inference/robot_client.py)。

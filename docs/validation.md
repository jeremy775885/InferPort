# InferPort 验证概览

更新日期：2026-10-02。历史发布基线：`v0.1.0` / `3cc9077779fa072f0a77a32c86cc842b19d5b6b8`。
本文汇总已取得的证据及适用范围；逐轮环境、命令和实验细节保存在
[0.1.0 验证归档](archive/v0.1.0-validation.md)。后续源码改动应以对应提交的检查结果为准。

## 0.2.0 发布前检查 — 2026-10-02

发布前按锁文件完成开发环境同步；源码完整测试 **204 passed（8.53 s）**，ruff check/format、
wheel/sdist 构建和 `twine check --strict` 全部通过。版本为 0.2.0，通信标识为 `inferport`。
本机的默认镜像覆盖曾使锁文件检查要求改写源地址；隔离该覆盖后，原锁文件在 CI 默认配置及
原索引下均通过检查。没有保留依赖升级、镜像迁移或锁文件改写。
GitHub 完整 CI 和正式上传由发布工作流执行；实际发布结果在验证完成后补充。

## 0.2.0 / inferport 候选版本检查 — 2026-10-02

发布者将候选包版本从 0.1.1 调整为 0.2.0，以体现契约 API 的改动规模。
WebSocket subprotocol 仍为不带版本后缀的 `inferport`，不恢复旧协议标识。
本次仅修改包版本、安装提示和说明；以下 0.1.1 测试保留其原始版本身份。
此前同为 0.2.0 的开发候选使用过不同握手标识，不能只凭包版本判断是否为当前构建。

当前版本的打包元数据测试、wheel/sdist 严格检查和实际 wheel 导入通过。十个运行模块与已评审
0.1.1 wheel 相比，仅 `__version__` 字符串变化；通信标识仍为 `inferport`。未重跑完整矩阵或 GPU。

## 历史候选：0.1.1 / inferport（未发布）— 2026-10-02

发布者选择将本轮接入修正作为 0.1.1 维护版本发布，WebSocket subprotocol 统一为不带版本后缀的
`inferport`。先前 0.2.0 只是未发布候选。必需的 describe、数据契约和收发边界校验均保留；
旧后端仍需补齐 describe，两端需使用当前 SDK，不提供旧接口或握手标识兼容分支。

- 源码完整测试 **204 passed**，真实 wheel 在仓库外独立环境测试 **204 passed**。
  新增固定字符串 `inferport` 的真实握手/describe/infer 检查，拒绝旧的带版本标识连接。
  既有消息、超时、状态、NumPy 标量和契约回归均通过；ruff check/format 通过。
- 独立 wheel 环境：Python 3.12.3、NumPy 2.5.3、msgpack 1.2.3、websockets 17.1。
  临时环境复用先前独立测试环境的相同依赖文件，排除旧 InferPort，再安装新 wheel；
  包导入来自新环境的 site-packages，依赖一致性检查通过。
- wheel/sdist 构建与严格元数据检查通过；最终 wheel 与实际安装并通过完整测试的 wheel 字节一致。
- 同一个 wheel 临时加入两个消费方的导入路径，LeRobot 服务/转换 **50 passed**（90 条既有警告），
  RoboTwin CPU/真实通信/录制/进程清理 **65 passed**；Python 3.10/3.12 双向 echo/policy 共四组通过。
  两个消费环境的包版本清单前后完全一致，未替换其已安装包，也未操作正在运行的模型服务。
- 运行源码逐文件对照：除包版本、协商标识和诊断文字外，全部与此前已验证实现相同；
  LeRobot 仅修改安装提示，RoboTwin 评测实现不变。新契约和校验逻辑未删减。

本候选未发布；新版全 CI 矩阵、跨机器网络和真实硬件仍未验证。本次没有新 GPU 运行。
此前接入方已在原 0.2.0/带版本协议候选上完成 PI05/RoboTwin smoke 1/1、正式评估 10/10；
该成绩保留其原始源码和协议身份，不能改记为本次 0.1.1 的新实测。

## 历史开发阶段：0.2.0 候选（未发布）— 2026-10-02

本地源码已实现 inferport.v2、必需的 describe、通用规格校验和 joint-targets.v1 profile，尚未发布。
所有后端必须声明契约；不保留旧后端的无规格分支。
Client 在连接期限内取得契约并缓存，收到结果后自动校验；业务字段与扩展由接入方声明。
本轮验证与下文 0.1.0 CI/发布证据分开：

- 源码完整测试 **202 passed**；ruff check/format 通过。
- 实际 wheel 在仓库外独立环境完整测试 **202 passed**：Python 3.12.3、NumPy 2.5.3、
  msgpack 1.2.3、websockets 17.1。独立进程导入/打包检查不依赖 Torch 或机器人 SDK。
- 两个已有消费环境以实际 wheel 双向调用：Python 3.10.20 / NumPy 1.26.4 /
  websockets 16.1.1 与 Python 3.12.3 / NumPy 2.2.6 / websockets 17.1，msgpack 均为 1.2.3。
  显式声明规格的 echo 与机器人策略各双向通过，共四组；echo 包含后端返回 NumPy
  整数/浮点/bool 标量，另有固定/有界动作 horizon 和数值检查。
- v1 握手拒绝、未知/非法规格、语义/通道顺序不匹配、无状态变更的输入/context 拒绝、
  非法输出断连、描述快照与线程归属均有测试。缺失 describe 无法实例化，None/非法声明在
  READY 前失败，远端空声明关闭客户端连接。短 chunk 和可选诊断由消费方测试覆盖。
- 独立 wire peer 绕过服务端输出校验，验证客户端拒绝错误形状、dtype、NaN、缺失/多余字段并关闭连接。
  契约读取超时、describe 缓存不产生 RPC、reset 保留缓存和应用自定义字段均有回归覆盖。
- 修复后端 NumPy 标量在编码前被错误拒绝的问题。新增 28 个回归用例覆盖所有支持的
  bool/整数/浮点宽度、重复 RPC 与原始输出保留、bool/数字类型分离、NaN/Inf、精确整数
  上下界、零维数组及不支持的 NumPy 标量。非法输入仍在 backend 调用前被拒绝；
  非法输出仍触发致命错误。首个回归在修复前复现 invalid_output，修复后通过。
- 上一轮已验证数值/有状态示例及基准载荷；本轮未运行新性能实验。
- LeRobot PI05 服务公共观测到原模型特征的 CPU 边界测试及转换回归共 **50 passed**，
  非法输入/context 改为经真实 SDK 拒绝且不推进模型状态；RoboTwin CPU/真实通信/原生录制/
  进程清理测试 **65 passed**。仅涉及消费方自身仓库的实现。
- wheel/sdist 构建及 `twine check --strict` 通过。锁文件保持原依赖与源地址；新版本号由源码提供。

在该轮源码检查时，新版 CI 全矩阵、Windows/macOS、最低 NumPy、GPU 闭环、跨机器网络与真实硬件均未执行。
本轮没有发布、推送或服务进程接管。规格兼容不证明 checkpoint 相同或策略成功率。

复现库侧验证：

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv build --no-sources --out-dir /tmp/inferport-dist
uvx twine check --strict /tmp/inferport-dist/*
# 跨环境：替换为两个已安装同一候选 wheel 的 Python
uv run --no-sync python tests/cross_environment.py --python-a /path/a/python --python-b /path/b/python
```

## 0.1.0 发布基线（历史）

| 项目 | 已验证范围 |
|---|---|
| 核心功能 | `Backend` / `Client` / `serve`、codec、协议、生命周期、错误与超时、可选认证和 TLS；完整测试集 144 项 |
| Python / NumPy | Linux CI 覆盖 Python 3.10–3.14、最低依赖与常见 NumPy 组合；下文列出本地环境证据 |
| 跨环境调用 | Python 3.10 / NumPy 1.21.3 与 Python 3.12 / NumPy 2.x 的独立 wheel 环境，互换两端角色通过 |
| 其他平台 | Windows/macOS 通过 CI smoke 子集；Linux ARM64 仅完成二进制依赖解析 |
| 打包与发布 | wheel/sdist 构建、元数据检查、GitHub Release、Trusted Publishing 上传和官方 PyPI 安装通过 |
| 性能 | 本机回环、独立进程、echo 后端的数组传输基准；尚无跨机器部署或真实模型性能结果 |
| 真实接入 | 模型、processor、仿真或真机闭环均未验证；执行步骤见 [真实推理验证计划](integration-validation.md) |

## GitHub Actions

[首次 CI](https://github.com/jeremy775885/InferPort/actions/runs/36215291050)
对应提交 `fb7f0b01fd324f5d5f1989e35e9c6efb863c5ccf`，11 个任务全部成功：

- 8 个 Linux 任务：Python 3.10–3.14，以及 Python 3.10 最低依赖、
  Python 3.11 / NumPy 1.23.5、Python 3.12 / NumPy 1.26.4。
- 1 个跨环境任务：从 wheel 安装，在 Python 3.10 / NumPy 1.21.3 与 Python 3.12 / NumPy 2.x 之间双向调用。
- 2 个 Windows/macOS smoke 任务：覆盖 codec、protocol、lifecycle、adapter contracts 与 packaging。

[0.1.0 发布工作流](https://github.com/jeremy775885/InferPort/actions/runs/36218657260)
在发布提交 `3cc9077` 上再次复用完整 CI，随后完成构建、wheel 测试和 PyPI 上传，结果为 success。
Windows/macOS 的 smoke 结果不代表完整网络故障测试或实际机器人部署已通过。

## 本地功能与安装验证

下列环境均运行过完整的 **144 项测试**。本地功能检查、依赖调整和发布准备各轮的范围见
[详细归档](archive/v0.1.0-validation.md)；早期 141 项测试记录不视为自动补跑了新增测试。

| 安装方式 | CPython | NumPy | msgpack | websockets |
|---|---|---|---|---|
| editable | 3.12.3 | 2.4.4 | 1.2.2 | 17.1 |
| 构建的 wheel | 3.10.17 | 1.21.3 | 1.1.0 | 16.1.1 |
| 构建的 wheel | 3.11.12 | 1.23.5 | 1.1.0 | 16.1.1 |
| 构建的 wheel | 3.12.3 | 2.5.3 | 1.2.2 | 17.1 |

功能测试包含畸形消息、数组边界、发送阻塞、握手/READY 超时、断连清理、后端独占、迟到结果隔离、
跨线程关闭、Ctrl+C、TLS 校验、输出数组复用和线程绑定资源生命周期。
500 次连续调用检查了线程及文件描述符数量，不等于长时间内存浸泡测试。

NumPy 1.21.3 / 1.23.5 / 1.26.4 的兼容 Python 环境先安装 NumPy 再安装 InferPort，版本保持不变。
Ruff、sdist/wheel 构建、严格元数据检查及依赖检查通过；发布工作流也通过 actionlint 检查。
完整的回归要求见 [协议与接口参考](inferport-design.md)，日常命令见 [README](../README.md)。

## PyPI 发布与安装

[GitHub Release v0.1.0](https://github.com/jeremy775885/InferPort/releases/tag/v0.1.0)
于 2026-09-26 发布；[官方 PyPI](https://pypi.org/project/inferport/0.1.0/) 提供 wheel 与 sdist。
Trusted Publishing 的 OIDC 绑定已由实际上传验证。

在仓库外的干净 Linux 虚拟环境中，从官方 PyPI 正常安装 `inferport==0.1.0`：

- 解析到 InferPort 0.1.0、NumPy 2.5.3、msgpack 1.2.2、websockets 17.1。
- `pip check` 和 `python -I` 隔离导入通过，`Backend`、`Client`、`serve` 可导入。
- 模块来自该环境的 site-packages，导入版本和发行元数据均为 0.1.0，未使用本地源码或 editable 安装。

上传时间和实际检查命令见 [首次发布详细记录](archive/v0.1.0-validation.md#pypi-首次发布与安装验证)。
后续版本的发布流程见 [发布指南](releasing.md)。

## 性能证据

原始测量数据保留在 [benchmark-results.json](benchmark-results.json)，测量环境、数值表与内存统计限制见
[首版性能记录](archive/v0.1.0-validation.md#实测性能)。
这是 Linux 本机回环、独立 echo 服务、无 TLS/压缩、每种输入 3 次预热及 100 次往返的结果，
不能代替真实模型、局域网/WAN 或生产尾延迟测量。JSON/Base64 比较反映编码体积差异，
不能用于宣称 WebSocket 相对原生 TCP 的网络性能优势。

新测量应保存为单独结果文件并记录环境、版本和命令，保留首版基线。

## 剩余验证范围

- 真实模型与 processor 的数值一致性、跨机器推理、仿真或真机闭环。
- 实际网络下的长期运行、资源趋势与异常恢复。
- Windows/macOS 完整故障测试和实际部署；ARM64 设备安装与运行。
- 大规模异常输入模糊测试、独立安全审计，以及相同 codec/socket 设置下的原生 TCP 对照基准均未开展。

永久挂起的后端仍需要外部进程管理，线程模型不能安全强杀模型计算。
下一阶段优先按 [真实推理验证计划](integration-validation.md) 取得实际接入证据，
其他维护优先级见 [roadmap](roadmap.md)。

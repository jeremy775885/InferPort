# InferPort 验证概览

更新日期：2026-09-26。发布基线：`v0.1.0` / `3cc9077779fa072f0a77a32c86cc842b19d5b6b8`。
本文汇总已取得的证据及适用范围；逐轮环境、命令和实验细节保存在
[0.1.0 验证归档](archive/v0.1.0-validation.md)。后续源码改动应以对应提交的检查结果为准。

## 当前状态

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
- 不同 InferPort SDK 版本的双向互通；现有跨 Python/NumPy 测试均使用同一 SDK 版本。
- 大规模异常输入模糊测试、独立安全审计，以及相同 codec/socket 设置下的原生 TCP 对照基准均未开展。

永久挂起的后端仍需要外部进程管理，线程模型不能安全强杀模型计算。
下一阶段优先按 [真实推理验证计划](integration-validation.md) 取得实际接入证据，
其他维护优先级见 [roadmap](roadmap.md)。

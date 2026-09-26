# InferPort 首版实现与验证

日期：2026-09-26。本记录对应工作区中的 InferPort 0.1.0；未发布到 PyPI。

## 已实现

- `Backend`、`Client`、`serve` 和公共错误类型；删除原 `policy_runtime` 实现。
- WebSocket / MessagePack v1、严格信封和请求 ID 校验、数值 ndarray 扩展。
- 单活动连接、READY、显式 reset、断连清理、后端独立串行工作线程。
- 建连/就绪期限、发送与接收共同期限、超时失效、无自动重放。
- 用公开 transport 回调实现有界关闭；包含通信库内部触发 close 的路径。
- 可选 bearer token、TLS、Origin 拒绝、完整消息及结构限制。
- 无状态、有状态示例，测试、CI 配置、双进程互通脚本和可复现基准。

这些覆盖设计 A–D 的核心交付物；完整发布验收仍有下文列出的平台和运行验证缺口。
真实模型/环境适配属于 E，尚未执行。未修改 LeRobot、mimix 或机器人执行仓库。

## 独立仓库迁移验证

当前实现、测试、示例、CI、文档与锁文件已迁入独立的 `InferPort` 本地 Git 仓库，后续维护以本仓库为准。
原 `policy-runtime` 工作区保留为迁移备份。当前分支为 `main`，
`origin` 指向 `git@github.com:jeremy775885/InferPort.git`。

在新目录重新执行 `uv sync --locked --python 3.12 --group dev` 创建环境，确认源码导入来自本仓库。
Python 3.12.3 / NumPy 2.4.4 / msgpack 1.2.2 / websockets 17.1 完整测试 **144 passed**，
Ruff lint/格式检查与 sdist/wheel 构建通过。

新建隔离 wheel 环境，正常安装本仓库构建的 wheel，`uv pip check` 与隔离导入检查通过。
Python 3.12.3 / NumPy 2.5.3 / msgpack 1.2.2 / websockets 17.1 下，
`tests/cross_environment.py` 的两次独立进程网络往返通过；这次两端使用同一环境。
下文保留迁移前的跨版本、跨环境与性能证据，不把这些记录视为在新目录全部重跑。

## GitHub Actions

2026-09-26 核对 [首次 CI 运行](https://github.com/jeremy775885/InferPort/actions/runs/36215291050)：
提交 `fb7f0b01fd324f5d5f1989e35e9c6efb863c5ccf`，11 个任务全部成功。

- 8 个 Linux 测试任务：Python 3.10–3.14 的最新依赖组合，以及 Python 3.10 最低依赖、
  Python 3.11 / NumPy 1.23.5、Python 3.12 / NumPy 1.26.4 组合。
- 1 个跨环境任务：构建 wheel 后，在 Python 3.10 / NumPy 1.21.3 与 Python 3.12 / NumPy 2.x 之间双向调用。
- 2 个平台 smoke 任务：Windows 和 macOS，覆盖 codec、protocol、lifecycle、adapter contracts 与 packaging。

此记录补充下面本地验证阶段尚未取得的远程 CI 证据。Windows/macOS 结果只覆盖配置中的 smoke 子集，
不代表所有网络故障测试、真实模型或机器人部署均已验证。后续改动仍以各自提交的 CI 结果为准。
后续工作的优先级与完成标准见 [成熟度评估与计划](roadmap.md)。

## PyPI 发布准备

2026-09-26 新增 [发布工作流](../.github/workflows/publish.yml) 与 [发布说明](releasing.md)，
版本来源统一为 `src/inferport/__init__.py`；发行元数据增加项目链接并明确包含 MIT LICENSE。
现有 GitHub CI 增加 `workflow_call` 入口，正式发布时复用同一提交的完整矩阵。

本轮本地验证：

| 安装方式 | CPython | NumPy | msgpack | websockets | 结果 |
|---|---|---|---|---|---|
| editable | 3.12.3 | 2.4.4 | 1.2.2 | 17.1 | 144 passed |
| 构建的 wheel | 3.12.3 | 2.5.3 | 1.2.2 | 17.1 | 144 passed |
| 构建的 wheel | 3.10.17 | 1.21.3 | 1.1.0 | 16.1.1 | 144 passed |

`uv build --no-sources`、`twine check --strict`、Ruff lint/格式检查和 wheel 环境的 `uv pip check` 均通过。
打包测试同时验证导入版本与发行元数据一致、MIT 许可表达式与 LICENSE 文件声明一致。
两个工作流通过 actionlint 1.7.12；直接执行发布标签检查步骤，`v0.1.0` 被接受，`v0.2.0` 被拒绝。

用户已反馈完成 PyPI pending publisher 配置，绑定 `publish.yml` 和 `pypi` 环境。
本地验证不证明 GitHub OIDC 与 PyPI 绑定已经生效；本轮尚未执行正式上传或从 PyPI 安装。
手动运行发布工作流只验证和保存产物；实际上传由 GitHub Release 的 `published` 事件触发。

## 实际运行的兼容矩阵

平台为 Linux x86_64，内核 6.8.0-139-generic，glibc 2.39。
首版初次验证时，下列每个环境均通过 **141 项测试**；不是只测试 import 或依赖解析。
后续新增测试的复核范围单独记录如下，不把旧矩阵自动视为已重跑。

| CPython | NumPy | msgpack | websockets | 安装方式 |
|---|---|---|---|---|
| 3.10.17 | 1.26.4 | 1.1.0 | 16.1.1 | editable，当时的最低组合 |
| 3.10.17 | 1.26.4 | 1.2.2 | 16.1.1 | 构建的 wheel |
| 3.11.12 | 2.4.6 | 1.2.2 | 17.1 | editable |
| 3.12.3 | 1.26.4 | 1.2.2 | 17.1 | editable |
| 3.12.3 | 2.4.4 | 1.2.2 | 17.1 | editable，开发环境 |
| 3.12.3 | 2.5.3 | 1.2.2 | 17.1 | 构建的 wheel |
| 3.13.3 | 2.5.3 | 1.2.2 | 17.1 | editable |
| 3.14.7 | 2.5.3 | 1.2.2 | 17.1 | editable |

Python 3.14 使用正式解释器，没有把旧 uv 下载的 alpha 版本算作验证结果。
版本来自实际安装环境；这些组合不代表依赖范围内每个历史版本都测试过。

复现初版验证时的最低组合（放宽 NumPy 下限后的验证见下文）：

```bash
uv venv --python 3.10 .venv-310
uv pip install --python .venv-310/bin/python -e . pytest \
  numpy==1.26.4 msgpack==1.1.0 websockets==16.1.1
uv run --no-project --python .venv-310/bin/python -m pytest -q
```

测试包含全部允许 dtype、大小端、非连续视图、标量/空数组、畸形扩展、重复键、深度与大小限制，
以及状态重置、并发调用拒绝、busy、非法响应 ID、非法操作、认证和版本失败、TLS 信任与主机名检查。

故障测试使用可控 socket 或同步事件，实际覆盖握手/READY 超时、16 MiB 发送堵塞、
发送超时后的强制关闭、通信库内部 close 堵塞、服务端发送超时、初始化中断连、清理期间 busy、
推理超时后仍持有后端、跨线程 close、KeyboardInterrupt、stop_event 和独立进程 Ctrl+C。
500 次连续调用检查线程和文件描述符数量不随请求数增长；这不等于长时间内存浸泡测试。

### RLinf 对照后的补充验证

新增 [适配契约测试](../tests/test_adapter_contracts.py) 共 3 项：

- 后端重复使用输出数组及其非连续视图，后续推理、显式 reset 和断连清理不改变旧的客户端结果。
- [线程绑定示例](../examples/thread_bound_backend.py) 在首次 reset 时加载一次模型，
  初始化、推理、显式/断连 reset 和释放均在同一工作线程完成。
- 没有客户端接入就停止服务时，不为清理而加载模型。

本次完整测试结果：

| CPython | NumPy | msgpack | websockets | 结果 |
|---|---|---|---|---|
| 3.12.3 | 2.4.4 | 1.2.2 | 17.1 | 144 passed |
| 3.10.17 | 1.26.4 | 1.1.0 | 16.1.1 | 144 passed |

使用 `uv run pytest -q` 和当时最低依赖环境的同一测试集执行；Ruff lint 与格式检查通过。
跨平台 CI smoke 已纳入新增文件；后续远程运行结果见上方 GitHub Actions 记录。
本次不新增真实模型验证结论：原生模型与远程推理的数值对照留到首次实际适配时完成。

### 放宽 NumPy 下限后的验证

运行依赖已调整为 `numpy>=1.21.3,<3`，设计、README、开发锁文件和 CI 同步更新。
下列环境通过正常依赖解析安装本次构建的 wheel，没有使用 `--no-deps` 绕过下限：

| CPython | NumPy | msgpack | websockets | 结果 |
|---|---|---|---|---|
| 3.10.17 | 1.21.3 | 1.1.0 | 16.1.1 | 144 passed |
| 3.11.12 | 1.23.5 | 1.1.0 | 16.1.1 | 144 passed |
| 3.12.3 | 2.5.3 | 1.2.2 | 17.1 | 144 passed |

另在 Python 3.12.3 / NumPy 1.26.4 环境验证 wheel 安装与依赖一致性，本轮未重跑该组合的完整测试。
前三个旧版 NumPy（1.21.3、1.23.5、1.26.4）均先安装，再安装 InferPort；NumPy 版本保持不变。
四个 wheel 环境的 `uv pip check` 均通过，发行元数据包含 `numpy<3,>=1.21.3`。

复现当前最低组合：

```bash
uv build
uv venv --python 3.10 .venv-wheel-numpy1213
uv pip install --python .venv-wheel-numpy1213/bin/python \
  numpy==1.21.3 msgpack==1.1.0 websockets==16.1.1 'pytest>=8,<9'
uv pip install --python .venv-wheel-numpy1213/bin/python dist/inferport-0.1.0-py3-none-any.whl
uv pip check --python .venv-wheel-numpy1213/bin/python
uv run --no-project --python .venv-wheel-numpy1213/bin/python -m pytest -q
```

Python 3.10.17 / NumPy 1.21.3 与 Python 3.12.3 / NumPy 2.5.3 的 wheel 环境，
使用 `tests/cross_environment.py` 交换服务端和客户端角色，两个方向的真实网络调用均通过。
载荷包含图像、大小端与非连续 state、batch、嵌套数据及 NumPy 标量。

`uv lock --check`、Ruff lint/格式检查、`git diff --check` 和 sdist/wheel 构建通过。
CI 最低组合和跨环境互通使用 NumPy 1.21.3，增加 Python 3.11 / NumPy 1.23.5 组合，
保留 Python 3.12 / NumPy 1.26.4；后续远程运行结果见上方 GitHub Actions 记录。

## 双进程、安装与打包

首版初次验证时，两个独立 wheel 环境进行了真实网络调用，图像、大小端 state、batch 和嵌套数据往返通过：

1. Python 3.10.17 / NumPy 1.26.4 / websockets 16.1.1 服务端 → Python 3.12.3 / NumPy 2.5.3 / websockets 17.1 客户端。
2. 交换两端角色，再次通过。

复现命令：

```bash
uv build
uv venv --python 3.10 .venv-wheel-310
uv venv --python 3.12 .venv-wheel-312
uv pip install --python .venv-wheel-310/bin/python numpy==1.26.4
uv pip install --python .venv-wheel-310/bin/python dist/inferport-0.1.0-py3-none-any.whl
uv pip install --python .venv-wheel-312/bin/python dist/inferport-0.1.0-py3-none-any.whl
uv run --no-project --python .venv-wheel-312/bin/python tests/cross_environment.py \
  --python-a .venv-wheel-310/bin/python --python-b .venv-wheel-312/bin/python
```

先安装 NumPy 1.26.4 再安装 wheel，NumPy 保持 1.26.4，没有被强制升级到 2.x。
wheel 环境中分别运行 `examples/serve_value.py` 和 `examples/call_value.py`，得到 `[7. 7. 7. 7.]`。

`uv build` 同时生成 sdist 与 `inferport-0.1.0-py3-none-any.whl`。
wheel 约 17 KB，只有 `inferport` 模块、类型标记、MIT 许可证和发行元数据；没有旧包、模型依赖或本地环境。
`Requires-Python` 为 `>=3.10`，运行依赖正好三个，约束与设计一致。
隔离解释器的 import 测试确认没有加载 Torch、JAX、LeRobot、Gymnasium、ROS 或旧包。

Linux ARM64 使用 `uv pip compile --python-platform aarch64-unknown-linux-gnu --only-binary :all:`
对 Python 3.10 和 3.12 完成二进制依赖解析；这不是 ARM64 设备安装/运行测试。
`ruff check` 与 `ruff format --check` 已通过；后续远程运行结果见上方 GitHub Actions 记录。

## 实测性能

原始数据见 [benchmark-results.json](benchmark-results.json)。复现：

```bash
uv run benchmarks/roundtrip.py --iterations 100 --output docs/benchmark-results.json
```

Python 3.12.3 / NumPy 2.4.4 / msgpack 1.2.2 / websockets 17.1；Linux 本机回环，独立服务端进程，
echo 后端，无 TLS、无压缩。每种输入 3 次预热、100 次往返；往返包含本地编码/解码。
100 个样本的高百分位仅用于本次诊断，不足以预测生产尾延迟。

| 输入 | MessagePack 消息字节 | JSON/Base64 字节 | 往返 p50 / p95 / p99（ms） |
|---|---:|---:|---:|
| 32 维 float32 state | 166 | 289 | 0.460 / 1.018 / 1.101 |
| 224×224 RGB | 150,576 | 200,828 | 0.325 / 0.595 / 0.638 |
| 3×480×640 RGB | 2,764,852 | 3,686,528 | 7.107 / 7.920 / 9.633 |
| batch=8，每项 3 张 224 RGB + state | 3,613,768 | 4,818,481 | 9.478 / 10.044 / 13.313 |
| 接近 64 MiB 上限 | 67,107,888 | 89,477,241 | 414.161 / 419.210 / 499.948 |

大数组消息体积约减少 25%；这是编码比较，没有把它解释成 WebSocket 相对原生 TCP 的加速。
小 state 的此次延迟高于单张图像，反映小样本、调度及测试时同机其他验证任务的影响，不能据此推导大小关系。
64 MiB 消息有明显的序列化、掩码、缓冲和可写数组拷贝成本，仍需按真实部署的输入大小测量。

JSON 记录同时包含编码/解码中位耗时、吞吐、两端 CPU 时间和峰值 RSS。
该进程的最高 RSS 约为客户端 521 MiB、服务端 437 MiB；客户端数值还包含单独 JSON 编码比较的分配，
并且 RSS 是各输入场景累计的进程高水位，不能当作某次 WebSocket 交换的独立内存占用。

## 尚未声称完成的验证

- Windows/macOS 已通过 CI smoke，尚未验证完整故障测试与真实部署；ARM64 只有二进制解析。
- 未运行真实 PI05/mimix、processor、仿真或真机闭环；未验证实际局域网/WAN 时延与长期重连行为。
- 未做长期内存浸泡、大规模异常输入模糊测试，未进行独立安全审计。
- 未做同 codec、同 socket 设置下的原生 TCP 公平基准。
- 永久挂起的后端无法由线程安全强杀；服务仍需外部进程管理。

以上缺口不影响开始编写适配器和进行本地集成；发布或部署到具体机器人时，应按所用平台和业务补齐验证。

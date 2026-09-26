# 发布 InferPort

发行包名为 `inferport`，GitHub 仓库为 `jeremy775885/InferPort`。
版本号的唯一来源是 `src/inferport/__init__.py` 中的 `__version__`，Hatchling 从中生成发行元数据。
GitHub Release 标签必须为 `v<版本号>`，首版为 `v0.1.0`。

## 一次性配置

在 PyPI 账户的 Publishing 页面添加 GitHub pending publisher：

| 字段 | 值 |
|---|---|
| PyPI Project Name | `inferport` |
| Repository owner | `jeremy775885` |
| Repository name | `InferPort` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

用户已反馈完成此项配置；只有首次成功上传才能验证 PyPI 端绑定生效。
工作流位于 [.github/workflows/publish.yml](../.github/workflows/publish.yml)，上传任务使用 `pypi` 环境。
可在 GitHub 仓库 Settings → Environments 中创建该环境；若尚未创建，GitHub 在运行引用它的任务时自动创建。
发布采用 OIDC，无需在仓库中配置 PyPI 密码或长期 API Token。

Pending publisher 本身不保留项目名；首个成功上传才创建项目。
参见 [PyPI 首次发布配置](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)、
[官方上传 Action](https://docs.pypi.org/trusted-publishers/using-a-publisher/) 和
[GitHub 环境说明](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments)。

## 发布前验证

1. 更新 `__version__`、[变更记录](../CHANGELOG.md) 和需要随发行修正的文档，执行 `uv lock`。
2. 运行以下本地检查，核对包名、版本、许可证及实际包含的文件。
3. 将准备发布的改动提交并推送到 GitHub；标签应指向包含本发布工作流的提交。

```bash
uv sync --locked --python 3.12 --group dev
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv build --no-sources --out-dir dist/release-check
uvx twine check --strict dist/release-check/*
```

在 GitHub Actions → Publish to PyPI → Run workflow 手动运行，可验证完整 CI、构建和 wheel 安装。
手动运行只上传 GitHub 构建产物，不向 PyPI 上传，也不能提前证明 OIDC 授权已生效。

## 正式发布

在 GitHub Releases 页面创建 Release，选择准备好的提交并填写标签，例如 `v0.1.0`，
使用对应版本的变更记录填写说明。点击 Publish release 才会触发正式上传；保存草稿和仅推送标签都不会上传。

工作流依次执行：

1. 对 Release 对应的提交运行 [.github/workflows/ci.yml](../.github/workflows/ci.yml) 的完整矩阵。
2. 核对 Release 标签和包版本，构建 sdist/wheel，执行严格元数据检查。
3. 在新环境中安装实际 wheel，运行完整测试后保存构建产物。
4. 独立上传任务通过 `pypi` 环境取得短期凭据，上传上述产物。

普通提交、PR 和手动验证不会触发 PyPI 发布。只有上传任务具备 `id-token: write` 权限。
当前流程只配置正式 PyPI；若以后使用 TestPyPI，需要另外注册相应 publisher。

## 发布后检查

在 PyPI 项目页面确认版本和两个发行文件，再从仓库外的新环境安装已上传版本：

```bash
python -m venv /tmp/inferport-pypi-check
/tmp/inferport-pypi-check/bin/python -m pip install --index-url https://pypi.org/simple 'inferport==0.1.0'
/tmp/inferport-pypi-check/bin/python -I -c 'import inferport; print(inferport.__version__)'
```

上述命令是 Linux/macOS 示例；Windows 使用虚拟环境的 `Scripts/python.exe`。
确认安装来源是 PyPI 后更新验证记录及首次发布状态。API、模型和平台验证范围仍以验证记录为准。

若上传失败，先检查标签、publisher 的 owner/repository/workflow/environment 是否与此文档一致，
以及 GitHub Actions 上传任务日志。不要用新的文件覆盖已发布的相同文件名；内容变化时递增版本。

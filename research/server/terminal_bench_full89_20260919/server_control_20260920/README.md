# Terminal-Bench 三组任务的服务器版本

**最新方案见 [NO_DOCKER.md](NO_DOCKER.md)：按用户要求，三组准备版本已切到无 Docker 的受管理 Apptainer 后端。两类真实镜像及并发隔离、磁盘写入、内存硬限制和清理检查已通过。正式批量提交仍需逐题环境兼容检查与已有作业交接。**

下文保留上一阶段改造记录；其中 Docker 默认后端和旧 CoMem 已退出的状态已被上述新方案与最新运行状态更新。[SERVER_CONTAINER_SETUP.md](SERVER_CONTAINER_SETUP.md) 是上一轮 Docker 权限诊断记录，不再是默认执行路线。

已完成代码改造并同步到服务器；尚未切换正在运行的旧 Dense，也未提交新的 GPU 作业。

服务器代码根目录：

`/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920`

服务器运行环境、任务文件、请求和结果根目录：

`/srv/encbank/qcomem_runtime_20260911/server_control_20260920`

## 改动

- 三个独立运行目录：`dense_server_r6_20260920`、`comem_k12_server_r6_20260920`、`comem_k48_server_r6_20260920`。
- Slurm 作业同时监督 GPU holder 和 Linux Harbor 控制器。提交后不需要 Windows、WSL、客户端 SSH 常驻连接或本地 Docker。
- 控制器通过集群内认证 HTTP 连接本组模型服务。请求、回复、校验凭据、进程退出记录和逐题结果均保存在服务器。
- 用 Linux 文件锁与 `/proc` 进程身份替换 Windows 锁与进程 API。按节点预约任务 RAM，读取 host/cgroup 可用内存；旧 owner 的资源预约不会被自动清除。
- CoMem 加载器使用新模型路径；checkpoint 校验继续使用完整的原训练身份。没有删除 j、L、revision、模型字典或 adapter 校验。
- 保留原模型、adapter、检索、采样、上下文、无任务总时限协议和一次执行策略。默认容器后端仍为 Docker。
- 服务器使用独立 Harbor 0.23.0 环境，按旧环境依赖版本约束安装。模型和 checkpoint 均未下载到本地。

## 验证

- 89 个任务、945 个任务文件在服务器准备完成并校验；任务 verifier 只做不透明字节复制和哈希核对。
- 三组 CPU 预检通过，分别覆盖原接续清单的 49 / 79 / 79 个任务。
- 两组 CoMem 的真实 final4000 checkpoint 在服务器用 CPU 检查训练身份与运行路径，没有创建 CUDA context。
- 9 项服务器 CPU 测试通过，包括两个模拟任务的完整控制器生命周期、并发请求隔离、回复哈希验证、防重复请求、资源预约和 Docker 服务端检测。
- 这些测试没有运行实际 benchmark 或 GPU 推理，不能当作质量/性能结果。

本地证据：`reports/VALIDATION.json`、`reports/cpu_tests.txt`。服务器证据：根目录 `VALIDATION.json` 与各运行目录的 `server_cpu_preflight.json`。

## 正式切换前尚未满足的条件

1. **Docker 访问权限。** 当前 `liuhanzuo` 访问 `/var/run/docker.sock` 返回 `Permission denied`。需要服务器管理员提供该账号获授权的 Docker 服务，且计划调度到的计算节点能使用它。预检会拒绝只安装客户端、没有可连接服务端的环境。没有修改 socket 权限、用户组或其他人的容器。
2. **旧 Dense 的正常交接。** 最后核查时 `111876` 仍在运行，继续依赖旧电脑的 owner 和任务容器。新的 Dense 不能直接重跑原 49 题清单；需等旧控制器及活跃任务正常闭合，核对新增结果，生成剩余任务清单与 `predecessor_reconciliation.json`。新提交入口会阻止重复启动。当前不要因本次代码改造就关闭旧电脑。

CoMem 原 R5 的 `111884` / `111886` 已真实退出且未 ready；新版本修复了其路径问题。Docker 条件满足后，它们可按原 79 题接续范围启动。

Apptainer 的 Ubuntu 基础可写容器检查通过，但它与 Docker 的网络隔离和资源限制不同；尚未做全部任务兼容性验证，因此默认没有换用这个后端。Alpine fakeroot 检查失败也保留在服务器诊断记录中。

## 服务器入口

以下命令在 `gpu-node1` 上执行，客户端只需发送命令，不承担运行控制：

```bash
cd /srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/runs/comem_k12_server_r6_20260920
/srv/encbank/qcomem_runtime_20260911/server_control_20260920/harbor_env/bin/python deploy.py check
/srv/encbank/qcomem_runtime_20260911/server_control_20260920/harbor_env/bin/python deploy.py submit
/srv/encbank/qcomem_runtime_20260911/server_control_20260920/harbor_env/bin/python observe.py
```

top48 使用对应目录。Dense 必须先完成上一节的旧结果交接。

每次提交都重新检查冻结源码、模型位置、任务文件、容器服务、旧作业退出、重复提交和本线程最多四个 GPU 请求；Slurm 还会在实际分配节点执行启动前检查。

`sync_code.py` 只是可选的源码上传工具，不参与实验运行。它拒绝覆盖已有提交或 owner 的新运行目录。原 R5 和本地旧论文目录保留原样。

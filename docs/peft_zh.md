# CoMem PEFT 加载入口

以PEFT0.20.0、Torch2.14.0、Transformers5.16.1为本次验证版本。代码与attempts/peft_r1/comem_peft.py相同；最终结果见上级PEFT_INTEGRATION_20260922.md。

服务器上的标准adapter目录：

`/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/comem_vllm_pilot_20260922/attempts/peft_r1/peft_adapter`

只在新的服务器模型进程中使用，原始模型、adapter文件保持原样：

```python
from comem_peft import load_comem_peft
from hybrid_reader import HybridReader

# model由既有load_model(runtime_cfg)加载；转换文件与该27B revision绑定。
model, peft_owner, targets = load_comem_peft(
    model, adapter_directory, mode="compat"
)
reader = HybridReader(model, j=21)
# 此处不要再调用reader.attach()或load_state()，否则会重复挂载LoRA。
```

- `compat`默认：PEFT负责标准格式加载；LegacyArithmetic使用PEFT拥有的FP32 A/B，执行原始matmul及舍入顺序。固定adapter推理，不是训练或动态多adapter管理入口。不得对这个兼容包装后的模型调用PEFT save_pretrained/merge/switch-adapter；使用原始导出文件，或在新模型上重新选择加载模式。
- `native`：直接调用PEFT前向。PEFT先以FP32加回基础输出再转BF16，与原实现先将LoRA分支转BF16再相加不同，需独立标记数值变化。
- `merged_bf16`：调用PEFT merge_and_unload(safe_merge=True)，将新加载模型中的上层LoRA合并，移除分支。safe_merge检查非有限权重，并不保证与未合并推理逐位一致；本次严格等价门槛未通过，保留为实验选项。

不要在正在服务的模型对象上调用这些初始化函数。本文件不连接请求队列、不提交Slurm、不恢复或重放任何题目；正式控制器的切换需要新的、独立且有唯一题目归属的运行版本。

标准导出由export_adapter产生：333个上层Linear，666个FP32张量，rank32/alpha32、dropout0、bias none。目标模块使用完整路径，前21层不包含LoRA。conversion_receipt.json记录目标名单和两个标准文件的SHA；加载时先验证SHA。模型/adapter及合并权重不下载到本机。

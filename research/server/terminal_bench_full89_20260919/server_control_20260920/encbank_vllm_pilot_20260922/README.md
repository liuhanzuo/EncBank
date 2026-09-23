# Encbank / vLLM 内核适配调查，2026-09-22

PEFT接入已完成（120274）：默认兼容路径在三组整模型检查中logits逐位一致；合并路径k12快35%–39%、k48批8快11%，但严格数值等价未通过，保留为实验选项。见[接入与验证报告](PEFT_INTEGRATION_20260922.md)和[使用入口](prototype/PEFT_USAGE.md)。

新增同环境Dense/Encbank测速已完成（120163）：Transformers批8每路Dense约19.3，Encbank k12约13.6、k48约8.6 token/s；Dense加同一FP32 LoRA约13.8。详见[同环境测速报告](TRANSFORMERS_COMPARISON_20260922.md)。试验卡已释放，正式补测未改动。

结论：已完成独立GPU和27B整模型验证。GDN新原型单层约4.5倍，结合KV追加缓存后8路k12/k48实测约1.082/1.110倍；KV单独优化的logits逐位一致，GDN版本严格数值门槛未过。尚未形成完整vLLM后端，也未替换正式补测。详见[本轮结果](VALIDATION_RESULTS.md)。

服务器目录：`/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/encbank_vllm_pilot_20260922`。模型及 adapter 始终留在服务器。

## 观察到的速度

只统计已完整返回、status=ok、生成至少128 token的请求；数据见 performance_audit.json。

|方法|样本数|解码 token/s 中位数|生成8192–32767 token的请求中位数|
|---|---:|---:|---:|
|Dense vLLM|798|56.43|52.87|
|Encbank k12|1878|4.41|3.99|
|Encbank k48|1168|3.53|3.23|

这是运行日志的观察比较，并非控制了输入、输出长度、并发队列及软件版本的严格benchmark。Encbank decode计时含其他请求加入时的预处理暂停；未完成长请求未纳入，存在选择偏差。不能声称内核替换必然带来13–16倍加速。

同一初始提示SHA的adaptive-rejection-sampler：Dense首轮生成357 token，而k12/k48分别生成47642/30336 token；circuit-fibsqrt为160/23118/14250。输出长度也显著拉长部分题目耗时。不同方法/采样实现的输出不同，不能把这些差异都归因于引擎。

Dense本轮只有18道新题，k12/k48分别75/69道；历史保留71/14/20。因此每组累计完成89题的先后本身也不是速度基准。

## 已核实的实现差异

1. 当前模型64层，其中48层Gated DeltaNet、16层普通全注意力。Encbank环境是torch2.14.0、transformers5.16.1；在与作业一致的PYTHONPATH下，fla、causal_conv1d、kernels均未找到。检查实际函数闭包后，GDN recurrent/chunk及causal-conv均绑定PyTorch回退函数，详见runtime_kernel_audit.json。短上下文GPU profiler已完成，耗时分布及其适用边界见VALIDATION_RESULTS.md。
2. 当前HF DynamicLayer.update每token通过torch.cat重建keys和values。Encbank还将不同长度请求的KV补齐到队列最长行，在refill/compact时复制和重排。生成越长，历史缓存搬运越多；回包时共享缓存的观察中位数约17.0GiB(k12)/23.5GiB(k48)，该数字也含GDN状态，不能直接当成每步KV复制字节数。
3. 每行每token有isfinite的布尔取值和采样.item()，造成主机/设备同步；新请求Write/prefill串行插入同一循环，暂停当前decode。使用普通PyTorch调用，未做vLLM式完整decode CUDA graph。
4. 上层全部Linear包装了FP32低秩分支，每次执行额外两次矩阵乘法及类型转换。融合/合并有优化空间，但合并为BF16权重会改变舍入，不能未经数值验证直接作为等价替换。
5. Dense启动日志明确选择FlashInfer普通注意力、FlashInfer采样、Triton/FLA GDN prefill和torch.compile。其运行环境是vLLM0.22.1、torch2.11.0，不能将对应二进制扩展直接装入正在运行的torch2.14环境并假定ABI兼容。
6. Encbank重用的是历史片段的前21层残差状态。每个新生成token仍执行前21层及后43层，总共64层；历史压缩不意味着长生成的每token计算自动大幅减少。

## 原型与验收边界

- vllm_gdn_adapter.py 对接服务器已安装vLLM0.22.1的fused_recurrent_gated_delta_rule。HF递归状态布局[B,H,K,V]与vLLM的[B,H,V,K]不同；适配层显式转置并保留输入状态。实际模型K=V=128，形状相等会掩盖方向错误，因此测试另外覆盖K不等于V。
- 保留HF在BF16精度下先做Q/K归一化的舍入顺序；在内核内部用FP32归一化是后续单独验证项。
- validate_gdn_adapter.py 的CPU合同测试使用独立矩阵公式验证布局、多步更新、零初态和输入不变；不是GPU内核或整模型质量验收。支持1/3/8批及实际48头128维形状，结果见cpu_validation.json。
- 实际vLLM算子的CPU Triton解释器试探遇到解释器/NumPy标量转换兼容错误，未成功执行；不据此宣称GPU内核不可用，也不将其算作验证通过。
- gpu_probe.py已在119748执行并通过：独立Slurm单GPU上，以原安装HF回退函数源码为参考，验证FP32/BF16、非方形/实际维度、16步状态、输入不变，然后测单层算子延迟（包含状态转置）。不加载模型，不发送benchmark请求，结果不能计入正式成绩。
- 用户已明确批准1张临时额外GPU用于此独立验证；benchmark仍最多4张，本试验最多1张运行或排队。模型前检查实际分配卡的空闲状态，避免调度记录与实际显存占用不一致。

## 完整接入路线

1. 先完成独立GDN与causal-conv GPU等价验证/微基准，确定在实测中值得接入的算子。GDN原型通过后还需要prefill及conv路径，单个decode函数不能代表完整优化。
2. 将普通注意力缓存改为分页池，替代每步cat和左侧补齐；上下层分别维护长度、RoPE位置和block table。GDN每层递归状态也需要独立行映射和释放。
3. 实现Encbank专用模型runner：Write时独立分块、前21层输出未做末端norm的残差H；Read时下层只看到query/generated，上层看到sink+selected H+query/generated。不能将H塞到普通prompt_embeds的第0层，也不能共用一个位置/缓存映射假装完成适配。H的身份要包含模型/adapter/分层点/块内容与顺序，不能复用仅按普通token前缀计算的缓存键。
4. 最后接入批量采样、prefill分块调度、固定地址缓存与CUDA graphs。精确保留top-k/top-p/temperature/stop和上下文规则；同seed跨采样器不保证逐token相同，需记录采样后端与随机数消耗差异。
5. 新后端验收先做固定token序列的teacher-forced logits/state比对、H逐字节不变、变长batch/refill/退出隔离，再做1/4/8并发及短/长上下文性能测试，最后开展独立标识的质量对照。未过门槛前不替换补测结果或热改活跃源码。

参考官方实现：[Qwen3.5模型](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5.py)、[自定义模型接入](https://docs.vllm.ai/en/latest/contributing/model/basic/)、[Hybrid KV管理](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)、[CUDA graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/)。以服务器安装的0.22.1源码为原型接口依据，在线main文档用于结构参考。

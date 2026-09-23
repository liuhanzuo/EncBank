# Qwen3-8B hidden Reader 快速试验

用户授权多组服务器 GPU 实验，本研究不受另一个 benchmark 对话的 4 GPU 上限限制。现有 benchmark 作业不修改。

四组：exact 保存 H12…H35；h12 从 H12 为各层拟合独立仿射投影；dual 用 H12 支持层 13…24，H24 支持层 25…36；h24_late 保留精确 KV13…24，以 H24 支持剩余层。紧邻 hidden 的层使用原投影作为精确锚点。H24 从固定组合的 teacher 前文计算，成本不免费，不能当作任意检索组合均可复用。

正文来自服务器已有 qasper_pilot：按 document_id 去重，并从训练候选中排除整个 dev 文档集合。32 个文档拟合、4 个验证、8 个 held-out 测试；每个文档 4×512 memory tokens，正文后续 128 tokens 为 query。仅保留正文 token，不使用 question、answer_ids 或 references。源文件的 train/dev 是历史内部划分，不是正式 QASPER 任务成绩。

快速阶段使用 16,384 个抽样历史位置作仿射 ridge 拟合，在 3 个固定正则强度中依据验证集输出 KL 选取，然后测试未参与选择的文档。这不是完整 SwiftKV 蒸馏；没有更新 backbone、原 LoRA 或 query 投影。报告 logits KL、top-1 一致率、正文 continuation NLL，并保存少量自由生成示例；不声称 agent benchmark 质量。

decode 实现两种策略：预测 memory KV 常驻；或仅保留 H，在每层 attention 更新时临时投影、与近期 KV 拼接后联合 attention。后一策略不将历史 KV 长期存入 cache，但仍逐 token 重做历史投影，可能慢。每个 arm 对比同一 checkpoint 两种策略的输出一致性。

系统计时使用前轮 PG19 固定输入，1/4/12 个 chunk，64-token query 后执行 32 次固定 token decode，每格 2 次预热 + 3 次计时；预测 KV 初始化单独测 2+5 次。模型、heads、数据和结果均在服务器。显存报告历史持久张量及 CUDA allocated/reserved 峰值，后者还包含模型、头与 workspace，不混成历史缓存大小。运行控制器驻留 Slurm，保存真实子进程退出。

不修改已经使用的源码或结果。GPU 运行前的语法预检曾发现一处括号错误，修复后才首次提交 GPU 作业；没有据此重跑任何模型结果。

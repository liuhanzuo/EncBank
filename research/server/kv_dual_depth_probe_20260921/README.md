# H12/H24 双深度 KV 重建计时

服务器专用实验；使用与前一轮相同的 Qwen3-8B、36 层、j=12、512-token chunks、BF16 和原始 LoRA。

比较 H12 串行重建 13–36 层、双 hidden 串行对照、双 CUDA stream，以及双 host thread + 双 CUDA stream。另测两个 12 层段各自耗时作为理想并行参考。三份保存的 PG19 输入，1/2/4/8/12 个 chunk，每格 2 次 warmup + 7 次计时，随机交错方法顺序，记录同步 wall/CUDA 时钟。

独立 chunk batch 为 hot-buffer 设计的计算探针；joint-pack 对照保留跨 chunk 因果注意力。每个配置的 H24 均事先从相同 H12、位置、attention graph 及 batch shape 生成；joint H24 不代表改变检索组合后仍可复用。H24 的准备耗时及额外字节单列。完整 24 层 KV 逐元素对照，数值一致性不代表 agent 质量验证。

用户当前设计约束：命中指在 hidden memory 上检索到的 chunk 是否已有可用 hot KV；新生成的当前 chunk 完成后也要进入 hot buffer，不能仅向 hot buffer 插入检索的历史块。本实验只测双深度重建，不声称测过真实检索命中率或实现了上述插入策略。

所有 GPU 推理与权重留在服务器；本地仅保存代码和结果。独立 Slurm 作业，不改变其他 benchmark 控制器或已完成实验。

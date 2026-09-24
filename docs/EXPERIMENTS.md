# Experiment source index

All links point to code, not a declaration that an experiment completed or passed.
Attempt suffixes and dates distinguish historical snapshots. See the
[release inventory](source_inventory.json) for per-file provenance and duplicate
workspace mappings.

| Family | Entry point |
| --- | --- |
| Base Encbank, selection, and resumed decoding | [`encbank/model.py`](../encbank/model.py), [`selectors.py`](../encbank/selectors.py) |
| Base LoRA distillation | [`train/distill.py`](../train/distill.py) |
| 9B/27B backbone training and evaluation | [`encbank_new_backbones_formal_20260915`](../research/dependencies/encbank_new_backbones_formal_20260915/) |
| Original hybrid reader and model preparation | [`encbank_new_backbones_20260915`](../research/dependencies/encbank_new_backbones_20260915/) |
| KV reconstruction and two-depth probes | [`kv_rebuild_probe_20260921`](../research/server/kv_rebuild_probe_20260921/), [`kv_dual_depth_probe_20260921`](../research/server/kv_dual_depth_probe_20260921/) |
| Hidden reader pilot and distillation | [`hidden_reader_pilot_20260921`](../research/server/hidden_reader_pilot_20260921/), [`hidden_reader_distill_20260921`](../research/server/hidden_reader_distill_20260921/) |
| Four-model and longer distillation | [`hidden_reader_four_distill_20260921`](../research/server/hidden_reader_four_distill_20260921/), [`hidden_reader_longkd_20260921`](../research/server/hidden_reader_longkd_20260921/) |
| Joint-band, local-anchor, shallow reader studies | [`hidden_reader_jointband_20260921`](../research/server/hidden_reader_jointband_20260921/), [`hidden_reader_localanchors_20260921`](../research/server/hidden_reader_localanchors_20260921/), [`hidden_reader_shallow_20260921`](../research/server/hidden_reader_shallow_20260921/) |
| Reader RULER evaluation | [`hidden_reader_ruler_20260921`](../research/server/hidden_reader_ruler_20260921/) |
| Reader Terminal-Bench diagnosis | [`hidden_reader_terminal_analysis_20260921`](../research/server/hidden_reader_terminal_analysis_20260921/) |
| Early hidden/Hot Buffer experiments | [`hidden_hot_buffer_20260921`](../research/server/hidden_hot_buffer_20260921/), [`hot_buffer_20260921`](../research/server/hot_buffer_20260921/) |
| 27B Hot Buffer staging and qualification | [`tf27b_hot_terminal_20260921`](../research/server/tf27b_hot_terminal_20260921/), [`tf27b_hot_qualification_r4_20260921`](../research/server/tf27b_hot_qualification_r4_20260921/) |
| Dense / Cold / Hot runtime snapshots | [`tf27b_hot_live_20260921`](../research/server/tf27b_hot_live_20260921/) |
| Hot24 + PEFT merged independent arm | [`runtime/hot_buffer`](../runtime/hot_buffer/), [`original snapshot`](../research/server/tf27b_hot_live_20260921/hot24_peft_merged_r3/) |
| Full89 Dense / Encbank k12 / k48 | [`server_control_20260920`](../research/server/terminal_bench_full89_20260919/server_control_20260920/) |
| Full89 six-GPU admission and task transfer | [`prepare_scale6_20260924.py`](../research/server/terminal_bench_full89_20260919/server_control_20260920/unbounded_20260921/prepare_scale6_20260924.py), [`protocol`](../research/server/terminal_bench_full89_20260919/server_control_20260920/unbounded_20260921/SCALE6_20260924.md) |
| Full89 accuracy-only storage-failure continuations | [`main selection and submission`](../research/server/terminal_bench_full89_20260919/server_control_20260920/unbounded_20260921/accuracy_recovery_20260924.py), [`Hot24+PEFT continuation`](../research/workspace/experiments/tf27b_hot_peft_terminal_20260922/hot_accuracy_recovery_20260924.py), [`source inventory`](accuracy_recovery_source_inventory.json) |
| GPU kernel, same-Transformers TPS, PEFT and Hot-merged pilots | [`encbank_vllm_pilot_20260922`](../research/server/terminal_bench_full89_20260919/server_control_20260920/encbank_vllm_pilot_20260922/) |
| Earlier long-context comparisons and baselines | [`research/server/repo`](../research/server/repo/) |
| Additional local preparation and handoff code | [`research/workspace`](../research/workspace/) |

`encbank_vllm_pilot_20260922/prototype` contains the kernel adapter prototypes and
PEFT helper; its `attempts` directories preserve the tested implementations.
The kernel pilots did not establish a full vLLM Encbank runner. The Hot24 merged
Terminal-Bench arm is a separate numeric configuration from legacy FP32 LoRA.

Benchmark task datasets and scoring assets must come from their original
distributions. Source snapshots include project-owned agents, execution wrappers,
selection/configuration logic, resource guards, launchers, and observers.

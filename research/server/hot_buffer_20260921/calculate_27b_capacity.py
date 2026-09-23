"""Analytical capacity scenarios, NOT a GPU allocation or throughput benchmark."""
import csv, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
e = json.loads((ROOT/'memory_27b_evidence.json').read_text(encoding='utf-8-sig'))
c = e['config']['text_config']
GIB, MIB = 2**30, 2**20
j, chunk = 21, 512
full = c['layer_types'].count('full_attention')
upper_full = c['layer_types'][j:].count('full_attention')
linear = c['layer_types'].count('linear_attention')
upper_linear = c['layer_types'][j:].count('linear_attention')
layer_kv = 2*c['num_key_value_heads']*c['head_dim']*2
h_token = c['hidden_size']*2
state_layer = (c['linear_num_value_heads']*c['linear_key_head_dim']*c['linear_value_head_dim']*4
    +(2*c['linear_num_key_heads']*c['linear_key_head_dim']+c['linear_num_value_heads']*c['linear_value_head_dim'])*c['linear_conv_kernel_dim']*2)
kv_chunk = upper_full*layer_kv*chunk
checkpoint_chunk = kv_chunk+upper_linear*state_layer
state = linear*state_layer
cap, weights, workspace = 248, 52, 32
available = cap-weights-workspace
# Design assumption: all pinned/current tokens bounded to 2048, memory top12.
# Active selected KV is a COPY, in addition to the persistent hot pool.
active = (12*kv_chunk+2048*full*layer_kv+state)/GIB
rows=[]
for nk in [32,64,128,256]:
    n=nk*1024
    dense=(n*full*layer_kv+state)/GIB
    for mode,unit in [('kv_only_lower_bound',kv_chunk),('kv_and_prefix_checkpoint',checkpoint_chunk)]:
        for hot in [12,16,24,28,32,48,64]:
            per=n*h_token/GIB+active+hot*unit/GIB
            b=math.floor(available/per)
            rows.append(dict(history_k=nk,hot_chunks_per_session=hot,mode=mode,per_session_gib=per,
                max_concurrency_memory_only=b,estimated_total_gib=weights+workspace+b*per,
                dense_per_session_gib=dense,dense_max_same_budget=math.floor(available/dense),
                max_concurrency_ratio=b/math.floor(available/dense)))
with (ROOT/'capacity_27b.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
recommend=[]
for n,b,cs in [(64,40,[24]),(128,32,[24,28]),(256,24,[24,32])]:
    for hot in cs:
        row=next(x for x in rows if x['history_k']==n and x['hot_chunks_per_session']==hot and x['mode']=='kv_and_prefix_checkpoint')
        recommend.append(dict(history_k=n,concurrency=b,hot_chunks_per_session=hot,
            estimated_total_gib=weights+workspace+b*row['per_session_gib'],
            dense_capacity_same_budget=row['dense_max_same_budget'],
            concurrency_ratio=b/row['dense_max_same_budget']))
out=dict(kind='analytical_only_not_measured_27b_hot_buffer',full_layers=full,upper_full_layers=upper_full,
    linear_layers=linear,upper_linear_layers=upper_linear,dense_kv_bytes_per_token=full*layer_kv,
    h21_bytes_per_token=h_token,kv_only_chunk_mib=kv_chunk/MIB,
    upper_state_checkpoint_mib=upper_linear*state_layer/MIB,kv_and_checkpoint_chunk_mib=checkpoint_chunk/MIB,
    active_recurrent_state_mib=state/MIB,active_per_session_gib=active,
    old_encbank_measured_post_release_allocated_gib=e['old_k12']['worker_complete']['allocated_bytes']/GIB,
    allocator_budget_gib=cap,rounded_shared_weights_gib=weights,workspace_allowance_gib=workspace,
    recommendations= recommend,dense_existing_pool_tokens=2569933,
    dense_existing_pool_slots_128k=2569933//131072,dense_existing_pool_slots_256k=2569933//262144,
    dense_existing_controller_slots=8,
    current_no_compaction_max_with_hot24_at_128k=math.floor(available/(1.25+24*checkpoint_chunk/GIB+16+state/GIB)))
(ROOT/'capacity_27b.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))

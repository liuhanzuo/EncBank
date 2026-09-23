"""Assemble only the fixed 07:31 snapshot's already CPU-verified new cells."""
import json
from pathlib import Path
from datetime import datetime, timezone

R = Path('/data/liuhanzuo/comem_v2_20260908')
B = R / 'workspace/exp/comem_v2_benchmarks_20260908'
def read(p):
    return json.loads(p.read_text())

r = read(B / 'heartbeat_extended_20260909_0731_ruler.json')
i = read(B / 'heartbeat_infinitebench_20260909_0731_new.json')
s = read(B / 'heartbeat_extended_20260909_0731_snapshot.json')
assert r['snapshot_timestamp_utc'] == i['snapshot_timestamp_utc'] == s['timestamp_utc']
assert r['checks']['new_ruler_raw_rescores'] == 150
assert i['checks']['official_rescores'] == i['checks']['new_source_token_pack_exact_input_own_cache_checks'] == 458
assert r['checks']['old_outputs_rescored'] == r['checks']['longeval_rescores'] == 0
assert i['checks']['reference_scorer_calls'] == i['checks']['reference_cache_lookups'] == 0
assert all(c['complete'] for c in r['writeback'] + i['writeback'])
assert len(r['writeback']) == 3 and len(i['writeback']) == 2
cells = r['writeback'] + i['writeback']
def mc_saved_rows(arm):
    result = {}
    for shard in range(4):
        run = R / 'outputs/benchmarks_v2/full/infinitebench' / arm / f'longbook_choice_eng_s{shard}of4'
        marker = read(run / 'COMPLETED.json')
        path = Path(marker['output_dir']) / f'longbook_choice_eng_{shard}.jsonl'
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert row['index'] not in result and row['score'] in (0, 1)
            result[row['index']] = row
    assert sorted(result) == list(range(229))
    return result

# New scores were independently checked in the preceding audit; references
# use previously verified saved scores. This tally does not call any scorer.
ref_rows = {arm: mc_saved_rows(arm) for arm in ('fix_all', 'j0')}
paired_counts = []
for cell in i['writeback']:
    cell['raw_mean'] = cell['correct'] / cell['n']
    new_rows = mc_saved_rows(cell['arm'])
    assert sum(row['score'] for row in new_rows.values()) == cell['correct']
    cell['paired_against_references'] = {}
    for refarm, refs in ref_rows.items():
        counts = {'new_arm_wins': 0, 'ties': 0, 'new_arm_losses': 0,
                  'both_correct': 0, 'new_only_correct': 0,
                  'reference_only_correct': 0, 'both_wrong': 0}
        for index in range(229):
            a, b = new_rows[index], refs[index]
            assert all(a[key] == b[key] for key in ('id', 'answers', 'input_tokens', 'pack'))
            x, y = a['score'], b['score']
            counts['new_arm_wins' if x > y else 'ties' if x == y else 'new_arm_losses'] += 1
            counts['both_correct' if x == y == 1 else 'new_only_correct' if x == 1 else
                   'reference_only_correct' if y == 1 else 'both_wrong'] += 1
        assert counts['new_arm_wins'] + counts['ties'] + counts['new_arm_losses'] == 229
        counts.update(paired_n=229, direction='new baseline relative to reference',
                      reference_scores_previously_audited=True, scorer_called=False,
                      significance_test_performed=False)
        cell['paired_against_references'][refarm] = counts
        paired_counts.append({'new_arm': cell['arm'], 'reference_arm': refarm, **counts})
lookup = {(c['benchmark'], c['task'], c['arm']): c for c in i['writeback']}
coverage, pending = [], []
for c in r['pending']:
    key = (c['benchmark'], c['task'], c['arm'])
    c = dict(c)
    if key in lookup:
        assert c['coverage_complete'] and c['n'] == lookup[key]['n'] == 229
        c.update(writeback_eligible=True, score_percent=lookup[key]['score_percent'],
                 independently_audited_complete_task=True)
    else:
        assert not c['coverage_complete']
        assert not any(x['coverage_complete'] for x in c.get('categories', []))
        assert c['score_percent'] is None
        pending.append(c)
    coverage.append(c)
ruler_waiting = []
for name, job in s['state']['jobs'].items():
    if not name.startswith(('ruler__pub__', 'ruler__pub_sink__', 'ruler__cbos__')):
        continue
    if not name.endswith(('_128k', '_256k')) or job.get('status') == 'completed':
        continue
    ruler_waiting.append({'job': name, 'status_at_snapshot': job.get('status'),
                          'complete': False, 'score_percent': None})

d = {
    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'snapshot_timestamp_utc': s['timestamp_utc'],
    'snapshot_file': str(B / 'heartbeat_extended_20260909_0731_snapshot.json'),
    'scope': 'New complete baseline cells only since 06:31; one fixed 07:31 heartbeat snapshot',
    'execution': {'location': 'remote CPU', 'CUDA_VISIBLE_DEVICES': '',
                  'torch_threads': 2, 'torch_interop_threads': 16, 'OMP_NUM_THREADS': 2,
                  'MKL_NUM_THREADS': 2, 'models_constructed': 0, 'cuda_initialized': False},
    'writeback': cells,
    'new_complete_cells': len(cells),
    'new_predictions_audited': sum(c['n'] for c in cells),
    'comparisons': i['comparisons'],
    'paired_counts': paired_counts,
    'coverage': coverage,
    'pending': pending,
    'pending_ruler_128k_256k': ruler_waiting,
    'checks': {'ruler': r['checks'], 'infinitebench': i['checks'],
               'all_new_full_tasks_audited': True, 'partial_means_reported': False,
               'longeval_rescored': False},
    'infinitebench_protocol': i['protocol'],
    'protocol_limits': [
        'RULER legacy source metadata/CLI/seed pairing only; no saved exact tokens, runtime pack or generation-cache keys.',
        'LongEval baseline grid was complete before this snapshot and was not rescored; missing historical runtime-pack evidence remains missing.',
        'MC uses complete 229-example task and official option accuracy including official answer-text extraction, bare yarn-mistral template and cap40.',
        'No partial MC or LoCoMo task/category mean; no significance or cross-task difficulty claim.',
        'Remote accuracy evidence does not establish local5090 timing or memory advantage.'
    ],
    'negative_results': [
        'At128k multi-key, isolated full KV scores32 versus BOS48; larger stored state does not ensure higher quality.',
        'V2/j0 old verified multi-key128k scores are both100; this audit checks new baseline raw scores and metadata only.'
    ],
    'evidence_files': ['heartbeat_extended_20260909_0731_ruler.json',
                       'heartbeat_infinitebench_20260909_0731_new.json'],
}
assert d['new_predictions_audited'] == 608
for name in ('heartbeat_extended_20260909_0731.json', 'heartbeat_extended_20260909_0731_writeback.json'):
    path = B / name
    assert not path.exists()
    value = d if not name.endswith('_writeback.json') else {
        'schema_version': 1, 'snapshot_timestamp_utc': s['timestamp_utc'],
        'new_complete_cells': cells, 'new_complete_cell_count': len(cells),
        'new_predictions_audited': 608, 'comparisons': i['comparisons'],
        'paired_counts': paired_counts,
        'pending': pending, 'protocol_limits': d['protocol_limits'],
        'evidence': str(B / 'heartbeat_extended_20260909_0731.json')}
    path.write_text(json.dumps(value, indent=2) + '\n')

md = ['# 07:31 扩展任务基线独立核验', '',
      f"固定快照为 {s['timestamp_utc']} UTC（北京时间07:33:41）。只纳入该时点主队列completed/exit_code0的完整输出，不追收后来完成的分片。", '',
      '全部新评分、源重建、分词、pack和缓存检查均在远端CPU执行：CUDA隐藏，Torch2线程/interop16，OMP2/MKL2。未加载模型或启动GPU，未改论文、根计划、队列或同步器。', '',
      '## 新完整格', '',
      '| Benchmark / task | Length | Arm | n | Exact score (%) |',
      '|---|---|---|---:|---:|']
for c in cells:
    md.append(f"| {c['benchmark']} / {c['task']} | {c.get('length', '—')} | {c['arm']} | {c['n']} | {c['score_percent']:.10f} |")
md += ['', '共5个新完整格、608个新预测。pub为原始CoMem，pub_sink为+BOS，cbos为isolated full KV控制，不能用它代表CacheBlend。', '',
       '## RULER：150条新raw核验，保留legacy边界', '',
       '多键128k的原版/BOS/isolated KV分别为8/48/32，每格50例。新raw按原answer-substring recall重评分，连续i0–49、完整summary、无OOM、实际参数与准入日志通过。与旧已核V2/j0逐例task/length/i/answers/n_tokens及CLI条件相同；两参考原分数均100，本轮不重新评分。', '',
       '历史RULER没有原始完整token、运行时selected pack或generation-cache键。因此配对仅限同remote runtime下metadata/CLI/seed，不能升级为exact-input/runtime-pack证明，也不能和旧Windows或其他tokenizer的draw混作配对。该格为BM25 top12/chunk512、seed42、cap48。', '',
       'isolated KV比BOS低16点，是必须保留的不利结果；更大存储不保证更高质量。128k五针三基线和256k单针/多键三基线在快照中均未完整，分数保持空白。', '',
       '## InfiniteBench MC：两条完整229例官方评测', '',
       '原版和+BOS各四个完整分片，索引恰好覆盖0–228，ID无重复。229个原始问题逐例重建官方vendored yarn-mistral裸模板、完整源与显式query边界，重分词并重建BM25 top12的有序pack。458个新预测均通过官方option-accuracy重评分及自身SQLite精确输入generation-cache匹配；每条缓存记录的预测和token总数均吻合。官方选择题评分包括其答案文本提取规则，不替换为简单首字母准确率。', '',
       '模型为stock Qwen3-8B BF16/SDPA、j12、相同repeat_kv策略、chunk512，greedy cap40，首步EOS抑制、之后自然EOS；无chat wrapper、无源截断。V2/j0只复用01:29已核分数并检查与新臂相同的源token和pack，未调用旧输出scorer或查询旧cache。', '']
for c in i['writeback']:
    md.append(f"- {c['arm']}：{c['correct']}/229 = {c['score_percent']:.10f}%，论文显示 {c['display']}。")
for c in i['comparisons']:
    md.append(f"- V2相对{c['new_arm']}：{c['v2_minus_new_arm_pp']:+.10f}点；旧V2={c['v2_previously_verified_score_percent']:.10f}，旧replay={c['full_recompute_previously_verified_score_percent']:.10f}。")
md += ['', '配对胜/平/负的方向是**新基线相对旧参考**，均n229；只按已验证的保存分数计数，不重新调用scorer：', '',
       '| New baseline | Reference | Wins | Ties | Losses |', '|---|---|---:|---:|---:|']
for c in paired_counts:
    md.append(f"| {c['new_arm']} | {c['reference_arm']} | {c['new_arm_wins']} | {c['ties']} | {c['new_arm_losses']} |")
md += ['', '只覆盖English long-book MC这一任务，不代表全InfiniteBench；均为描述性结果，不作显著性或跨任务难度结论。完整source token化警告不代表将整个长源一次送入模型；实际读入pack单独核验，其范围见机器JSON。', '',
       '## 未完整任务/类别仅列覆盖', '',
       '| Benchmark / arm | Complete-shard n | Formal n | Eligible score |',
       '|---|---:|---:|---|']
for c in pending:
    md.append(f"| {c['benchmark']} / {c['arm']} | {c['n']} | {c['expected_n']} | — |")
md += ['', 'LoCoMo仅核完整分片receipt、原配置期望索引、跨分片唯一ID及类别计数，未重评分局部输出。类别完整分母仍为282/321/96/841/446，总计1986。此快照三个基线均无完整类别，不能发布局部类别均值。机器JSON保留每类已完成n与null分数。', '',
       'LongEval三基线×六长度已在上轮全部完成，本轮没有重新读取其预测或缓存。此前“完整token/cache验证不补出缺失runtime pack”的限制保持。', '',
       '## 紧凑写回', '',
       '现有附录RULER基线块增加Multi-key/128k一行8/48/32；InfiniteBench现有MC行填入原版和+BOS两格，isolated仍空。保留已有V2/replay数值及RULER配对限定，无需新增正文表，也不要从远端准确率外推测速或峰值显存。', '',
       '报告和机器结果：`heartbeat_extended_20260909_0731.md/.json`；直接写回：`heartbeat_extended_20260909_0731_writeback.json`；固定快照：`heartbeat_extended_20260909_0731_snapshot.json`。', '']
(B / 'heartbeat_extended_20260909_0731.md').write_text('\n'.join(md))
print(json.dumps({'new_cells': cells, 'predictions_audited': 608,
                  'paired_counts': paired_counts,
                  'pending': [{k:c[k] for k in ('benchmark','arm','n','expected_n')} for c in pending]}), flush=True)

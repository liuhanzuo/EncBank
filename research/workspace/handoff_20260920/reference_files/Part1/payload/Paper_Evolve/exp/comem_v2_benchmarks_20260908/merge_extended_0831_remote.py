"""Assemble only the fixed 08:31 snapshot's already CPU-verified new cells."""
import json
from pathlib import Path
from datetime import datetime, timezone

R = Path('/data/liuhanzuo/comem_v2_20260908')
B = R / 'workspace/exp/comem_v2_benchmarks_20260908'
def read(p):
    return json.loads(p.read_text())

r = read(B / 'heartbeat_extended_20260909_0831_ruler.json')
i = read(B / 'heartbeat_infinitebench_20260909_0831_new.json')
s = read(B / 'heartbeat_extended_20260909_0831_snapshot.json')
assert r['snapshot_timestamp_utc'] == i['snapshot_timestamp_utc'] == s['timestamp_utc']
assert r['checks']['new_ruler_raw_rescores'] == 100
assert i['checks']['official_rescores'] == i['checks']['new_source_token_pack_exact_input_own_cache_checks'] == 229
assert r['checks']['old_outputs_rescored'] == r['checks']['longeval_rescores'] == 0
assert i['checks']['reference_scorer_calls'] == i['checks']['reference_cache_lookups'] == 0
assert all(c['complete'] for c in r['writeback'] + i['writeback'])
assert len(r['writeback']) == 2 and len(i['writeback']) == 1
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
    'snapshot_file': str(B / 'heartbeat_extended_20260909_0831_snapshot.json'),
    'scope': 'New complete baseline cells only since 07:31; one fixed 08:31 heartbeat snapshot',
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
    'evidence_files': ['heartbeat_extended_20260909_0831_ruler.json',
                       'heartbeat_infinitebench_20260909_0831_new.json'],
}
assert d['new_predictions_audited'] == 329
old_cells = read(B / 'heartbeat_extended_20260909_0129.json')['writeback'] + read(B / 'heartbeat_extended_20260909_0731.json')['writeback']
mc_matrix = {c['arm']: {'n': c['n'], 'correct': c['correct'], 'score_percent': c['score_percent'], 'rescored_this_round': False}
             for c in old_cells if c['benchmark'] == 'infinitebench' and c['task'] == 'longbook_choice_eng'}
for c in i['writeback']:
    mc_matrix[c['arm']] = {'n': c['n'], 'correct': c['correct'], 'score_percent': c['score_percent'], 'rescored_this_round': True}
assert set(mc_matrix) == {'pub', 'pub_sink', 'cbos', 'fix_all', 'j0'}
d['complete_MC_five_method_matrix'] = mc_matrix
d['negative_results'] = [
    'Five-needle128k original0 and BOS4.4 remain far below the previously verified V2/replay93.2/100.',
    'LoCoMo baselines remain incomplete in the fixed snapshot; no category score is extrapolated.'
]
for name in ('heartbeat_extended_20260909_0831.json', 'heartbeat_extended_20260909_0831_writeback.json'):
    path = B / name
    assert not path.exists()
    value = d if not name.endswith('_writeback.json') else {
        'schema_version': 1, 'snapshot_timestamp_utc': s['timestamp_utc'],
        'new_complete_cells': cells, 'new_complete_cell_count': len(cells),
        'new_predictions_audited': 329, 'comparisons': i['comparisons'],
        'paired_counts': paired_counts,
        'pending': pending, 'protocol_limits': d['protocol_limits'],
        'evidence': str(B / 'heartbeat_extended_20260909_0831.json')}
    path.write_text(json.dumps(value, indent=2) + '\n')

md=['# 08:31 扩展任务基线独立核验','',
    f"固定实际快照：{s['timestamp_utc']} UTC（北京时间08:33:47）。仅纳入该时点主队列completed/exit_code0、原始记录完整的新增单元；不追收后来完成的输出。",'',
    '全部新源重建、tokenization、检索pack、cache和官方评分核验均在远端CPU执行，CUDA隐藏、Torch2线程/interop16、OMP2/MKL2。没有加载模型、启动GPU或修改论文、根计划、队列、同步器。','',
    '## 新完整格','',
    '| Benchmark / task | Length | Arm | n | Exact score (%) |','|---|---|---|---:|---:|']
for c in cells:md.append(f"| {c['benchmark']} / {c['task']} | {c.get('length','—')} | {c['arm']} | {c['n']} | {c['score_percent']:.10f} |")
md+=['','共3个新格、329个新预测。pub=原始CoMem，pub_sink=CoMem+BOS，cbos=isolated full KV，不是CacheBlend。实测0保留0，未完成分数为null。','',
     '## RULER：只审新五针128k两格','',
     '原版为0%，BOS为4.4%，均50例。100个raw输出按原CoMem RULER substring-recall评分重算，连续i0–49、完整summary、模型/reader参数、无OOM及远端准入日志均通过；与旧V2/j0的task/length/i/answers/n_tokens和CLI条件逐例一致。V2/replay原已核均值93.2/100未重评分。', '',
     '该适配任务是five-needle retrieval，iter_BM25、top12、chunk512、hop4、seed42，命令行max_new_tokens48由VT路径按原规则使用有效cap60。它不测事实组合。历史RULER未持久化完整输入token、运行时selected pack或generation-cache键；因此所有配对仅限同remote runtime下的metadata/CLI/seed，不升级为exact-input/pack核验，也不与其他runtime或tokenizer的独立draw混作配对。','',
     '下表胜/平/负方向均为**新基线相对旧参考**，n50，仅为描述性配对计数：','',
     '| New baseline | Reference | Wins | Ties | Losses | Mean delta (pp) |','|---|---|---:|---:|---:|---:|']
for c in r['writeback']:
    for ref,p in c['paired_contrasts'].items():md.append(f"| {c['arm']} | {ref} | {p['new_arm_wins']} | {p['ties']} | {p['new_arm_losses']} | {p['new_arm_minus_reference_pp']:.10f} |")
md+=['','保留原版0和BOS4.4这一低准确率结果。isolated五针128k及三基线256k单针/多键在快照中均不完整，不填局部分数。','',
     '## InfiniteBench MC：isolated KV完整229例','',
     '四个完整分片恰覆盖index0–228、229个唯一源ID。逐源重建官方vendored yarn-mistral裸模板、显式完整query和未截断的源文本；使用同一Qwen tokenizer重新分词、重建有序BM25 top12 pack。229个新预测逐例通过官方option-accuracy评分、原始答案及自己的SQLite精确输入generation-cache检查。官方答案文本提取规则保留，不以简单首字母评分替代。', '',
     'stock Qwen3-8B BF16/SDPA，isolated KV有效j36，写入BOS，repeat_kv/禁GQA策略；chunk512、top12、greedy cap40、首步EOS抑制和之后自然EOS。没有chat wrapper、source truncation或padding。V2/j0仅用旧已核分数并确认与新臂完整source tokens/ordered pack相同；没有重评分旧预测或查询旧cache。', '']
for c in i['writeback']:md.append(f"- {c['arm']}：{c['correct']}/229，raw mean={c['raw_mean']:.15f}，{c['score_percent']:.10f}%，论文显示{c['display']}。")
md+=['','MC全部五种方法现在完整；此前四种方法分数沿用已核报告，不重新评分：','',
     '| Method | Correct / n | Score (%) | New audit this round |','|---|---|---:|---|']
for arm in ('pub','pub_sink','cbos','fix_all','j0'):
    c=mc_matrix[arm];md.append(f"| {arm} | {c['correct']}/{c['n']} | {c['score_percent']:.10f} | {c['rescored_this_round']} |")
md+=['','新isolated KV相对旧参考的胜/平/负，均n229：','',
     '| New baseline | Reference | Wins | Ties | Losses |','|---|---|---:|---:|---:|']
for c in paired_counts:md.append(f"| {c['new_arm']} | {c['reference_arm']} | {c['new_arm_wins']} | {c['ties']} | {c['new_arm_losses']} |")
for c in i['comparisons']:md.append(f"- V2相对{c['new_arm']}的均值差：{c['v2_minus_new_arm_pp']:+.10f}点。")
md+=['','这些结果只覆盖English long-book MC这一完整229例任务，不代表全InfiniteBench，也不作显著性、跨任务难度或系统性能结论。完整源tokenization的长度警告不表示模型一次读入全部源文本；实际有序read-pack范围另列机器JSON。', '',
     '## LoCoMo仍未完整：只记覆盖','',
     '| Arm | Complete-shard n | Formal n | Eligible score |','|---|---:|---:|---|']
for c in pending:md.append(f"| {c['arm']} | {c['n']} | {c['expected_n']} | — |")
md+=['','覆盖只计固定快照内完整shard。核对receipt、原配置期望index/id/task、完整分片索引、跨分片唯一性和类别计数；不评分局部输出。类别完整分母为282/321/96/841/446，总计1986。三臂均无完整类别，机器JSON保留各类别已完成n与null分数，不能用部分分片均值填正式表。','',
     'LongEval与此前MC原版/BOS、multi-key128k三格均未重新评分。此前LongEval缺runtime pack、RULER只保留legacy metadata配对的限制不变。','',
     '## 紧凑写回','',
     '在既有附录RULER基线块增加Five-needle/128k行：0/4.4/—；在既有MC行仅补isolated KV新完整值，其余四方法沿用已核数字。LoCoMo及未完成RULER保持空白，不新增正文表。','',
     '文件：`heartbeat_extended_20260909_0831.md/.json`、`heartbeat_extended_20260909_0831_snapshot.json`和直接写回`heartbeat_extended_20260909_0831_writeback.json`。','']
(B/'heartbeat_extended_20260909_0831.md').write_text('\n'.join(md))
print(json.dumps({'writeback':cells,'paired_counts':paired_counts,'MC_matrix':mc_matrix,'predictions':329,'pending':pending}),flush=True)

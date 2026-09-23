"""Recheck saved answers and report question-paired quality and Write costs."""
import csv
import gzip
import json
import statistics
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'encbank_overlap_20260913'
sys.path.insert(0, str(PRIOR))
from common import dump, score
from transformers import AutoTokenizer


def main():
    out = HERE / 'results'
    complete = json.loads((out / 'complete.json').read_text())
    assert complete['complete'] and complete['examples'] == 200
    metadata = json.loads((out / 'metadata.json').read_text())
    with gzip.open(out / 'paired_inputs.jsonl.gz', 'rt', encoding='utf-8') as handle:
        inputs = {row['id']: row for row in map(json.loads, handle)}
    predictions = {}
    for line in (out / 'predictions.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line); key = (row['id'], row['arm'])
        assert key not in predictions
        predictions[key] = row
    costs = {}
    for line in (out / 'costs.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line); key = (row['id'], row['arm'], row['rep'])
        assert key not in costs
        costs[key] = row
    assert len(inputs) == 200
    assert set(predictions) == {(ident, arm) for ident in inputs for arm in ('w0','w32')}
    assert set(costs) == {(ident, arm, rep) for ident in inputs for arm in ('w0','w32') for rep in (-1,0,1,2)}
    failures = sum(row['status'] != 'ok' for row in predictions.values()) + sum(row['status'] != 'ok' for row in costs.values())
    if failures:
        dump(HERE / 'summary.json', dict(full_support_complete=False, failed_records=failures,
                                       scores=None, timing=None, reason='No full-cohort average with missing/OOM records'))
        (HERE / 'RESULTS_zh.md').write_text('# Qasper 配对实验\n\n存在 OOM/缺失测量，完整分母下暂无可报告结果；详见 summary.json。\n', encoding='utf-8')
        return
    tokenizer = AutoTokenizer.from_pretrained(metadata['model'], local_files_only=True)
    pairs = []
    for ident, source in inputs.items():
        pair = dict(id=ident)
        persistent = set()
        for arm in ('w0','w32'):
            prediction = predictions[(ident, arm)]
            text = tokenizer.decode(prediction['generated_ids'], skip_special_tokens=True)
            assert text == prediction['prediction']
            recomputed = score(source, text)
            assert abs(recomputed - prediction['score']) < 1e-12
            assert prediction['selected'] == source['selected']
            assert prediction['max_new_tokens'] == source['budget'] == 128
            measured = [costs[(ident, arm, rep)] for rep in (0,1,2)]
            assert all(row['selected'] == source['selected'] for row in measured)
            persistent.update(row['persistent_bytes'] for row in measured)
            pair[arm+'_f1'] = 100*recomputed
            pair[arm+'_write_median_ms'] = statistics.median(row['document_write_ms'] for row in measured)
            pair[arm+'_generated_tokens'] = len(prediction['generated_ids'])
        assert len(persistent) == 1
        assert predictions[(ident,'w0')]['cached_positions'] == predictions[(ident,'w32')]['cached_positions']
        pair['persistent_bytes'] = persistent.pop()
        pair['source_positions'] = costs[(ident,'w0',0)]['source_positions']
        pair['f1_delta_pp'] = pair['w32_f1']-pair['w0_f1']
        pair['write_delta_ms'] = pair['w32_write_median_ms']-pair['w0_write_median_ms']
        pairs.append(pair)
    rng = np.random.default_rng(20260917)
    indices = rng.integers(0, len(pairs), size=(20000,len(pairs)))
    def paired_delta(column):
        values = np.array([row[column] for row in pairs])
        means = values[indices].mean(axis=1)
        return dict(mean=float(values.mean()), ci95=[float(x) for x in np.quantile(means,[.025,.975])],
                    bootstrap_replicates=20000, unit='paired Qasper question')
    arms = {}
    for arm in ('w0','w32'):
        times = [row[arm+'_write_median_ms'] for row in pairs]
        arms[arm] = dict(f1=statistics.mean(row[arm+'_f1'] for row in pairs),
                        write_mean_ms=statistics.mean(times), write_median_ms=statistics.median(times))
    result = dict(full_support_complete=True, questions=200, predictions=400,
                  measured_write_records=1200, excluded_warmups=400, old_predictions_reused=False,
                  cpu_decode_and_rescore_equal=True, selected_ids_and_cached_shapes_equal=True,
                  arms=arms, f1_delta_pp=paired_delta('f1_delta_pp'),
                  write_delta_ms=paired_delta('write_delta_ms'),
                  mean_write_ratio=arms['w32']['write_mean_ms']/arms['w0']['write_mean_ms'],
                  gpu=metadata['gpu'], torch=metadata['torch'], transformers=metadata['transformers'],
                  quality_and_timing_same_run=True,
                  timing_scope='Full document Write, including transfer to CPU-pinned store; no query/decode',
                  interpretation='Question-paired result on this Qasper protocol; no claim of general QA improvement')
    dump(HERE / 'summary.json', result)
    with (HERE / 'paired_results.csv').open('w', newline='', encoding='utf-8-sig') as target:
        writer = csv.DictWriter(target, fieldnames=list(pairs[0])); writer.writeheader(); writer.writerows(pairs)
    lines = ['# Qasper w=0 / w=32：同机严格配对结果', '',
             'RTX5090，Qwen3-8B，j=12，原 final4000 adapter。200例两臂全部重新生成，未拼接旧预测；检索块、顺序、query、128-token上限与评分函数固定。', '',
             '| 方法 | F1（0–100） | Write均值（ms） | Write中位数（ms） |',
             '|---|---:|---:|---:|']
    for arm, cell in arms.items():
        lines.append(f"| {arm} | {cell['f1']:.2f} | {cell['write_mean_ms']:.2f} | {cell['write_median_ms']:.2f} |")
    quality, timing = result['f1_delta_pp'], result['write_delta_ms']
    lines += ['', f"w32−w0 的 F1差为 {quality['mean']:+.2f} 点，95%配对区间 [{quality['ci95'][0]:+.2f}, {quality['ci95'][1]:+.2f}]。",
              f"平均 Write差为 {timing['mean']:+.2f} ms，95%配对区间 [{timing['ci95'][0]:+.2f}, {timing['ci95'][1]:+.2f}]；均值比 {result['mean_write_ratio']:.3f}×。", '',
              '每例每臂三次正式Write先取中位数，再汇总200例；另有一次预热不计入。区间按问题配对重采样20,000次，不把计时重复当作独立样本。', '',
              'Write包含完整文档下层编码及CPU-pinned缓存建立，排除加载、分词、检索、query编码和解码。两臂缓存字节与检索位置完全一致；w32扩大了Write可见的前文，不能据此将差异完全归因于边界修复。', '',
              '全部400条答案已从token重新解码并复评分。正、负或持平结果均保留，不将本设置的结果推广为普遍自然问答增益。原始记录见results/，逐例配对见paired_results.csv。']
    (HERE / 'RESULTS_zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()

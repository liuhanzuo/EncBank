"""Independently re-decode/rescore, report every final arm and paired uncertainty."""
import collections, csv, gzip, json
import config
import numpy as np
from train_support import atomic_json, digest


def main():
    from transformers import AutoTokenizer
    from eval import longeval
    model='/srv/encbank/legacy_workspace/models/Qwen3-8B' if config.os.name=='nt' else config.MODEL
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    with gzip.open(config.SAMPLES,'rt',encoding='utf-8') as f:
        samples={r['id']:r for r in map(json.loads,f)}
    assert len(samples)==600
    names=[a['name'] for a in config.ARMS]+['original_lora32_4000','frozen_j12','selected_text_replay']
    all_records={}
    for name in names:
        out=config.ROOT/'evaluation'/name
        assert json.loads((out/'complete.json').read_text())['records']==600
        protocol=json.loads((out/'protocol.json').read_text())
        assert protocol['samples_sha256']==digest(config.SAMPLES)
        rows=[json.loads(l) for l in (out/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows)==600 and len({r['id'] for r in rows})==600
        assert {r['id'] for r in rows}==samples.keys()
        all_records[name]={}
        for row in rows:
            sample=samples[row['id']]
            assert row['status']=='ok' and row['arm']==name
            assert row['selected']==sample['selected'] and row['budget']==16 and len(row['generated_ids'])<=16
            assert (row['split'],row['cell'])==(sample['split'],sample['cell'])
            text=tok.decode(row['generated_ids'],skip_special_tokens=True)
            score=float(longeval.extract_prediction(text)==sample['answers'][0])
            assert row['prediction']==text and row['score']==score
            all_records[name][row['id']]=row
    result=[];paired=[];rng=np.random.default_rng(20260917)
    for split in ('existing','confirmation'):
        for cell in ('longeval_8k','longeval_16k','longeval_32k','macro'):
            ids=sorted(k for k,s in samples.items() if s['split']==split and (cell=='macro' or s['cell']==cell))
            assert len(ids)==(300 if cell=='macro' else 100)
            # Paired bootstrap; macro is stratified by length, preserving 100/100/100.
            strata=[np.array([i for i,k in enumerate(ids) if samples[k]['cell']==c]) for c in ('longeval_8k','longeval_16k','longeval_32k')]
            strata=[s for s in strata if len(s)]
            draws=np.concatenate([rng.choice(s,size=(20000,len(s)),replace=True) for s in strata],axis=1)
            baseline=np.array([all_records['lora32_4000'][k]['score'] for k in ids])
            for name in names:
                values=np.array([all_records[name][k]['score'] for k in ids])
                result.append(dict(split=split,cell=cell,arm=name,n=len(ids),accuracy_pct=float(values.mean()*100)))
                if name!='lora32_4000':
                    delta=values-baseline;ci=np.quantile(delta[draws].mean(axis=1)*100,[.025,.975])
                    paired.append(dict(split=split,cell=cell,arm=name,reference='lora32_4000',n=len(ids),
                        delta_pp=float(delta.mean()*100),ci95_low=float(ci[0]),ci95_high=float(ci[1]),
                        wins=int((delta>0).sum()),losses=int((delta<0).sum())))
    delivery=config.ROOT/'delivery';delivery.mkdir(exist_ok=True)
    for filename,rows in [('results.csv',result),('paired_vs_baseline.csv',paired)]:
        with (delivery/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Qwen3-8B 自蒸馏对照结果','',
        '五组固定训练配置、原 adapter、frozen j=12 与相同检索文本 replay 均完成。所有 4800 条输出已独立解码与重新评分，样本、检索块和输出预算一致。','']
    for split in ('existing','confirmation'):
        lines += [f'## {split}：各长度 100 题','', '| 方法 | 8k | 16k | 32k | 宏平均 |','|---|---:|---:|---:|---:|']
        for name in names:
            vals=[next(r['accuracy_pct'] for r in result if r['split']==split and r['cell']==c and r['arm']==name)
                  for c in ('longeval_8k','longeval_16k','longeval_32k','macro')]
            lines.append('| '+name+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
        lines+=['','相对本次重训 rank32/4000 的宏平均差异（百分点）：','']
        for r in paired:
            if r['split']==split and r['cell']=='macro' and r['arm'] in [a['name'] for a in config.ARMS]:
                lines.append(f"- {r['arm']}: {r['delta_pp']:+.2f}, 95% 配对区间 [{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}]")
        lines.append('')
    lines+=['这些是固定单训练 seed 的探索性结果；区间反映问题采样，不包含训练随机性，也未校正多重比较。请分别检查既有样本和独立种子样本；不根据此结果宣称跨任务提升。8000 步组使用预设 8000 步 cosine 日程；全参数组更换为独立 teacher 且下层也更新，缓存必须重建。共享集群训练耗时不作为 infra 测速。']
    (config.ROOT/'RESULTS_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic_json(delivery/'verification.json',dict(complete=True,records=4800,groups=8,independent_decode_and_score=True,
        same_samples_selection_budget=True,bootstrap_samples=20000,bootstrap_seed=20260917,macro_bootstrap_stratified_by_length=True))


if __name__=='__main__':
    main()

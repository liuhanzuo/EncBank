"""Verify completed remote evidence against saved generation; no API calls."""
import collections, csv, hashlib, json, math, tarfile
from pathlib import Path, PurePosixPath
from locomo_judge_reference import _REFUSAL_RE

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'delivery'
ARCHIVE=OUT/'locomo_astra_judge_complete_20260917.tar.gz'
LOCAL=ROOT/'judge_gpt6_astra/formal_new_models'
PREFIX='judge_gpt6_astra/formal_new_models/'
LABELS={'cache_lora':'CoMem','cache_without_lora':'CoMem (without LoRA)',
        'replay_base':'Selected-text replay','replay_shared_lora':'Replay (shared CoMem LoRA)',
        'kvdirect':'KV-Direct (full source)','streamingllm':'StreamingLLM-style reference',
        'hcache_style':'HCache-style reference'}


def digest(row,protocol):
    stimulus=dict(protocol=protocol['protocol_id'],category=int(row['category']),status=row['status'],
        question=row['question'],gold=' OR '.join(map(str,row['answers'])),pred=row['pred'])
    return hashlib.sha256(json.dumps(stimulus,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def main():
    mainfiles={};votes={};calls=collections.defaultdict(list);events={};answers={};members=0
    # Read the complete archive without replacing the earlier local failed-call evidence.
    with tarfile.open(ARCHIVE,'r|gz') as tar:
        for member in tar:
            assert not PurePosixPath(member.name).is_absolute() and '..' not in PurePosixPath(member.name).parts
            if member.isdir():continue
            assert member.isfile(),member.name
            members+=1
            name=member.name
            if name.startswith(PREFIX+'cache/'):
                parts=PurePosixPath(name).parts;key=parts[3]
                assert len(key)==64
                if parts[-1]=='decision.json':
                    assert key not in votes;votes[key]=json.load(tar.extractfile(member))
                elif parts[-1]=='result.json':
                    calls[key].append((str(PurePosixPath(name).parent),json.load(tar.extractfile(member))))
                elif parts[-1]=='events.jsonl':
                    lines=tar.extractfile(member).read().decode('utf-8').splitlines();items=[]
                    for line in lines:
                        try:items.append(json.loads(line))
                        except ValueError:pass
                    events[str(PurePosixPath(name).parent)]=dict(
                        completed=any(e.get('type')=='turn.completed' for e in items),
                        tool_actions=sum(e.get('item',{}).get('type') in ('command_execution','file_change','mcp_tool_call','web_search') for e in items))
                elif parts[-1]=='answer.txt':
                    answers[str(PurePosixPath(name).parent)]=tar.extractfile(member).read().decode('utf-8').strip()
            elif name in [PREFIX+n for n in ('judge_decisions.jsonl','protocol.json','status.json','transport_calibration.json','worker_status.json')]+['judge_inputs/manifest.json']:
                mainfiles[name]=tar.extractfile(member).read()
    protocol=json.loads(mainfiles[PREFIX+'protocol.json']);status=json.loads(mainfiles[PREFIX+'status.json'])
    calibration=json.loads(mainfiles[PREFIX+'transport_calibration.json'])
    assert protocol['model']=='gpt-6-astra' and protocol['protocol_id']=='midcache-locomo-gpt6-astra-v1'
    assert protocol['reasoning_effort']=='low' and protocol['seed'] is None
    assert status['available_complete'] and not status['errors'] and status['decisions']==27804 and status['oom']==0
    assert json.loads(mainfiles[PREFIX+'worker_status.json'])['phase']=='COMPLETE'
    assert calibration['passed'] and calibration['n']==8
    assert all(c['transport_ok'] and c['returned']==c['expected'] for c in calibration['cases'])
    manifest=json.loads((LOCAL/'input_manifest.json').read_text())
    assert manifest==json.loads(mainfiles['judge_inputs/manifest.json'])
    inputs={};question_map={};counts=collections.Counter()
    for entry in manifest['files']:
        rawdir=ROOT/'results_locomo'/entry['cohort']/f"shard{entry['shard']}"
        rawrows=[json.loads(l) for l in (rawdir/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        raw={(r['arm'],r['id']):r for r in rawrows};assert len(raw)==len(rawrows)==entry['records']
        rows=[json.loads(l) for l in (LOCAL/'inputs'/entry['file']).read_text(encoding='utf-8').splitlines()]
        assert len(rows)==entry['records']
        for row in rows:
            key=(row['cohort'],row['arm'],row['id']);assert key not in inputs
            source=raw[(row['arm'],row['id'])]
            assert row['status']==source['status']=='ok' and row['pred']==source['text']
            assert row['answers']==source['answers'] and row['category']==source['extra']['category']
            assert row['cohort']==entry['cohort'] and row['shard']==entry['shard']
            q=(row['question'],row['answers'],row['category'])
            assert question_map.setdefault(row['id'],q)==q
            inputs[key]=row;counts[key[:2]]+=1
    assert len(inputs)==27804 and len(question_map)==1986 and len(counts)==14 and set(counts.values())=={1986}
    rows=[json.loads(l) for l in mainfiles[PREFIX+'judge_decisions.jsonl'].decode('utf-8').splitlines()]
    decisions={};semantic=set();local=set()
    for row in rows:
        key=(row['cohort'],row['arm'],row['id']);assert key not in decisions and key in inputs
        src=inputs[key];ident=digest(src,protocol)
        assert row['stimulus_digest']==ident and row['category']==src['category'] and row['shard']==src['shard']
        assert row['status']=='ok' and row['reasoning_effort']=='low' and row['seed'] is None
        vote=votes[ident]
        for item in ('judge_correct','judge_raw','model','protocol_id','stimulus_digest'):
            assert row[item]==vote[item],(key,item)
        assert row['model']==protocol['model'] and row['protocol_id']==protocol['protocol_id']
        assert row['judge_correct'] in (0,1)
        if int(src['category'])==5:
            assert row['judge_raw']=='LOCAL_ABSTENTION_RULE'
            assert row['judge_correct']==int(not src['pred'].strip() or bool(_REFUSAL_RE.search(src['pred'])))
            local.add(ident)
        else:
            assert row['judge_raw'] in ('CORRECT','WRONG')
            assert row['judge_correct']==int(row['judge_raw']=='CORRECT')
            semantic.add(ident)
        decisions[key]=row
    assert decisions.keys()==inputs.keys() and len(rows)==27804
    failed_attempts=0
    for ident in semantic:
        successful=[]
        for path,result in calls[ident]:
            if not result['ok']:
                failed_attempts+=1;continue
            assert result['model_requested']=='gpt-6-astra' and result['returncode']==0 and result['turn_completed']
            assert result['tool_actions']==0 and result['credential_persisted'] is False
            assert result['answer'].strip()==answers[path]==votes[ident]['judge_raw']
            assert events[path]['completed'] and events[path]['tool_actions']==0
            successful.append(path)
        assert len(successful)==1,(ident,len(successful))
    assert len(semantic)==16624 and len(semantic)+len(local)==len(votes)
    aggregate=[];categories=[];deltas={}
    for model,j in [('Qwen3.5-9B',6),('Qwen3.8-27B',21)]:
        for arm,label in LABELS.items():
            group=[r for k,r in decisions.items() if k[:2]==(model,arm)]
            assert len(group)==1986
            result=dict(model=model,j=j,arm=arm,method=label,n=1986,correct=sum(r['judge_correct'] for r in group))
            result['score_pct']=100*result['correct']/1986
            sem=[r for r in group if int(r['category'])<5]
            result['semantic_1_to_4_n']=len(sem)
            result['semantic_1_to_4_pct']=100*sum(r['judge_correct'] for r in sem)/len(sem)
            assert len(sem)==1540
            assert math.isclose(result['score_pct'],status['summary'][model][arm]['score'],abs_tol=1e-9)
            aggregate.append(result)
            for category in range(1,6):
                cell=[r for r in group if int(r['category'])==category]
                categories.append(dict(model=model,j=j,arm=arm,method=label,category=category,n=len(cell),
                    correct=sum(r['judge_correct'] for r in cell),score_pct=100*sum(r['judge_correct'] for r in cell)/len(cell)))
        value={r['arm']:r for r in aggregate if r['model']==model}
        deltas[model]=dict(overall_pp=value['cache_lora']['score_pct']-value['cache_without_lora']['score_pct'],
            semantic_pp=value['cache_lora']['semantic_1_to_4_pct']-value['cache_without_lora']['semantic_1_to_4_pct'])
    for name,values in [('locomo_astra_results.csv',aggregate),('locomo_astra_categories.csv',categories)]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    complete=OUT/'locomo_astra_remote_complete';complete.mkdir(exist_ok=True)
    for path,payload in mainfiles.items():
        target=complete/Path(path).name
        if target.exists():assert target.read_bytes()==payload
        else:target.write_bytes(payload)
    verification=dict(complete=True,records=27804,groups=14,questions_per_group=1986,semantic_records=21560,
        local_rule_records=6244,unique_semantic_calls=len(semantic),unique_local_votes=len(local),oom=0,
        unscored=0,failed_attempts_retained_not_scored=failed_attempts,archive_files=members,
        normalized_inputs_match_original_predictions=True,exact_stimulus_match=True,
        semantic_votes_backed_by_successful_call_and_events=True,local_rule_recomputed=True,
        model='gpt-6-astra',underlying_model_snapshot_verified=False,adaptation_deltas=deltas)
    (OUT/'locomo_astra_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    lines=['# 两个新模型的 LoCoMo 完整结果','',
        '两模型各 1,986 题、七个方法，共 27,804 条评分全部完成并核验，0 OOM、0 未评分。类别 1–4 用同一 gpt-6-astra 服务别名和固定盲评 prompt；类别 5 沿用原来的本地拒答规则。','',
        '逐条检查了原始预测、gold、类别、样本标识、评分协议、stimulus digest、有效调用结果及事件记录，并重新计算本地拒答规则。相同输入的 21,560 条语义评分共享 16,624 次有效调用；失败尝试保留但不作为错误答案计分。','',
        '| 方法 | Qwen3.5-9B 总分 | Qwen3.8-27B 总分 | 9B 类别1–4 | 27B 类别1–4 |',
        '|---|---:|---:|---:|---:|']
    for arm,label in LABELS.items():
        a=next(r for r in aggregate if r['model']=='Qwen3.5-9B' and r['arm']==arm)
        b=next(r for r in aggregate if r['model']=='Qwen3.8-27B' and r['arm']==arm)
        lines.append(f"| {label} | {a['score_pct']:.2f} | {b['score_pct']:.2f} | {a['semantic_1_to_4_pct']:.2f} | {b['semantic_1_to_4_pct']:.2f} |")
    lines+=['','总分按全部 1,986 题微平均；类别1–4按 1,540 题微平均。另 446 题为类别5，不经过语义 Judge。9B 使用 j=6，27B 使用 j=21；两者分别训练 4,000 步。','']
    for model,d in deltas.items():
        lines.append(f"- {model}：CoMem 相比 without LoRA 的总分变化 {d['overall_pp']:+.2f} pp；类别1–4变化 {d['semantic_pp']:+.2f} pp。")
    lines+=['','本结果仅覆盖这两个新模型，不能将旧 Qwen3-8B 的 GPT-4o 分数改标为 Astra。API 服务的具体底层模型快照未经核实；8个固定 sanity checks 不代表人工评审一致性。单训练 seed 结果不支持跨模型显著性或速度优势结论。',
        '', 'Selected-text replay 使用与 CoMem 相同的检索文本；shared CoMem LoRA 是复用 CoMem adapter，并非独立训练 replay。KV-Direct 使用完整 source；StreamingLLM-style 与 HCache-style 均为本文兼容参考实现。本批不包含另一次 Qwen3-8B 的 CacheBlend-style LoRA 实验。',
        '', '数据：`locomo_astra_results.csv`、`locomo_astra_categories.csv`、`locomo_astra_verification.json`。原始评分与所有调用证据：`locomo_astra_judge_complete_20260917.tar.gz`；原始生成答案：`locomo_verified_8shards.tar.gz`。论文尚未写入这些新结果。']
    (OUT/'LOCOMO_ASTRA_RESULTS_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(verification));print(json.dumps(aggregate,ensure_ascii=False))


if __name__=='__main__':main()

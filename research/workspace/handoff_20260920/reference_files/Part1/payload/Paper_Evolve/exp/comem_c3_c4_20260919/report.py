"""Aggregate real request tails and original quality scoring, retaining all denominators."""
import collections,datetime,json,math,statistics
import config
import common as original
from eval.ruler import _string_match_all_one
from queue_metrics import summarize
from controlled_train_core import atomic_json
ROOT=config.ROOT
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def main():
    serving=[];quality=[];latency=[];layout=None;initial=None
    for method in ['selected_replay','comem']:
        for c in [1,4]:
            out=ROOT/'serving/rep0/32768'/(method+'_c'+str(c));wait=read(out/'parent_exit.json')
            assert wait['actual_wait'] and wait['returncode']==0
            result=read(out/(method+'_c'+str(c))/'complete.json')
            if result['status']=='ok':
                rows=lines(out/(method+'_c'+str(c))/'requests.jsonl')
                assert len(rows)==1000 and len({r['id'] for r in rows})==1000
                for row in rows:
                    assert 0<=row['arrival']<=row['start']<=row['first']<=row['end']
                    assert math.isclose(row['ttft_s'],row['first']-row['arrival'],abs_tol=1e-7)
                    assert math.isclose(row['e2e_s'],row['end']-row['arrival'],abs_tol=1e-7)
                    assert all(len(ids)==32 for ids in row['detail']['generated_ids'])
                    assert row['batch_size']==c
                calculated=summarize(rows,max(r['end'] for r in rows),32)
                for key in ['requests_per_s','output_tokens_per_s','wall_s']:
                    assert math.isclose(calculated[key],result[key],rel_tol=1e-10)
                for metric in ['queue_s','ttft_s','e2e_s','service_s']:
                    assert calculated[metric]==result[metric]
            serving.append(dict(method=method,concurrency=c,result=result,metadata=read(out/'metadata.json'),cpu_state=read(out/'persistent_cpu.json')))
    import gzip
    with gzip.open(ROOT/'data/quality.jsonl.gz','rt',encoding='utf-8') as f:samples={r['id']:r for r in map(json.loads,f)}
    for j in [6,12,18]:
        tag=config.tag(j,42);training=ROOT/'training'/tag;out=ROOT/'quality'/tag
        meta=read(training/'metadata.json');names=read(training/'trainable_parameters.json')
        if layout is None:layout=names;initial=meta['initial_adapter_sha256']
        assert names==layout and meta['initial_adapter_sha256']==initial
        assert meta['trainable_parameters']==43646976
        train=read(training/'complete.json');status=read(training/'status.json')
        assert train['step']==4000 and train['train_input_tokens']==16384000 and train['loss_query_tokens']==2048000
        rows=lines(out/'predictions.jsonl');assert len(rows)==len({(r['id'],r['arm']) for r in rows})==800
        cells=collections.defaultdict(list)
        for r in rows:
            sample=samples[r['id']]
            assert r['status']=='ok' and r['j']==j and r['selected']==sample['selected']
            assert r['source_sha256']==sample['source_sha256'] and r['adapter_sha256']==train['adapter_sha256']
            score=_string_match_all_one(r['prediction'],sample['answers']) if sample['benchmark']=='ruler' else original.score(sample,r['prediction'])
            assert math.isclose(score,r['score'],abs_tol=1e-10)
            cells[(r['arm'],r['cell'])].append(score)
        scores={}
        for arm in ['raw','comem']:
            assert len(cells[(arm,'qasper')])==200 and len(cells[(arm,'multikey_8k')])==len(cells[(arm,'multikey_16k')])==100
            scores[arm]=dict(multikey=50*(statistics.mean(cells[(arm,'multikey_8k')])+statistics.mean(cells[(arm,'multikey_16k')])),
                qasper=100*statistics.mean(cells[(arm,'qasper')]))
        quality.append(dict(j=j,scores=scores,parameters=43646976,input_tokens=16384000,loss_tokens=2048000,
            training_elapsed_s=status['elapsed_training_s'],adapter_sha256=train['adapter_sha256'],records=800))
        out=ROOT/'latency'/tag;wait=read(out/'parent_exit.json');assert wait['actual_wait'] and wait['returncode']==0
        rows=lines(out/'records.jsonl');assert len(rows)==120 and len({(r['id'],r['arm'],r['rep']) for r in rows})==120
        formal=[r for r in rows if not r['warmup']];assert len(formal)==90
        arms={}
        for arm in ['raw','comem']:
            part=[r for r in formal if r['arm']==arm];assert len(part)==45
            assert all(r['selected']==samples[r['id']]['selected'] for r in part)
            arms[arm]={metric:statistics.median(r[metric] for r in part) for metric in ['write_ms','read_ms','ttft_ms','retrieval_ms','fetch_ms','query_write_ms']}
        latency.append(dict(j=j,arms=arms,protocol=read(out/'protocol.json')))
    summary=dict(at=datetime.datetime.now().astimezone().isoformat(),complete=True,serving=serving,quality=quality,latency=latency,
        same_trainable_layout=True,same_initial_adapter_sha256=initial,quality_predictions=2400,quality_errors=0,
        serving_target_requests=4000,serving_runs=1,training_seeds=[42],FLOPs_matched=False)
    atomic_json(ROOT/'summary.json',summary)
    text='# C3/C4 补测结果\n\n'+summary['at']+'。以下区分服务请求数与质量样本数；固定32-token服务输出不用于QA评分。\n\n'
    text+='## C3：同一参考服务承接并发请求\n\n同一RTX5090、Qwen3-8B、共享rank32/alpha32/final4000 adapter；每格独立模型进程，一份权重。FIFO同步微批，最多4请求/批，独立batch行KV状态，不是连续batching。3个32k PG19文档、96个固定续写请求模板重复构造负载，固定重复顺序、无答案/前缀缓存。BM25索引和H bank离线准备，实际迭代检索在线执行。CoMem j12/w0/CPU-pinned H。CPU线程2、CUDA allocator cap27GiB。\n\n'
    text+='| 方法 | 并发 | 完成/失败 | req/s | TTFT p95/p99 ms | E2E p95/p99 ms | allocated/reserved峰值 GiB |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in serving:
        s=r['result'];name='Raw replay' if r['method']=='selected_replay' else 'CoMem'
        if s['status']=='ok':
            text+=f"| {name} | {r['concurrency']} | 1000/0 | {s['requests_per_s']:.3f} | {s['ttft_s']['p95']:.2f}/{s['ttft_s']['p99']:.2f} | {s['e2e_s']['p95']:.2f}/{s['e2e_s']['p99']:.2f} | {s['peak_allocated_bytes']/2**30:.3f}/{s['peak_reserved_bytes']/2**30:.3f} |\n"
        else:text+=f"| {name} | {r['concurrency']} | {s.get('completed_before_oom',0)}/{s.get('failed_batch',[])} | OOM | 未完成 | 未完成 | OOM |\n"
    raw4=next(x['result'] for x in serving if x['method']=='selected_replay' and x['concurrency']==4)
    comem4=next(x['result'] for x in serving if x['method']=='comem' and x['concurrency']==4)
    if raw4['status']==comem4['status']=='ok':
        text+=f"\nRaw replay 对同样检索出的文本块重新执行全部层；32k指源文档长度。此参考负载下，并发4时CoMem的请求吞吐提升{100*(comem4['requests_per_s']/raw4['requests_per_s']-1):.2f}%，TTFT p99降低{100*(1-comem4['ttft_s']['p99']/raw4['ttft_s']['p99']):.2f}%，E2E p99降低{100*(1-comem4['e2e_s']['p99']/raw4['e2e_s']['p99']):.2f}%。\n"
    text+='\nTTFT/E2E从真实提交时刻到首/末token在CPU可用，含排队、检索、fetch、query Write和模型执行；不含网络、输入token化、模型加载、离线文档Write和索引构建。p95/p99直接从每条请求计算，未先平滑。吞吐=完成请求数/首提交至最后完成的时间。仅一次探索性运行，每格1000请求使p99约由最慢10请求决定，不宣称多租户或更新竞争的生产结论。CPU持久状态及各请求原始时戳见summary.json和serving目录。GPU allocated/reserved是PyTorch allocator统计，不是整卡nvidia-smi峰值。\n\n'
    text+='## C4：固定适配容量、数据与步数的深度消融\n\n三组从同一冻结骨干和完全相同的初始LoRA独立训练。j=6/12/18，均适配18–35层七种Linear，rank=alpha=32；43,646,976可训练参数，名称/形状/初始权重逐项一致。每组4000步、16,384,000输入tokens（不含单独BOS）、2,048,000计损失query tokens；相同PG19、顺序、seed42、teacher、KL损失与优化器。数据/步数/容量匹配，不声称FLOPs相等。固定最终权重，不按测试选checkpoint。\n\n'
    text+='| j | 参数 M | 输入/计损失 tokens | Multikey raw/CoMem | Qasper raw/CoMem | CoMem Write ms | CoMem Read ms | 在线 TTFT raw/CoMem ms |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for q,t in zip(quality,latency):
        s=q['scores'];a=t['arms']
        text+=f"| {q['j']} | 43.647 | 16384000/2048000 | {s['raw']['multikey']:.2f}/{s['comem']['multikey']:.2f} | {s['raw']['qasper']:.2f}/{s['comem']['qasper']:.2f} | {a['comem']['write_ms']:.2f} | {a['comem']['read_ms']:.2f} | {a['raw']['ttft_ms']:.2f}/{a['comem']['ttft_ms']:.2f} |\n"
    text+='\n在相同适配层集合、参数量与数据/步数下，更深的切分仍使本次multikey质量下降，同时降低在线读取延迟、增加离线Write成本。Qasper的CoMem F1也随深度下降，但raw参考本身随独立adapter变化，需按各自配对结果解释。该共同布局控制与主实验的suffix-adapter布局不同，不替换原部署结果，也不证明j=12是通用最优深度。\n'
    text+='\n质量每深度两路径各400题：原保存RULER multikey 8k/16k各100，加修正边界的Qasper200；共2400条，全部按原评分函数复算。各路径使用相同输入、选块及顺序、48/128最大输出预算和自然EOS；multikey只是此子集，不是完整RULER macro。Qasper原4个边界受影响输入已重算BM25选块并固定给六条路径。\n\n成本统一在同一5090测量，选定每单元前5题（共15输入），各3次正式重复，逐输入预热单列；上表为45次正式记录中位数。Write含完整文档H和sink计算及CPU-pinned存储，索引构建单列。Read指后续层prefill、LM head和首token回传，不含decode；TTFT包括在线实际BM25、fetch、query Write和Read。原始source/selected pack固定，未把不同路径的自然生成时间当固定成本。训练实际时间、最终checkpoint指纹、完整曲线与时戳均保留。单种子结果不证明训练稳定性。原suffix-adapter部署曲线仍独立保留。\n'
    (ROOT/'RESULTS_zh.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(complete=True,report=str(ROOT/'RESULTS_zh.md'),quality=quality)))
if __name__=='__main__':main()

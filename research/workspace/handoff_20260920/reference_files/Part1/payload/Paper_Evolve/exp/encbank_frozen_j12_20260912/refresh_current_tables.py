"""Populate current benchmark tables from complete j=12 results, without revision history."""
from pathlib import Path
import json,re
HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]/'Encbank/paper_iclr2027_rewrite_20260912'
def replace_block(name,new_rows):
    p=PAPER/'sections'/name;s=p.read_text(encoding='utf-8');lines=s.splitlines()
    start=next(i for i,l in enumerate(lines) if l.startswith('Encbank (without LoRA)'))
    end=start+1
    while end<len(lines) and lines[end].startswith(' & '):end+=1
    lines[start:end]=new_rows
    p.write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    result=json.loads((HERE/'summary.json').read_text(encoding='utf-8'))
    assert result['complete'] and result['formal_examples']==5250 and len(result['cells'])==47
    def val(bench,task,length=''):return result['cells']['/'.join((bench,task,length))]['score']
    def row(items):return ' & '.join(items)+r' \\'
    lengths=('8k','16k','32k','64k','128k')
    lines=[]
    for i,(task,name) in enumerate((('niah_single_2','single-2'),('niah_multikey_1','multikey'),('variable_tracking','tracking'))):
        lines.append(row(['Encbank (without LoRA)' if i==0 else '',name,*[f'{val("ruler",task,l):.1f}' for l in lengths]]))
    replace_block('tab_ruler_a.tex',lines)
    values=[val('longeval','lines',l) for l in lengths]
    replace_block('tab_longeval_detail.tex',[row(['Encbank (without LoRA)',*[f'{v:.0f}' for v in values],f'{sum(values)/len(values):.1f}'])])
    lengths=('0k','1k','2k','4k','8k','16k','32k');lines=[]
    for i,task in enumerate(('qa1','qa2','qa5')):
        values=[val('babilong',task,l) for l in lengths]
        lines.append(row(['Encbank (without LoRA)' if i==0 else '',task,*[f'{v:.0f}' for v in values],f'{sum(values)/len(values):.1f}']))
    replace_block('tab_babilong_detail.tex',lines)
    values=[val('longbench',t) for t in ('narrativeqa','qasper','hotpotqa','2wikimqa','multifieldqa_en','musique')]
    replace_block('tab_longbench_detail.tex',[row(['Encbank (without LoRA)',*[f'{v:.2f}' for v in values],f'{result["scores"]["longbench"]:.2f}'])])
    p=PAPER/'sections/tab_accuracy.tex';s=p.read_text(encoding='utf-8')
    values=[result['scores'][b] for b in ('ruler','longeval','longbench','babilong','locomo','average')]
    new=row(['Encbank (without LoRA)',*[f'{v:.1f}' if i==1 else f'{v:.2f}' for i,v in enumerate(values)]])
    s=re.sub(r'^Encbank \(without LoRA\).*$',lambda m:new,s,flags=re.M)
    p.write_text(s,encoding='utf-8')
    print('Updated current j=12 benchmark rows from 5,250 scored examples.')
if __name__=='__main__':main()

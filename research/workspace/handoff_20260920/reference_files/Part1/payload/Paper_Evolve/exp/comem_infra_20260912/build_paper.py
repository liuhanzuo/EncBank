"""Compile and render the paper after local infra aggregation has succeeded."""
import json
import re
import shutil
import subprocess
from pathlib import Path
import fitz
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
paper = ROOT/'COMem/paper_iclr2027_rewrite_20260912'
build = ROOT/'tmp/comem_infra_20260912/build'
renders = ROOT/'tmp/comem_infra_20260912/render'
build.mkdir(parents=True, exist_ok=True)
renders.mkdir(parents=True, exist_ok=True)
for name in ('refs.bib', 'iclr2027_conference.bst', 'unsrtnat.bst'):
    shutil.copy2(paper/name, build/name)
for run in range(3):
    result = subprocess.run(['pdflatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error',
                             '-output-directory='+build.as_posix(),'main.tex'], cwd=paper,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (build/f'compile_{run+1}.txt').write_bytes(result.stdout)
    if result.returncode:
        print(result.stdout[-6000:].decode(errors='replace'))
        raise SystemExit(result.returncode)
    if run == 0:
        bibliography = subprocess.run(['bibtex', 'main'], cwd=build,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (build/'bibtex.txt').write_bytes(bibliography.stdout)
        if bibliography.returncode:
            print(bibliography.stdout.decode(errors='replace'))
            raise SystemExit(bibliography.returncode)
log = (build/'main.log').read_text(errors='replace')
issues = [line for line in log.splitlines() if any(x in line for x in
          ('Overfull', 'undefined', 'multiply defined', 'Rerun to get'))]
if issues:
    print('\n'.join(issues))
    raise SystemExit('Resolve compilation warnings before delivery')
doc = fitz.open(build/'main.pdf')
page_texts = [page.get_text() for page in doc]
print(json.dumps({'pages':len(doc),'refs_page':next((i+1 for i,t in enumerate(page_texts) if 'REFERENCES' in t),None),
                  'table2_pages':[i+1 for i,t in enumerate(page_texts) if 'Table 2:' in t],
                  'local_detail_pages':[i+1 for i,t in enumerate(page_texts) if 'LOCAL READ' in t]}, indent=2))
subprocess.run(['pdftoppm','-r','100','-png',str(build/'main.pdf'),str(renders/'page')], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
pages = [renders/f'page-{i:02d}.png' for i in range(1,len(doc)+1)]
for offset in range(0,len(pages),6):
    tiles=[]
    for f in pages[offset:offset+6]:
        im=Image.open(f).convert('RGB')
        im.thumbnail((420,550))
        tile=Image.new('RGB',(440,585),'#dddddd')
        tile.paste(im,((440-im.width)//2,20))
        ImageDraw.Draw(tile).text((12,565),f.stem,fill='black')
        tiles.append(tile)
    grid=Image.new('RGB',(440*3,585*2),'#dddddd')
    for i,t in enumerate(tiles): grid.paste(t,((i%3)*440,(i//3)*585))
    grid.save(renders/f'contact_{offset+1:02d}.png')
for i,page in enumerate(doc):
    if 'Table 2:' in page_texts[i]:
        page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(renders/'infra_page.png')
    if 'Table 1:' in page_texts[i]:
        assert all(x in page_texts[i] for x in ('CoMem','LongBench','HCache'))
(build/'qa_summary.json').write_text(json.dumps({'pages':len(doc),'issues':issues}),encoding='utf-8')
print(str(renders))

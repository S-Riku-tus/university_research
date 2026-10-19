"""Record selected original thesis pages and setup slides without editing them."""
from pathlib import Path
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
import zipfile

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
OUT=ROOT/'docs/thesis/2026-10-19_manuscript_audit'


def main():
    try:
        import fitz
        if not hasattr(fitz,'open'):raise ImportError()
    except ImportError:
        sys.path.insert(0,str(Path(os.environ['TEMP'])/'codex_thesis_pdf_tools'))
        import fitz
    records=[]
    excerpts=['# 学位論文原本とLaTeX導入資料の確認記録','',
              '2026-10-19。PDF番号は表紙を1とする物理ページ。全ページ精読・全図検査ではなく，目次から対象を絞った確認。原本は未変更。','']
    selections=[('*/8121048*.pdf',[1,2,3,4,7,8,9,16,17,32,33]),
                ('*/8124507*.pdf',[1,2,3,6,23,41,42]),
                ('*/8124532*.pdf',[1,2,3,4,7,8,9,10,34,40,58,59])]
    for pattern,pages in selections:
        p=next(ROOT.glob(pattern));d=fitz.open(p)
        records.append({'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                        'total_pdf_pages':len(d),'selected_pdf_pages':pages,'inspection':'selected text; not all-page visual review'})
        excerpts += ['## '+p.name,'']
        for page in pages:
            text=d[page-1].get_text()
            # Preserve complete TOCs and critical conclusions; bound other extracts.
            limit=6000 if page in [3,4,58,59] else 2800
            excerpts += [f'### PDF {page}', '', text[:limit].strip(), '']
    ppt=ROOT/'LaTeX/setting/LaTeX_setting.pptx'
    with zipfile.ZipFile(ppt) as z:
        selected=[1,4,7,26,27,28]
        for i in selected:
            root=ET.fromstring(z.read(f'ppt/slides/slide{i}.xml'))
            text=''.join(t.text or '' for t in root.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}t'))
            excerpts += [f'## LaTeX_setting.pptx / slide {i}','',text,'']
    records.append({'path':ppt.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(ppt.read_bytes()).hexdigest(),
                    'selected_slides':selected,'inspection':'OOXML text; images and all-page rendering not reviewed'})
    for p in [ROOT/'LaTeX/sample/document.tex',ROOT/'LaTeX/setting/contents_of_json.txt']:
        records.append({'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'inspection':'source text'})
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'reference_inventory.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'selected_reference_extracts.md').write_text('\n'.join(excerpts),encoding='utf-8')
    print(json.dumps(records,ensure_ascii=False))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()

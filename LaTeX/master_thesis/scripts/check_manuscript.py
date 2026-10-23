"""Check manuscript integrity and frozen evidence; optionally inspect the PDF."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import re
import sys

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]


def disk(path):
    path=Path(path).resolve()
    return Path('\\\\?\\'+str(path)) if os.name=='nt' else path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf',action='store_true')
    parser.add_argument('--preview-dir',type=Path,help='Write this revision preview outside earlier audit folders')
    args=parser.parse_args()
    errors=[]
    sources=[HERE/'main.tex', HERE/'preamble.tex', HERE/'frontmatter.tex',HERE/'references.tex',
             *sorted((HERE/'chapters').glob('*.tex')),*sorted((HERE/'appendices').glob('*.tex')),
             *sorted((HERE/'tables').glob('*.tex'))]
    content='\n'.join(p.read_text(encoding='utf-8') for p in sources)
    chapter_text='\n'.join(p.read_text(encoding='utf-8') for p in sorted((HERE/'chapters').glob('*.tex')))
    chapter_count=len(re.findall(r'\\chapter\{',chapter_text))
    if chapter_count!=6:errors.append(f'Expected 6 chapters, got {chapter_count}')
    bibliography=re.findall(r'\\bibitem\{([^}]+)\}',content)
    cited={x.strip() for group in re.findall(r'\\cite\{([^}]+)\}',content) for x in group.split(',')}
    labels=re.findall(r'\\label\{([^}]+)\}',content)
    referenced=set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',content))
    errors += ['Missing citation: '+x for x in sorted(cited-set(bibliography))]
    errors += ['Missing label: '+x for x in sorted(referenced-set(labels))]
    errors += ['Duplicate label: '+x for x in sorted(set(labels)) if labels.count(x)>1]
    errors += ['Duplicate bibliography key: '+x for x in set(bibliography) if bibliography.count(x)>1]
    for name in re.findall(r'\\input\{([^}]+)\}',content):
        if not (HERE/(name if name.endswith('.tex') else name+'.tex')).is_file():errors.append('Missing input: '+name)
    figures=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',content)
    for name in figures:
        if not (HERE/'figures'/name).is_file():errors.append('Missing figure: '+name)
    gaps=re.findall(r'\\need\{([^}]+)\}',content)
    if len(gaps)!=len(set(gaps)):errors.append('Duplicate gap ID')
    listed=set(re.findall(r'^([DLFV]\d{2})\s*&',(HERE/'appendices/gaps.tex').read_text(encoding='utf-8'),re.M))
    if listed!=set(gaps):errors.append('Gap registry and inline markers differ')
    manifest=json.loads((HERE/'data/source_manifest.json').read_text(encoding='utf-8'))
    historical_verified=[]
    live_code_differences=[]
    for name,expected in manifest['source_sha256'].items():
        source=ROOT/name
        snapshot=manifest.get('historical_code_snapshots',{}).get(name)
        if snapshot:
            source=(HERE/snapshot['snapshot_path']).resolve()
            if HERE.resolve() not in source.parents:
                errors.append('Historical source outside manuscript: '+name)
                continue
            historical_verified.append(name)
            if hashlib.sha256(disk(ROOT/name).read_bytes()).hexdigest()!=expected:
                live_code_differences.append(name)
        actual=hashlib.sha256(disk(source).read_bytes()).hexdigest()
        if actual!=expected:errors.append('Source changed: '+name)
    reference_inventory=ROOT/'docs/thesis/2026-10-19_manuscript_audit/reference_inventory.json'
    reference_count=0
    if reference_inventory.exists():
        for record in json.loads(reference_inventory.read_text(encoding='utf-8')):
            actual=hashlib.sha256(disk(ROOT/record['path']).read_bytes()).hexdigest()
            if actual!=record['sha256']:errors.append('Original reference changed: '+record['path'])
            reference_count+=1
    # Regenerate expected rows in memory by reading the original saved metrics.
    with (HERE/'data/pooled_metrics.csv').open(encoding='utf-8-sig') as f:metrics=list(csv.DictReader(f))
    look={(r['noise'],r['model_key']):r for r in metrics}
    core='ensemble__original3_mse';five='ensemble__original3_hgb_extra_trees'
    expected={('clean',core):34.07,('clean',five):27.80,('-20',core):81.14,('-20',five):75.23}
    for key,v in expected.items():
        if abs(float(look[key]['rmse_all_kW_m2'])-v)>.005:errors.append('Core manuscript numeric claim changed: '+str(key))
    for r in metrics:
        if int(float(r['tp']))+int(float(r['fn']))!=315:errors.append('Positive count differs')
        if int(float(r['tn']))+int(float(r['fp']))!=225:errors.append('Negative count differs')
    log=HERE/'build/main.log'
    warning_lines=[]
    if log.exists():
        logtext=log.read_text(encoding='utf-8',errors='replace')
        warning_lines=[s for s in logtext.splitlines() if any(t in s for t in ['Overfull','undefined','Missing character','multiply defined'])]
        if warning_lines:errors.extend(warning_lines)
    result={'status':'passed' if not errors else 'failed','chapters':chapter_count,
            'bibliography_items':len(bibliography),'cited_items':len(cited),'labels':len(labels),
            'figures_used':len(figures),'generated_tables':len(list((HERE/'tables').glob('*.tex'))),
            'gap_ids':sorted(gaps),'source_hashes_verified':len(manifest['source_sha256']),
            'historical_code_snapshots_verified':historical_verified,
            'live_code_differs_from_preserved_reference':live_code_differences,
            'original_reference_hashes_verified':reference_count,
            'numeric_source_rows':len(metrics),'log_issues':warning_lines,'errors':errors}
    if args.pdf:
        try:
            import fitz
            if not hasattr(fitz,'open'):raise ImportError('fitz is incomplete')
        except ImportError:
            sys.path.insert(0,str(Path(os.environ.get('TEMP','.'))/'codex_thesis_pdf_tools'))
            import fitz
        pdf=HERE/'build/main.pdf'
        doc=fitz.open(pdf)
        texts=[page.get_text() for page in doc]
        empty=[i+1 for i,s in enumerate(texts) if len(s.strip())<15]
        replacement=sum(s.count('\ufffd') for s in texts)
        out_of_page=[]
        for i,page in enumerate(doc):
            for b in page.get_text('blocks'):
                if b[0]<-1 or b[1]<-1 or b[2]>page.rect.width+1 or b[3]>page.rect.height+1:
                    out_of_page.append(i+1)
        result['pdf']={'pages':len(doc),'bytes':pdf.stat().st_size,'sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
                       'empty_pages':empty,'replacement_characters':replacement,'out_of_page_text':sorted(set(out_of_page)),
                       'text_characters':sum(map(len,texts))}
        if empty or replacement or out_of_page:errors.append('PDF text extraction or page bounds failed')
        if '??' in '\n'.join(texts):errors.append('Unresolved reference marker in PDF')
        result['status']='passed' if not errors else 'failed'
        preview=args.preview_dir.resolve() if args.preview_dir else ROOT/'docs/thesis/2026-10-19_manuscript_audit/preview'
        preview.mkdir(parents=True,exist_ok=True)
        candidates=[0,1,2]
        candidates += [i for i,s in enumerate(texts) if any(needle in re.sub(r'\s+','',s) for needle in ['1.3本研究の目的','3.5アンサンブル','4.1単体モデル','5.6特徴利用','6.1本研究で得られた','表B.1'])]
        candidates += [len(doc)-1]
        candidates=sorted(set(candidates))
        # Render all pages as small thumbnails so pagination can be reviewed.
        from PIL import Image,ImageDraw
        thumbs=[]
        for i,page in enumerate(doc):
            pix=page.get_pixmap(matrix=fitz.Matrix(.35,.35),alpha=False)
            im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
            tile=Image.new('RGB',(230,325),'#dedede')
            im.thumbnail((218,296));tile.paste(im,((230-im.width)//2,20))
            ImageDraw.Draw(tile).text((8,4),f'PDF {i+1}',fill='black')
            thumbs.append(tile)
            if i in candidates:page.get_pixmap(matrix=fitz.Matrix(1.2,1.2),alpha=False).save(preview/f'page_{i+1:03d}.png')
        for start in range(0,len(thumbs),24):
            group=thumbs[start:start+24];sheet=Image.new('RGB',(230*6,325*((len(group)+5)//6)),'white')
            for j,tile in enumerate(group):sheet.paste(tile,((j%6)*230,(j//6)*325))
            sheet.save(preview/f'contact_sheet_{start//24+1}.jpg',quality=88)
        result['pdf']['preview_pages']=[i+1 for i in candidates]
    (HERE/'build').mkdir(exist_ok=True)
    (HERE/'build/validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return bool(errors)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())

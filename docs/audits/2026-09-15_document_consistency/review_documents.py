"""Inventory and text-audit research documents without reading active run outputs.

Extracts Markdown, PDF, DOCX and PPTX. Keeps dated before/after manifests;
never changes originals. Text extraction is not a visual review of all pages.
"""
import argparse
from collections import Counter
from datetime import datetime
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).parent
PATTERNS={
 'old_execution':r'3\s*kHz.{0,30}(進行中|実行中|揃えば)|6条件|56/105|残り49|70条件|22\s*kHz.{0,15}(主材料|主結果)',
 'old_claim':r'RF.{0,15}(最強|主軸)|全モデル.{0,15}2.?5|ノイズ耐性.{0,12}(ではない|未実装)|原因.{0,10}(確定|解決)|未実装|未着手|主因',
 'method':r'Guided|IG.{0,20}(整合|不一致)|prediction_max|inner_holdout|KFold|AUC|閾値',
 'next':r'次に|優先|P0|P1|P2|最優先|今後|結論|目標',
}


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--stage',choices=['before','after','after_final'],required=True)
 args=ap.parse_args()
 target=OUT/args.stage
 if target.exists():raise SystemExit('Preserve existing audit stages.')
 target.mkdir(parents=True)
 sys.path.insert(0,str(Path(os.environ['TEMP'])/'codex_pdf_reader'))
 import fitz
 cmd=['rg','--files','--hidden','-g','*.md','-g','*.docx','-g','*.pptx','-g','*.pdf','-g','*.doc','-g','*.ppt',
      '-g','!Pool_boiling/**','-g','!.git/**','-g','!logs/**','-g','!**/__pycache__/**',
      '-g','!docs/audits/2026-09-15_document_consistency/**']
 names=subprocess.check_output(cmd,cwd=ROOT).decode('utf-8').splitlines()
 rows=[];texts=[];hits=[];errors=[]
 for name in sorted(names):
  p=ROOT/name
  data=p.read_bytes();ext=p.suffix.lower();parts=[];status='extracted';pages=0
  try:
   if ext=='.md':parts=[data.decode('utf-8-sig')]
   elif ext in ['.docx','.pptx']:
    with ZipFile(p) as z:
     members=([n for n in z.namelist() if n in ['word/document.xml','word/footnotes.xml','word/endnotes.xml']]
              if ext=='.docx' else sorted([n for n in z.namelist() if re.fullmatch(r'ppt/(slides/slide|notesSlides/notesSlide)\d+\.xml',n)],key=lambda n:(n.startswith('ppt/notes'),int(re.search(r'(\d+)\.xml',n).group(1)))))
     for n in members:
      tree=ET.fromstring(z.read(n))
      parts.append(n+'\n'+'\n'.join(t.text for t in tree.iter() if t.tag.endswith('}t') and t.text))
     pages=len(members)
   elif ext=='.pdf':
    with fitz.open(stream=data,filetype='pdf') as doc:
     pages=len(doc)
     parts=[f'PDF page {i+1}\n'+page.get_text() for i,page in enumerate(doc)]
     if sum(len(part.split('\n',1)[-1].strip()) for part in parts)<100:status='image_or_sparse_text'
   else:status='legacy_binary_not_extracted'
  except Exception as exc:status='extraction_error';errors.append({'path':name,'error':repr(exc)})
  text='\n\n'.join(parts).replace('\r\n','\n')
  path=p.relative_to(ROOT).as_posix()
  if ext=='.md':
   role=('template' if path.startswith(('templates/','experiments/_template/')) else
         'historical' if path.startswith(('archive/','trush_box/','研究進捗報告/')) or
             (path.startswith('experiments/') and path!='experiments/README.md') or
             (re.search(r'docs/(research_plan|progress)/20\d\d-',path) and not any(s in path for s in ['2026-09-18','2026-09-15_master'])) else 'current_or_index')
  else:role='original_or_generated_artifact'
  row=dict(path=path,format=ext,role=role,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
           status=status,text_characters=len(text),pages_or_xml_parts=pages,
           title=next((line for line in text.splitlines() if line.strip()),''))
  rows.append(row);texts.append(dict(path=path,text=text))
  for i,line in enumerate(text.splitlines(),1):
   tags=[tag for tag,pattern in PATTERNS.items() if re.search(pattern,line,re.I)]
   if tags:hits.append(dict(path=path,line=i,tags=';'.join(tags),text=line[:900]))
  if len(rows)%50==0:print(f'Extracted {len(rows)}/{len(names)}',flush=True)
 for name,data in [('document_catalog',rows),('review_candidates',hits)]:
  with (target/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 with (target/'extracted_text.jsonl').open('w',encoding='utf-8') as f:
  for t in texts:f.write(json.dumps(t,ensure_ascii=False)+'\n')
 summary=dict(created_at=datetime.now().astimezone().isoformat(),counts_by_format=dict(Counter(r['format'] for r in rows)),
              counts_by_status=dict(Counter(r['status'] for r in rows)),total_documents=len(rows),
              total_extracted_characters=sum(r['text_characters'] for r in rows),errors=errors,
              exclusions=['Pool_boiling: generated datasets and run outputs, including the ongoing run excluded by user',
                          '.git, logs, caches, this audit output'],
              inspection='All enumerated text extracted and scanned; selected original pages and critical claims reviewed semantically. Not a full visual rereading of every historical figure.')
 (target/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

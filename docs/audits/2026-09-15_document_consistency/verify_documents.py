# -*- coding: utf-8 -*-
"""Verify document revisions, preserved originals, and local Markdown links."""
import csv
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).parent


def main():
    before = list(csv.DictReader((AUDIT/'before/document_catalog.csv').open(encoding='utf-8-sig')))
    reasons = {r['path']: r['reason'] for r in json.loads((AUDIT/'changes.json').read_text(encoding='utf-8'))}
    rows = []
    preserved = []
    for old in before:
        p = ROOT/old['path']
        digest = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        changed = digest != old['sha256']
        if old['format'] != '.md':
            preserved.append(not changed)
        reason = reasons.get(old['path'], 'Progress index synchronized with the completed document review' if changed else
            'Original artifact retained; current applicability is explained in original_document_guide.md' if old['format'] != '.md' else
            'Retained: dated evidence/history or consistent current reference; current status is defined by research_status.md')
        rows.append(dict(path=old['path'],format=old['format'],decision='updated' if changed else 'retained',
                         reason=reason,before_sha256=old['sha256'],after_sha256=digest))
    with (AUDIT/'document_disposition.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    final_catalog=list(csv.DictReader((AUDIT/'after_final/document_catalog.csv').open(encoding='utf-8-sig')))
    final_hashes_match=all(hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in final_catalog)
    old_texts={r['path']:r['text'] for r in map(json.loads,(AUDIT/'before/extracted_text.jsonl').read_text(encoding='utf-8').splitlines())}
    historical_checks=[]
    for name,reason in reasons.items():
        if reason.startswith('Historical scope'):
            previous=old_texts[name].split('\n',1)[1].strip()
            historical_checks.append(previous in (ROOT/name).read_text(encoding='utf-8'))
    metrics=list(csv.DictReader((ROOT/'experiments/2026-09-15_onb_frequency_comparison/comparison/wav_metrics_both_frequencies.csv').open(encoding='utf-8-sig')))
    lookup={(r['maxfreq_khz'],r['snr'],r['model_key']):r for r in metrics}
    expected={('3','no_noise','cnntf_v2_gap'):0.9558,('3','-20','cnntf_v2_gap'):0.9194,
              ('3','-20','ensemble__simple_equal'):0.9227,('3','-20','ensemble__inner_holdout'):0.9234,
              ('22','no_noise','cnntf_v2_gap'):0.9623,('22','-20','cnntf_v2_gap'):0.9300,
              ('22','-20','ensemble__simple_equal'):0.9306,('22','-20','ensemble__inner_holdout'):0.9283}
    metric_checks=[round(float(lookup[k]['r2']),4)==v for k,v in expected.items()]
    snrs=sorted(set(r['snr'] for r in metrics))
    for band,wins,misses in [('3',[4,5],31),('22',[4,2],32)]:
        subset=[r for r in metrics if r['maxfreq_khz']==band]
        metric_checks.append(sum(int(r['false_negatives']) for r in subset)==misses)
        for strategy,win in zip(['ensemble__simple_equal','ensemble__inner_holdout'],wins):
            actual=sum(float(lookup[band,s,strategy]['r2'])>max(float(lookup[band,s,m]['r2']) for m in ['rf','cnntf_v2_gap','alexnet']) for s in snrs)
            metric_checks.append(actual==win)
    metric_checks += [len(metrics)==70,all(float(r['roc_auc_cont'])==1 and int(r['false_positives'])==0 for r in metrics)]
    missing=[];count=0;nonlink_notation=[]
    md_paths=[ROOT/r['path'] for r in before if r['format']=='.md']
    md_paths += [ROOT/'docs/analysis_workflow.md',ROOT/'docs/original_document_guide.md',AUDIT/'README.md']
    for p in md_paths:
        content=p.read_text(encoding='utf-8-sig')
        content=re.sub(r'```.*?```','',content,flags=re.S)
        # Balanced parentheses support file names and URLs containing parentheses.
        for match in re.finditer(r'\]\(',content):
            start=match.end();end=start;depth=1
            while end<len(content) and depth:
                if content[end]=='(':depth+=1
                elif content[end]==')':depth-=1
                end+=1
            if depth:continue
            target=content[start:end-1].strip()
            # A mathematical interval followed by a Japanese parenthesis is not an intended file link.
            if content[:start].endswith('λ∈[0,1](') and target=='実験は1':
                nonlink_notation.append(dict(document=p.relative_to(ROOT).as_posix(),notation='λ∈[0,1](実験は1)'))
                continue
            if target.startswith('<'):target=target[1:target.find('>')]
            target=target.split(' "',1)[0]
            if not target or re.match(r'^[a-zA-Z][\w+.-]*:',target) or target.startswith('#'):continue
            target=unquote(target.split('#',1)[0])
            if not target:continue
            count+=1
            resolved=(ROOT/target.lstrip('/')) if target.startswith('/') else p.parent/target
            if not resolved.exists():
                missing.append(dict(document=p.relative_to(ROOT).as_posix(),target=target,
                                    line=content[:match.start()].count('\n')+1))
    summary=dict(original_document_count=len(before),updated_original_markdown=sum(r['decision']=='updated' for r in rows),
                 preserved_binary_originals=sum(preserved),binary_original_count=len(preserved),all_binary_originals_unchanged=all(preserved),
                 new_current_guides=['docs/analysis_workflow.md','docs/original_document_guide.md'],
                 final_catalog_hashes_match=final_hashes_match,
                 historical_note_documents=len(historical_checks),historical_bodies_preserved=all(historical_checks),
                 summary_numeric_checks=len(metric_checks),summary_numeric_checks_passed=all(metric_checks),
                 checked_markdown_files=len(md_paths),checked_local_link_targets=count,missing_local_link_targets=missing,
                 excluded_mathematical_notation=nonlink_notation,
                 scope='Local file existence only; external URLs and anchor validity are not tested. Historical reference artifacts are retained.',
                 model_training_started=False,excluded_active_run_inspected=False)
    (AUDIT/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

"""Inspect the user's newest revised deck and render a read-only snapshot."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter
from io import BytesIO
import hashlib
import json
import posixpath
import sys
import xml.etree.ElementTree as ET
import zipfile

sys.stdout.reconfigure(encoding="utf-8")
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
NS={"p":"http://schemas.openxmlformats.org/presentationml/2006/main",
    "a":"http://schemas.openxmlformats.org/drawingml/2006/main",
    "r":"http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(source):
    pages=[]
    with zipfile.ZipFile(source) as z:
        presentation=ET.fromstring(z.read("ppt/presentation.xml"))
        size=presentation.find("p:sldSz",NS)
        rels={e.get("Id"):e.get("Target") for e in ET.fromstring(z.read("ppt/_rels/presentation.xml.rels"))}
        for ordinal,item in enumerate(presentation.findall("p:sldIdLst/p:sldId",NS),1):
            target=rels[item.get("{"+NS["r"]+"}id")]
            part=posixpath.normpath(posixpath.join("ppt",target))
            doc=ET.fromstring(z.read(part));shapes=[]
            for elem in doc.findall(".//p:sp",NS):
                texts=["".join(t.text or "" for t in p.findall(".//a:t",NS)) for p in elem.findall("p:txBody/a:p",NS)]
                if not any(texts):continue
                sizes=[int(e.get("sz"))/100 for e in elem.findall(".//a:rPr",NS) if e.get("sz")]
                tr=elem.find("p:spPr/a:xfrm",NS);rect=None
                if tr is not None:
                    off=tr.find("a:off",NS);ext=tr.find("a:ext",NS)
                    if off is not None and ext is not None:
                        rect=[int(off.get(k))/360000 for k in ("x","y")]+[int(ext.get(k))/360000 for k in ("cx","cy")]
                shapes.append(dict(text=texts,font_sizes=sizes,rect_cm=rect))
            tables=[]
            for table in doc.findall(".//a:tbl",NS):
                tables.append([["\n".join("".join(t.text or "" for t in p.findall(".//a:t",NS)) for p in c.findall("a:txBody/a:p",NS))
                                for c in row.findall("a:tc",NS)] for row in table.findall("a:tr",NS)])
            slide_rels={}
            rp=posixpath.join(posixpath.dirname(part),"_rels",posixpath.basename(part)+".rels")
            if rp in z.namelist():slide_rels={e.get("Id"):e for e in ET.fromstring(z.read(rp))}
            images=[]
            for pic in doc.findall(".//p:pic",NS):
                blip=pic.find("p:blipFill/a:blip",NS)
                if blip is None:continue
                rid=blip.get("{"+NS["r"]+"}embed")
                if rid not in slide_rels:continue
                media=posixpath.normpath(posixpath.join(posixpath.dirname(part),slide_rels[rid].get("Target")))
                raw=z.read(media)
                name=f"slide_{ordinal:02d}_{len(images)+1:02d}"+Path(media).suffix
                (OUT/"embedded_media"/name).write_bytes(raw)
                images.append(dict(part=media,extracted=name,sha256=hashlib.sha256(raw).hexdigest()))
            notes=[]
            for rel in slide_rels.values():
                if rel.get("Type","").endswith("/notesSlide"):
                    np=posixpath.normpath(posixpath.join(posixpath.dirname(part),rel.get("Target")))
                    ndoc=ET.fromstring(z.read(np))
                    notes=["".join(t.text or "" for t in p.findall(".//a:t",NS)) for p in ndoc.findall(".//a:p",NS)]
            pages.append(dict(page=ordinal,part=part,hidden=doc.get("show")=="0",shapes=shapes,tables=tables,images=images,notes=notes))
        return dict(source=str(source.relative_to(ROOT)),sha256=sha(source),
                    canvas_cm=[int(size.get(k))/360000 for k in ("cx","cy")],slides=pages)


def main():
    files=list(ROOT.glob("*/2026/1016*/*.pptx"))
    revised=[p for p in files if "修正版" in p.stem and not p.name.startswith("~$")]
    assert revised,"No revised source deck found"
    source=max(revised,key=lambda p:p.stat().st_mtime)
    before=sha(source)
    (OUT/"embedded_media").mkdir(parents=True,exist_ok=True)
    (OUT/"rendered_slides").mkdir(exist_ok=True)
    info=extract(source)
    (OUT/"deck_evidence.json").write_text(json.dumps(info,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# 作成済み修正版の本文・ノート抽出","",f"対象：{source.name}",""]
    for page in info["slides"]:
        lines += [f"## {page['page']}枚目（非表示={page['hidden']}）",""]
        for shape in page["shapes"]:lines+=shape["text"]
        for table in page["tables"]:lines += [" | ".join(row) for row in table]
        lines += ["", "ノート："+" / ".join(page["notes"]), ""]
    (OUT/"deck_text.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(dict(source=info["source"],slides=len(info["slides"]),canvas_cm=info["canvas_cm"]),ensure_ascii=False),flush=True)
    # A separate snapshot prevents closing or changing a presentation already open in the user's Office session.
    snapshot=OUT/"source_readonly_snapshot.pptx"
    snapshot.write_bytes(source.read_bytes())
    import pythoncom
    import win32com.client
    pythoncom.CoInitialize()
    app=None;pres=None;created=False
    render_status="not_started"
    try:
        try:app=win32com.client.GetActiveObject("PowerPoint.Application")
        except Exception:
            app=win32com.client.DispatchEx("PowerPoint.Application");created=True
        pres=app.Presentations.Open(str(snapshot),-1,0,0)
        for i in range(1,pres.Slides.Count+1):
            pres.Slides(i).Export(str(OUT/"rendered_slides"/f"slide_{i:02d}.png"),"PNG",1920,1080)
            print(f"Rendered {i}/{pres.Slides.Count}",flush=True)
        render_status="completed"
    finally:
        if pres is not None:pres.Close()
        if app is not None and created and app.Presentations.Count==0:app.Quit()
        pythoncom.CoUninitialize()
        record=dict(checked_at_jst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
            source=info["source"],source_sha256=before,source_unchanged=sha(source)==before,
            snapshot_matches_source=sha(snapshot)==before,slides=len(info["slides"]),render_status=render_status,
            original_pptx_edited=False,new_training=0,new_inference=0)
        (OUT/"verification.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__=="__main__":main()

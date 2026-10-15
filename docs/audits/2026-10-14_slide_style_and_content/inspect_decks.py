"""Read two original PPTX files and record text, layout and media evidence."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import statistics
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def extract(path, key):
    raw = path.read_bytes()
    with zipfile.ZipFile(path) as z:
        presentation = ET.fromstring(z.read("ppt/presentation.xml"))
        size = presentation.find("p:sldSz", NS)
        width, height = int(size.get("cx")), int(size.get("cy"))
        names = sorted([n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)],
                       key=lambda n: int(re.search(r"slide(\d+)", n).group(1)))
        pages = []
        for name in names:
            slide = ET.fromstring(z.read(name))
            index = int(re.search(r"slide(\d+)", name).group(1))
            shapes = []
            for element in slide.findall(".//p:sp", NS):
                paragraphs = []
                runs = []
                for paragraph in element.findall("p:txBody/a:p", NS):
                    text = "".join(t.text or "" for t in paragraph.findall(".//a:t", NS))
                    if text.strip():
                        paragraphs.append(text)
                    for run in paragraph.findall("a:r", NS):
                        props = run.find("a:rPr", NS)
                        runs.append({"text": "".join(t.text or "" for t in run.findall("a:t", NS)),
                                     "size_pt": None if props is None or props.get("sz") is None else int(props.get("sz"))/100,
                                     "bold": None if props is None else props.get("b"),
                                     "latin": None if props is None or props.find("a:latin", NS) is None else props.find("a:latin", NS).get("typeface"),
                                     "east_asian": None if props is None or props.find("a:ea", NS) is None else props.find("a:ea", NS).get("typeface")})
                transform = element.find("p:spPr/a:xfrm", NS)
                geometry = None
                if transform is not None:
                    off, ext = transform.find("a:off", NS), transform.find("a:ext", NS)
                    if off is not None and ext is not None:
                        geometry = {"x": int(off.get("x"))/width, "y": int(off.get("y"))/height,
                                    "w": int(ext.get("cx"))/width, "h": int(ext.get("cy"))/height}
                if paragraphs:
                    shapes.append({"paragraphs": paragraphs, "runs": runs, "geometry": geometry})
            media = []
            rel_name = f"ppt/slides/_rels/slide{index}.xml.rels"
            rels = {}
            if rel_name in z.namelist():
                for rel in ET.fromstring(z.read(rel_name)):
                    rels[rel.get("Id")] = rel.get("Target")
            for pic in slide.findall(".//p:pic", NS):
                blip = pic.find("p:blipFill/a:blip", NS)
                if blip is None:
                    continue
                rid = blip.get("{"+NS["r"]+"}embed")
                target = rels.get(rid, "")
                if not target:
                    continue
                media_name = "ppt/"+target.removeprefix("../")
                transform = pic.find("p:spPr/a:xfrm", NS)
                area, geometry = None, None
                if transform is not None:
                    off, ext = transform.find("a:off", NS), transform.find("a:ext", NS)
                    if off is not None and ext is not None:
                        geometry = {"x": int(off.get("x"))/width, "y": int(off.get("y"))/height,
                                    "w": int(ext.get("cx"))/width, "h": int(ext.get("cy"))/height}
                        area = geometry["w"]*geometry["h"]
                media.append({"part": media_name, "geometry": geometry, "area_fraction": area,
                              "sha256": hashlib.sha256(z.read(media_name)).hexdigest()})
            texts = [p for shape in shapes for p in shape["paragraphs"]]
            table_texts = [t.text or "" for t in slide.findall(".//a:tbl//a:t", NS)]
            sizes = [r["size_pt"] for shape in shapes for r in shape["runs"] if r["size_pt"]]
            pages.append({"slide": index, "texts": texts, "table_texts": table_texts,
                          "text_characters": sum(len(t) for t in texts)+sum(len(t) for t in table_texts),
                          "paragraphs": len(texts), "pictures": len(media),
                          "picture_area_sum": sum(m["area_fraction"] or 0 for m in media),
                          "font_sizes_pt": sorted(set(sizes)), "shapes": shapes, "media": media,
                          "xml_sha256": hashlib.sha256(z.read(name)).hexdigest()})
        return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest(),
                "width_emu": width, "height_emu": height, "slides": pages,
                "summary": {"slide_count": len(pages),
                            "text_chars_median": statistics.median(p["text_characters"] for p in pages),
                            "text_chars_range": [min(p["text_characters"] for p in pages), max(p["text_characters"] for p in pages)],
                            "pictures_by_slide": [p["pictures"] for p in pages],
                            "font_sizes_frequency": dict(Counter(r["size_pt"] for p in pages for s in p["shapes"] for r in s["runs"] if r["size_pt"]))}}


def main():
    decks = {}
    for key, suffix in [("copied_1016", "_1016.pptx"), ("reference_0918", "_0918.pptx")]:
        matches = [p for p in (ROOT/"研究進捗報告/2026").rglob("*.pptx") if p.name.endswith(suffix)]
        assert len(matches) == 1
        decks[key] = extract(matches[0], key)
    copied, original = decks["copied_1016"]["slides"], decks["reference_0918"]["slides"]
    assert len(copied) == len(original)
    differing = [a["slide"] for a, b in zip(copied, original) if a["xml_sha256"] != b["xml_sha256"]]
    comparison = {"differing_slide_xml": differing,
                  "text_difference_slides": [a["slide"] for a,b in zip(copied, original) if a["texts"] != b["texts"]],
                  "matching_slide_media": [a["slide"] for a,b in zip(copied, original) if a["media"] == b["media"]]}
    payload = {"decks": decks, "comparison": comparison, "source_files_modified": False}
    (OUT/"deck_style_evidence.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    text_lines = []
    for page in copied:
        text_lines.extend([f"## Slide {page['slide']} / chars={page['text_characters']} / paragraphs={page['paragraphs']} / pictures={page['pictures']} / image_area_sum={page['picture_area_sum']:.2f}", *page["texts"], *page["table_texts"], ""])
    (OUT/"copied_deck_text.md").write_text("\n".join(text_lines), encoding="utf-8")
    print(json.dumps({"summaries": {k:v["summary"] for k,v in decks.items()}, "comparison": comparison}, ensure_ascii=True))
    for page in copied:
        print(page["slide"], page["text_characters"], page["pictures"], "/".join(page["texts"][:3]).encode("unicode_escape").decode())
    # Review original figure assets; this sheet is not a PowerPoint rendering.
    from PIL import Image, ImageDraw, ImageFont
    from io import BytesIO
    selected = [(page["slide"], media) for page in copied if page["slide"] in [4,5,6,7]
                for media in page["media"] if (media["area_fraction"] or 0) > .025]
    sheet = Image.new("RGB", (1600, 460*((len(selected)+1)//2)), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
    with zipfile.ZipFile(ROOT/decks["copied_1016"]["path"]) as z:
        for i, (number, media) in enumerate(selected):
            pic = Image.open(BytesIO(z.read(media["part"]))).convert("RGB")
            pic.thumbnail((770, 410))
            x, y = (i%2)*800+15, (i//2)*460+42
            sheet.paste(pic, (x,y))
            draw.text((x,y-31), f"Slide {number}: {media['part']}", font=font, fill="black")
    sheet.save(OUT/"original_figure_assets_contact_sheet.png")


if __name__ == "__main__":
    main()

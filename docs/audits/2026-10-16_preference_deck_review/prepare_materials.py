"""Map plots into the current 15-page deck and update the original plan-table style."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Rectangle, FancyArrow
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
ASSETS=ROOT/"docs/progress/assets/2026-10-16"
FONT="C:/Windows/Fonts/YuGothM.ttc"
fontManager.addfont(FONT)
PROP=FontProperties(fname=FONT)
plt.rcParams.update({"font.family":PROP.get_name(),"svg.fonttype":"none"})
NS={"p":"http://schemas.openxmlformats.org/presentationml/2006/main","a":"http://schemas.openxmlformats.org/drawingml/2006/main"}

PLAN_ROWS=[
    dict(group=0,text="主系列の学習／評価分割を整備",status="完了",when="10/16までに実施済み"),
    dict(group=0,text="音響選別の検討・採否を整理\n（現行：選別なし）",status="検討済み",when="10/16までに検討・不採用判断済み"),
    dict(group=0,text="固定モデルで全7雑音条件を評価・解析",status="完了",when="10/16までに実施済み"),
    dict(group=1,text="3・4・5モデルの性能と誤り補完を比較",status="解析済み",when="10/16までに主解析済み"),
    dict(group=1,text="改善条件・限界・採用理由を\n修論へ整理・記述",status="現在以降",when="10月後半〜11月"),
    dict(group=1,text="【条件付き】入力に応じた\n統合比率の自動変更を検討",status="候補",when="時期未定・時間とコストに応じて判断"),
    dict(group=2,text="可視化・共通入力加工・\n誤りとの接続分析を実施",status="実施済み",when="10/16までに実施済み"),
    dict(group=2,text="説明性の意味と物理的に言える範囲を\n考察・記述",status="現在以降",when="10月後半〜11月"),
    dict(group=3,text="背景・実験方法・章別の本文整理",status="予定",when="10〜11月"),
    dict(group=3,text="結果・考察・結論の統合と推敲",status="予定",when="12月"),
    dict(group=3,text="修論本文の完成・発表資料作成・練習",status="予定",when="翌1〜2月"),
]
GROUPS=["① 評価設計・学習\nデータの整理","② アンサンブル手法\nの検証","③ 説明性を用いた\n要因分析・検証","④ 修論執筆・\n発表準備"]


def label(ax,x,y,value,size=18,ha="left",color="#14282c",va="center"):
    return ax.text(x,y,value,fontproperties=PROP,fontsize=size,ha=ha,va=va,color=color,linespacing=1.25)


def plan_table():
    fig,ax=plt.subplots(figsize=(19.4,8.8),facecolor="white")
    ax.set_xlim(0,19.4);ax.set_ylim(8.8,0);ax.axis("off")
    x0=.05;left=3.85;task=6.0;month=1.56;timeline=x0+left+task;right=timeline+6*month
    top=.35;header=.62;row_h=.635
    for x,w,name in [(x0,left,"大項目"),(x0+left,task,"小項目")]:
        ax.add_patch(Rectangle((x,top),w,header,facecolor="#72a5a7",edgecolor="white",lw=1))
        label(ax,x+w/2,top+header/2,name,22,ha="center",color="white")
    for index,name in enumerate(["9月","10月","11月","12月","1月","2月"]):
        x=timeline+index*month
        ax.add_patch(Rectangle((x,top),month,header,facecolor="#72a5a7",edgecolor="white",lw=1))
        label(ax,x+month/2,top+header/2,name,21,ha="center",color="white")
    spans=[(0,3),(3,3),(6,2),(8,3)]
    start_y=top+header
    current=timeline+1.5*month
    for group,(begin,count) in enumerate(spans):
        y=start_y+begin*row_h
        ax.add_patch(Rectangle((x0,y),left,count*row_h,facecolor="#9fc2c3",edgecolor="white",lw=1))
        label(ax,x0+.13,y+count*row_h/2,GROUPS[group],20)
    for index,row in enumerate(PLAN_ROWS):
        y=start_y+index*row_h;mid=y+row_h/2
        face="#e7eff0" if row["group"]%2==0 else "#f1f4f4"
        ax.add_patch(Rectangle((x0+left,y),task,row_h,facecolor=face,edgecolor="white",lw=.8))
        label(ax,x0+left+.10,mid,row["text"],17)
        for m in range(6):
            ax.add_patch(Rectangle((timeline+m*month,y),month,row_h,facecolor=face,edgecolor="white",lw=.8))
        if row["status"] in ["完了","検討済み","解析済み","実施済み"]:
            x=timeline+1.04*month;w=.42*month
            ax.add_patch(Rectangle((x,mid-.19),w,.38,facecolor="#3c9b85",edgecolor="none"))
            label(ax,x+w/2,mid,"済",17,ha="center",color="white")
        elif row["status"]=="候補":
            x=current+.10;w=right-x-.12
            ax.add_patch(Rectangle((x,mid-.23),w,.46,facecolor="white",edgecolor="#c58b45",lw=1.4,linestyle="--"))
            label(ax,x+w/2,mid,"時期未定・時間とコストに応じて判断",16,ha="center",color="#a66b2b")
        else:
            if index in [4,7,8]:start,end=current+.08,timeline+3*month-.10
            elif index==9:start,end=timeline+3*month+.12,timeline+4*month-.12
            else:start,end=timeline+4*month+.12,right-.12
            ax.add_patch(FancyArrow(start,mid,end-start,0,width=.16,head_width=.32,head_length=.22,
                                    length_includes_head=True,facecolor="#ffa146",edgecolor="none"))
    ax.plot([current,current],[top+header,start_y+11*row_h],color="#5d777a",lw=1.4,linestyle="--")
    label(ax,current,top-.09,"10/16",15,ha="center",color="#476368")
    label(ax,x0,start_y+11*row_h+.34,"緑＝10/16までに実施済み（実施期間は示さない）　橙＝現在以降の予定　点線枠＝条件付き候補",15)
    label(ax,x0,start_y+11*row_h+.65,"期間は既存計画に基づく見通し。2026年9〜12月／2027年1〜2月。",14,color="#647d80")
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    for suffix in ["png","svg"]:
        fig.savefig(ASSETS/f"semester_plan_table_updated.{suffix}",dpi=160,facecolor="white")
    plt.close(fig)
    (OUT/"semester_plan_rows.json").write_text(json.dumps(dict(groups=GROUPS,rows=PLAN_ROWS,
        formal_deadlines_verified=False,manuscript_completion_not_claimed=True),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (OUT/"semester_plan_rows.tsv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.writer(stream,delimiter="\t");writer.writerow(["行","大項目","小項目","状態","時期"])
        for index,row in enumerate(PLAN_ROWS,1):writer.writerow([index,GROUPS[row["group"]].replace("\n",""),row["text"].replace("\n",""),row["status"],row["when"]])


def main():
    deck=json.loads((OUT/"deck_evidence.json").read_text(encoding="utf-8"))
    old=json.loads((ROOT/"docs/audits/2026-10-15_created_deck_review/deck_evidence.json").read_text(encoding="utf-8"))
    pairs=[(8,6),(9,7),(12,9)]
    assert all(deck["slides"][now-1]["tables"]==old["slides"][prev-1]["tables"] for now,prev in pairs)
    assets=[
        (5,ROOT/"experiments/2026-10-14_error_feature_linkage/slide_old_noise_shortcut.png"),
        (8,ASSETS/"current_regression_noise_trend.png"),
        (9,ASSETS/"current_onb_false_negatives.png"),
        (11,ASSETS/"reconciled_preview/regional_mse_reduction.png"),
        (11,ASSETS/"reconciled_preview/error_group_contribution.png"),
        (12,ASSETS/"reconciled_preview/time_permutation_models.png"),
    ]
    slots={};insertion=[]
    with zipfile.ZipFile(OUT/"source_readonly_snapshot.pptx") as archive:
        for page in sorted({p for p,_ in assets}):
            doc=ET.fromstring(archive.read(deck["slides"][page-1]["part"]));frames=[]
            for shape in doc.findall(".//p:sp",NS):
                if "".join(t.text or "" for t in shape.findall(".//a:t",NS)).strip():continue
                dash=shape.find("p:spPr/a:ln/a:prstDash",NS);tr=shape.find("p:spPr/a:xfrm",NS)
                if dash is None or dash.get("val")!="dash" or tr is None:continue
                off=tr.find("a:off",NS);ext=tr.find("a:ext",NS)
                if off is None or ext is None:continue
                rect=[int(off.get(k))/360000 for k in ["x","y"]]+[int(ext.get(k))/360000 for k in ["cx","cy"]]
                if rect[2]>10 and rect[3]>5:frames.append(rect)
            slots[page]=sorted(frames,key=lambda r:r[0])
    for page,path in assets:
        rect=slots[page].pop(0);x,y,w,h=rect
        with Image.open(path) as im:ratio=im.width/im.height
        pw=min(w,h*ratio);ph=pw/ratio
        image_rect=[round(v,3) for v in [x+(w-pw)/2,y+(h-ph)/2,pw,ph]]
        insertion.append(dict(actual_page=page,file=str(path.relative_to(ROOT)).replace("\\","/"),
            frame_cm=rect,image_rect_cm=image_rect,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(insertion)==6 and all(not values for values in slots.values())
    (OUT/"figure_insertion_map.json").write_text(json.dumps(insertion,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (OUT/"figure_insertion_map.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["actual_page","file","left_cm","top_cm","width_cm","height_cm"])
        for item in insertion:writer.writerow([item["actual_page"],item["file"],*item["image_rect_cm"]])
    titles=["表紙","全体目次","第1章の位置","研究概要","雑音評価方法","条件・モデル","第2章の位置","回帰結果","ONB判定","第3章の位置","誤り補完","前回指摘への対応","第4章の位置","修論構成","後期計画"]
    fig,axes=plt.subplots(5,3,figsize=(18,18.8),facecolor="#edf1f4")
    for index,ax in enumerate(axes.flat,1):
        ax.imshow(plt.imread(OUT/"rendered_slides"/f"slide_{index:02d}.png"));ax.axis("off")
        ax.set_title(f"実{index:02d}：{titles[index-1]}",fontproperties=PROP,fontsize=12)
    fig.tight_layout(pad=1);fig.savefig(OUT/"actual_deck_contact_sheet.png",dpi=120);plt.close(fig)
    plan_table()
    source=ROOT/deck["source"]
    record=json.loads((OUT/"verification.json").read_text(encoding="utf-8"))
    record.update(actual_page_numbers_used=True,table_pairs_equal_to_verified_previous_version=pairs,
        figure_slots=6,figure_slot_actual_pages=[5,8,9,11,11,12],
        old_numbering_mixed=True,plan_table_rows=11,plan_table_groups=4,
        source_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==deck["sha256"],
        new_model_fits=0,new_input_interventions=0,new_plan_is_proposal=True,
        checked_at_jst=datetime.now(timezone(timedelta(hours=9))).isoformat())
    assert record["source_unchanged"]
    (OUT/"verification.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(figures=insertion,plan_rows=11,source_unchanged=record["source_unchanged"]),ensure_ascii=False,indent=2))


if __name__=="__main__":main()

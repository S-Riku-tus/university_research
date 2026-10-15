"""Prepare missing diagrams and an updated semester plan; preserve PPTX files."""
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties, fontManager

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
from utils.experiment.run_helpers import windows_long_path
OUT=ROOT/"docs/progress/assets/2026-10-16"
FONT="C:/Windows/Fonts/YuGothM.ttc"
fontManager.addfont(FONT)
PROP=FontProperties(fname=FONT)
plt.rcParams.update({"font.family":PROP.get_name(),"font.size":24,"svg.fonttype":"none"})
BLUE="#208bdd";PALE="#e7f3fc";GRAY="#526779";GREEN="#3b9b88";ORANGE="#c18435"


def label(ax,x,y,text,size=24,ha="center",va="center",color=GRAY):
    ax.text(x,y,text,fontsize=size,ha=ha,va=va,color=color,fontproperties=PROP)


def box(ax,x,y,w,h,text,size=24,face=PALE,edge=BLUE,dashed=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.008,rounding_size=.018",
                              facecolor=face,edgecolor=edge,linewidth=1.9,
                              linestyle="--" if dashed else "-"))
    label(ax,x+w/2,y+h/2,text,size)


def arrow(ax,start,end):
    ax.annotate("",xy=end,xytext=start,arrowprops={"arrowstyle":"->","lw":2.4,"color":GRAY})


def canvas(width=17,height=5.6):
    fig,ax=plt.subplots(figsize=(width,height));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    return fig,ax


def save(fig,name):
    fig.tight_layout(pad=.7)
    for ext in ["svg","png"]:fig.savefig(OUT/(name+"."+ext),dpi=180,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    fig,ax=canvas()
    names=["音響録音","スペクトロ\nグラム","複数モデル\n＋統合","熱流束の\n予測","ONB判定"]
    for i,text in enumerate(names):
        x=.015+i*.2;box(ax,x,.44,.165,.28,text,26)
        if i<4:arrow(ax,(x+.169,.58),(x+.194,.58))
    label(ax,.5,.17,"モデルごとの特徴利用・誤り方と，統合の効果を対応づける",26)
    save(fig,"overview_pipeline")
    fig,ax=canvas()
    box(ax,.025,.38,.24,.34,"無雑音のみで学習",25)
    arrow(ax,(.28,.55),(.36,.55))
    box(ax,.37,.31,.27,.48,"モデル・前処理\n統合重みを固定",26)
    arrow(ax,(.65,.55),(.72,.55))
    box(ax,.735,.19,.24,.73,"同じ評価chunk\nclean / 0 / −4 / −8\n−12 / −16 / −20 dB",22)
    label(ax,.47,.10,"雑音条件ごとに学習し直さず，同じfitを使う",25)
    save(fig,"fixed_clean_evaluation")
    fig,ax=canvas(18,7.0)
    label(ax,.24,.91,"各録音の未使用区間を評価",27)
    label(ax,.24,.73,"1録音 = 60 chunk（1秒ずつ）",24)
    box(ax,.025,.44,.30,.18,"45：学習",25)
    box(ax,.34,.44,.12,.18,"15：評価",23,face="#fff1df",edge=ORANGE)
    label(ax,.24,.25,"36録音：学習1620／評価540",25)
    label(ax,.74,.91,"同じ単体予測から3／4／5を構成",26)
    box(ax,.535,.61,.44,.18,"元3：RF・CNN＋Transformer・AlexNet",22)
    box(ax,.535,.36,.44,.18,"4：元3＋HGB ／ 元3＋ET",25)
    box(ax,.535,.11,.44,.18,"5：元3＋HGB＋ET",26)
    label(ax,.5,.01,"2025年6/11＋6/18，3 kHz・選別なし・clean_only",22)
    save(fig,"evaluation_and_models")

    # Use the completed run's existing tables; never read partial rerun metrics.
    source_metrics=ROOT/"experiments/2026-10-15_clean_only_result_analysis/pooled_metrics.csv"
    source_hash=hashlib.sha256(source_metrics.read_bytes()).hexdigest()
    metrics=pd.read_csv(source_metrics,dtype={"noise":str})
    keys=["ensemble__original3_mse","ensemble__original3_hgb","ensemble__original3_extra_trees","ensemble__original3_hgb_extra_trees"]
    names=["Original 3 MSE","3 + HGB","3 + ET","3 + HGB + ET"]
    colors=["#087db7","#c079a0","#16a080","#222222"]
    order=["clean","0","-4","-8","-12","-16","-20"]
    fig,ax=plt.subplots(figsize=(10.0,5.0))
    for key,name,color in zip(keys,names,colors):
        vals=[float(metrics.loc[metrics.noise.eq(n)&metrics.model_key.eq(key),"rmse_all_kW_m2"].iloc[0]) for n in order]
        ax.plot(range(7),vals,"o-",color=color,label=name,lw=2.6)
    ax.set_xticks(range(7),["Clean","0","−4","−8","−12","−16","−20"],fontsize=17)
    ax.set_xlabel("Reference SNR [dB]",fontsize=19);ax.set_ylabel("Overall RMSE [kW/m²]",fontsize=19)
    ax.tick_params(axis="y",labelsize=17);ax.legend(ncol=2,fontsize=14);ax.grid(axis="y",alpha=.16)
    ax.spines[["top","right"]].set_visible(False)
    save(fig,"current_regression_noise_trend")
    fig,ax=plt.subplots(figsize=(9.3,5.0))
    for i,noise in enumerate(["clean","-20"]):
        vals=[int(metrics.loc[metrics.noise.eq(noise)&metrics.model_key.eq(k),"fn"].iloc[0]) for k in keys]
        bars=ax.bar(np.arange(4)+(i-.5)*.34,vals,width=.34,color=["#9dcfdf",BLUE][i],label="Clean" if i==0 else "−20 dB")
        ax.bar_label(bars,padding=3,fontsize=19)
    ax.set_xticks(range(4),["Original 3 MSE","3 + HGB","3 + ET","5 models"],fontsize=16)
    ax.set_ylabel("False negatives / 315 positive chunks",fontsize=18);ax.tick_params(axis="y",labelsize=17)
    ax.legend(fontsize=17);ax.set_ylim(0,80);ax.spines[["top","right"]].set_visible(False)
    save(fig,"current_onb_false_negatives")
    assert hashlib.sha256(source_metrics.read_bytes()).hexdigest()==source_hash

    # Calendar position is an approximate planning interval, not a formal deadline.
    fig,ax=plt.subplots(figsize=(20,8.7))
    ax.set_xlim(0,6);ax.set_ylim(-.75,5.3);ax.invert_yaxis()
    ax.set_xticks([.5,1.5,2.5,3.5,4.5,5.5],["9月","10月","11月","12月","1月","2月"],fontsize=25)
    ax.xaxis.tick_top();ax.tick_params(axis="both",length=0,pad=16)
    ax.set_yticks([0,1,2,3,4],["① 評価条件・データ","② アンサンブル手法の検証","② 条件付きの追加候補","③ 説明性・要因分析","④ 修論・発表準備"],fontsize=23)
    for spine in ax.spines.values():spine.set_visible(False)
    for x in range(7):ax.axvline(x,color="#dce3e9",lw=1,zorder=0)
    ax.axvline(1.5,color=ORANGE,ls="--",lw=2)
    ax.text(1.53,-.57,"10/16",fontsize=20,color=ORANGE)
    def bar(y,start,end,text,color=BLUE,dashed=False):
        ax.add_patch(FancyBboxPatch((start,y-.30),end-start,.60,boxstyle="round,pad=.015,rounding_size=.05",
                                   facecolor="white" if dashed else color,edgecolor=color,linewidth=2,
                                   linestyle="--" if dashed else "-"))
        ax.text((start+end)/2,y,text,ha="center",va="center",fontsize=21,color=color if dashed else "white",fontproperties=PROP)
    bar(0,.08,1.47,"主系列の条件\n整備・実行済",GREEN)
    bar(1,.55,1.47,"3→4→5\n解析済",GREEN)
    bar(1,1.58,3.0,"効果・採否理由を\n発表・修論へ整理")
    bar(2,1.60,5.90,"入力に応じた自動重み切替：時期未定・時間と費用に応じて判断",ORANGE,True)
    bar(3,.08,1.47,"可視化・加工対照\n接続分析済",GREEN)
    bar(3,1.58,3.0,"特徴利用と誤りの\n解釈・考察")
    bar(4,1.58,3.0,"目次・章別整理\n本文への展開")
    bar(4,3.08,3.95,"結果・考察\n結論の整理")
    bar(4,4.05,5.90,"本文完成・発表準備")
    ax.text(0,5.02,"緑：完了した成果　青：現在〜今後の整理　点線：実施を約束しない候補",fontsize=20,color=GRAY,fontproperties=PROP)
    ax.text(0,5.38,"時期は既存計画に基づく案。正式な提出期限・執筆完了を示すものではない。",fontsize=17,color=GRAY,fontproperties=PROP)
    fig.subplots_adjust(left=.28,right=.985,top=.88,bottom=.12)
    for ext in ["svg","png"]:fig.savefig(OUT/("semester_plan_progress."+ext),dpi=180,bbox_inches="tight",facecolor="white")
    plt.close(fig)

    evidence=json.loads((ROOT/"docs/audits/2026-10-14_slide_style_and_content/deck_style_evidence.json").read_text(encoding="utf-8"))
    source=ROOT/evidence["decks"]["copied_1016"]["path"]
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    exported=[]
    with zipfile.ZipFile(source) as z:
        for part,name in [("ppt/media/image5.png","mask_recap_0918.png"),
                          ("ppt/media/image6.png","ig_alexnet_recap_0918.png"),
                          ("ppt/media/image7.png","ig_transformer_recap_0918.png")]:
            content=z.read(part);(OUT/name).write_bytes(content)
            exported.append({"output":name,"source_part":part,"sha256":hashlib.sha256(content).hexdigest(),"operation":"verbatim extraction"})
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    bar_extracts=[]
    consistency=ROOT/"experiments/2026-10-15_presentation_readiness/run_consistency.json"
    if consistency.exists():
        checked=json.loads(consistency.read_text(encoding="utf-8"))
        if checked.get("all_results_numerically_equal") and checked.get("main_metric_comparison"):
            rerun=ROOT/checked["rerun"] / "maxfreq=3kHz"
            selected=[("heatflux_no_noise/bar/legacy4/r2_ep200_no_noise_zoom.png","current_bar_clean_legacy4.png"),
                      ("heatflux_no_noise/bar/addition4/r2_ep200_no_noise_zoom.png","current_bar_clean_addition4.png"),
                      ("heatflux_reference_SNR=-20/bar/addition4/r2_ep200_-20_zoom.png","current_bar_minus20_addition4.png"),
                      ("heatflux_reference_SNR=-20/bar/integration5/r2_ep200_-20_zoom.png","current_bar_minus20_integration5.png")]
            for relative,name in selected:
                path=rerun/relative
                content=Path(windows_long_path(path)).read_bytes()
                (OUT/name).write_bytes(content)
                bar_extracts.append({"output":name,"source":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(content).hexdigest(),
                                     "operation":"verbatim copy from completed equivalent rerun 194258"})
    record={"source_pptx":str(source.relative_to(ROOT)),"source_pptx_sha256":before,
            "source_unchanged":True,"recap_assets":exported,
            "generated_diagrams":["overview_pipeline","fixed_clean_evaluation","evaluation_and_models","semester_plan_progress"],
            "current_result_figures":["current_regression_noise_trend","current_onb_false_negatives"],
            "current_result_source":{"path":str(source_metrics.relative_to(ROOT)),"sha256":source_hash,"run":"143609"},
            "rerun_bar_extracts":bar_extracts,
            "new_research_results":False,"new_model_training":0,"new_model_inference":0,
            "semester_plan_is_proposal":True,"formal_deadlines_verified":False}
    (OUT/"asset_sources.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"diagram_pairs":4,"result_figure_pairs":2,"verbatim_recap_images":3,"completed_rerun_bar_copies":len(bar_extracts),"pptx_unchanged":True}))


if __name__=="__main__":main()

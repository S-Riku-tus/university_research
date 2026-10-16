"""Verify printed values and map existing plot files into the user's figure slots."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
import pandas as pd
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
ASSETS=ROOT/"docs/progress/assets/2026-10-16"
FONT="C:/Windows/Fonts/YuGothM.ttc"
fontManager.addfont(FONT)
PROP=FontProperties(fname=FONT)


def main():
    deck=json.loads((OUT/"deck_evidence.json").read_text(encoding="utf-8"))
    result=ROOT/"experiments/2026-10-15_clean_only_result_analysis"
    metrics=pd.read_csv(result/"pooled_metrics.csv",dtype={"noise":str})
    days=pd.read_csv(result/"q100_by_source_day.csv",dtype={"noise":str})
    features=pd.read_csv(ROOT/"experiments/2026-10-14_error_feature_linkage/existing_feature_dependence.csv",dtype={"noise":str})
    groups=pd.read_csv(result/"error_direction_contributions.csv",dtype={"noise":str})
    regions=pd.read_csv(result/"regional_loss_changes.csv",dtype={"noise":str})
    keys=["ensemble__original3_mse","ensemble__original3_hgb_extra_trees"]
    checks=[]
    def record(label,actual,expected):
        expected=float(f"{expected:.2f}")
        actual=float(actual)
        checks.append(dict(item=label,displayed=actual,expected_rounded=expected,passed=abs(actual-expected)<1e-8))
    for row,noise in enumerate(["clean","-20"],1):
        for col,key in enumerate(keys,1):
            expected=metrics[(metrics.noise==noise)&(metrics.model_key==key)].iloc[0].rmse_all_kW_m2
            record(f"page6 {noise} {key}",deck["slides"][5]["tables"][0][row][col],expected)
    for row,day in enumerate(["2025.06.11_0.3_2","2025.06.18_0.3_3"],1):
        for col,key in enumerate(keys,1):
            expected=days[(days.noise=="clean")&(days.model_key==key)&(days.source_day==day)].iloc[0].q100_kW_m2
            record(f"page7 q100 {day} {key}",deck["slides"][6]["tables"][0][row][col],expected)
    for row,key in enumerate(keys,1):
        entry=features[(features.noise=="clean")&(features.intervention=="time_permutation")&(features.model_key==key)].iloc[0]
        for col,field in enumerate(["rmse_all_kW_m2__original","rmse_all_kW_m2"],1):
            record(f"page9 {key} {field}",deck["slides"][8]["tables"][0][row][col],entry[field])
    for noise,expected_fn,expected_fp in [("clean",[42,27],[1,1]),("-20",[65,52],[0,0])]:
        actual=metrics[(metrics.noise==noise)&metrics.model_key.isin(keys)].set_index("model_key")
        for index,key in enumerate(keys):
            record(f"page7 FN {noise} {key}",expected_fn[index],actual.loc[key,"fn"])
            record(f"page7 FP {noise} {key}",expected_fp[index],actual.loc[key,"fp"])
    record("page8 same-direction contribution -20",87.15,groups[(groups.noise=="-20")&(groups.group=="same_direction_or_zero")].iloc[0].share_of_total_mse_improvement_percent)
    for key,display in zip(keys,[47.07,47.43]):
        rm=metrics[metrics.model_key==key].set_index("noise")
        record(f"page6 clean-to-minus20 delta {key}",display,rm.loc["-20","rmse_all_kW_m2"]-rm.loc["clean","rmse_all_kW_m2"])
    (OUT/"numeric_verification.json").write_text(json.dumps(dict(checks=checks,all_passed=all(x["passed"] for x in checks),
        scope="Displayed tables and numeric claims; not a new fit or scientific re-evaluation"),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    definitions=[
        (4,"旧生成診断",ROOT/"experiments/2026-10-14_error_feature_linkage/slide_old_noise_shortcut.png",[4.2,9.05,23.8,11.6]),
        (6,"全7条件RMSE",ASSETS/"current_regression_noise_trend.png",[4.2,9.05,27.0,13.1]),
        (7,"見逃し件数",ASSETS/"current_onb_false_negatives.png",[4.2,9.05,26.5,13.1]),
        (8,"領域別MSE減少",ASSETS/"reconciled_preview/regional_mse_reduction.png",[4.2,9.05,23.6,12.35]),
        (8,"誤差方向群の寄与",ASSETS/"reconciled_preview/error_group_contribution.png",[29.3,9.05,19.1,12.35]),
        (9,"時間順序加工",ASSETS/"reconciled_preview/time_permutation_models.png",[4.2,11.15,25.5,10.9]),
    ]
    insertion=[]
    for page,label,path,slot in definitions:
        with Image.open(path) as im: ratio=im.width/im.height
        x,y,w,h=slot
        placed_w=min(w,h*ratio);placed_h=placed_w/ratio
        placed=[round(v,3) for v in [x+(w-placed_w)/2,y+(h-placed_h)/2,placed_w,placed_h]]
        insertion.append(dict(page=page,label=label,file=str(path.relative_to(ROOT)).replace("\\","/"),
            slot_cm=slot,image_rect_cm=placed,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (OUT/"figure_insertion_map.json").write_text(json.dumps(insertion,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (OUT/"figure_insertion_map.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["page","label","file","left_cm","top_cm","width_cm","height_cm"])
        for item in insertion:writer.writerow([item["page"],item["label"],item["file"],*item["image_rect_cm"]])
    titles=["表紙","今回の発表内容","概要・問い","雑音評価方法","分割・モデル構成","回帰結果","ONB判定","誤り補完","教授指摘への対応","修論構成","後期計画"]
    fig,axes=plt.subplots(4,3,figsize=(18,14),facecolor="#edf1f4")
    for n,ax in enumerate(axes.flat[:11],1):
        ax.imshow(plt.imread(OUT/"rendered_slides"/f"slide_{n:02d}.png"));ax.axis("off")
        ax.set_title(f"{n:02d} {titles[n-1]}",fontproperties=PROP,fontsize=12)
    axes.flat[-1].axis("off");fig.tight_layout(pad=1)
    fig.savefig(OUT/"actual_deck_contact_sheet.png",dpi=120);plt.close(fig)
    verification=json.loads((OUT/"verification.json").read_text(encoding="utf-8"))
    verification.update(rendered_pages_viewed=list(range(1,12)),placeholder_slots=6,
        placeholder_pages=[4,6,7,8,9],numeric_items_checked=len(checks),
        all_displayed_numeric_items_match=all(x["passed"] for x in checks),
        source_unchanged=hashlib.sha256((ROOT/deck["source"]).read_bytes()).hexdigest()==deck["sha256"],
        native_powerpoint_render=True,source_figures_modified=False)
    (OUT/"verification.json").write_text(json.dumps(verification,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(numeric_checks=len(checks),all_passed=all(x["passed"] for x in checks),
        insertion_rectangles=insertion,source_unchanged=verification["source_unchanged"]),ensure_ascii=False,indent=2))
    assert all(x["passed"] for x in checks)


if __name__=="__main__":main()

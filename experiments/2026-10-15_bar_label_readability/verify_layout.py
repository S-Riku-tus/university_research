"""Render real saved bar metrics and check readable text bounds; no fitting."""
from pathlib import Path
import hashlib
import json
import sys
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"code"))
from utils.experiment.run_helpers import open_text
import utils.plotting.regression_plots as plots


def main():
    comparisons=pd.read_csv(ROOT/"experiments/2026-10-07_onb_200epoch_result_analysis/comparison_across_runs.csv")
    source=Path(comparisons.loc[comparisons.seed.eq(42)].iloc[0].source_result)/"metrics_summary_no_noise.csv"
    with open_text(source,"rb") as f:before=hashlib.sha256(f.read()).hexdigest()
    with open_text(source,"r",encoding="utf-8-sig") as f:data=pd.read_csv(f)
    records=[]
    original_save=plots._save_current_figure
    current_values=[]

    def check_and_save(path):
        fig=plt.gcf();ax=plt.gca()
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        labels=ax.get_xticklabels()
        assert all(t.get_fontsize()==ax.yaxis.label.get_fontsize()==23 for t in labels)
        assert all(t.get_rotation()==0 for t in labels)
        bounds=[t.get_window_extent(renderer) for t in labels]
        gaps=[right.x0-left.x1 for left,right in zip(bounds,bounds[1:])]
        assert not gaps or min(gaps)>=3, gaps
        assert all(b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height for b in bounds)
        heights=[rectangle.get_height() for rectangle in ax.patches]
        assert np.allclose(heights,current_values)
        original_save(path)
        records.append({"path":str(Path(path).relative_to(OUT)),"model_count":len(labels),
                        "font":plt.rcParams["font.family"],"tick_fontsize":23,"axis_fontsize":23,
                        "figure_size_inches":fig.get_size_inches().tolist(),
                        "minimum_label_gap_pixels":min(gaps) if gaps else None,
                        "all_labels_inside_figure":True,"values_unchanged":True,
                        "labels":[t.get_text() for t in labels]})
    for font,key in [("Times New Roman","times"),("DejaVu Sans","sans")]:
        for count in [5,11]:
            frame=data.iloc[:count]
            current_values=frame.r2_mean.to_numpy()
            with plt.rc_context({"font.family":font,"savefig.dpi":150}):
                with patch.object(plots,"_save_current_figure",side_effect=check_and_save):
                    plots.RegressionPlotter().plot_bar("R2 Score",frame.model.tolist(),current_values,
                        frame.r2_se.to_numpy(),200,OUT/f"{key}_{count}","no_noise")
    with open_text(source,"rb") as f:after=hashlib.sha256(f.read()).hexdigest()
    assert before==after
    verification={"status":"passed","source":str(source.relative_to(ROOT)),"source_sha256":before,
                  "source_unchanged":True,"new_fits":0,"new_inference":0,"renders":records,
                  "scope":"Typography preview using old saved R2 means/SE; not a new performance evaluation"}
    (OUT/"verification.json").write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":"passed","renders":len(records),"source_unchanged":True,
                      "minimum_label_gap":min(r["minimum_label_gap_pixels"] for r in records)}))


if __name__=="__main__":main()

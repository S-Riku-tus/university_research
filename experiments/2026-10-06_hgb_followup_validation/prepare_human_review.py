"""Make an editable review sheet and original-audio excerpts for selected chunks."""
import csv
from pathlib import Path
import re
import sys

import numpy as np
from scipy.io import wavfile

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"experiments/2026-10-06_chunk_complementary_model_pilots"))
import pilot as p


def main():
    rows,folds,data,outer,base,y,days=p.setup()
    metadata={p.identity(r):r for r in p.read_csv(data/p.NOISE_DIRS["clean"]/"chunk_manifest.csv")}
    selected=p.read_csv(OUT/"representative_chunks.csv")
    # The two remaining training bottlenecks are supplementary, not an adoption gate.
    for chunk in [28,36]:selected.append({"scope":"supplementary_q100_bottleneck","selection":"all_pilots_negative","noise":"clean","source_wav_id":"20250611::index=10.3.15E+05","chunk_index":chunk})
    predictions=OUT/"outer_predictions.csv"
    if predictions.exists():
        alarms=[r for r in p.read_csv(predictions) if r["seed"]=="43" and r["method"]=="hgb" and r["noise"]=="-20"
            and float(r["y_kW_m2"])<float(r["onb_kW_m2"])<=float(r["prediction_kW_m2"])]
        alarms.sort(key=lambda r:float(r["prediction_kW_m2"])-float(r["onb_kW_m2"]))
        picked=set()
        for selection,cells in [("smallest_alarm_margin",alarms[:2]),("largest_alarm_margin",alarms[-2:])]:
            for r in cells:
                if p.identity(r) in picked:continue
                picked.add(p.identity(r))
                selected.append({"scope":"seed43_minus20_hgb_false_positive","selection":selection,"noise":"-20","source_wav_id":r["source_wav_id"],
                    "chunk_index":r["chunk_index"],"diagnostic_seed":43,"hgb":r["prediction_kW_m2"]})
    annotations={}
    sheet=OUT/"human_review_sheet.csv"
    if sheet.exists():
        for row in p.read_csv(sheet):
            annotations[(row["scope"],row["evaluation_noise"],p.identity(row))]={key:row.get(key,"") for key in
                ["recording_anomaly_observed","synchronized_video_or_log_available","bubble_observation","notes"]}
    folder=OUT/"audio_review";folder.mkdir(exist_ok=True)
    records=[];cache={};clips={}
    experiment=data.parents[3]
    for item in selected:
        row=metadata[p.identity(item)]
        source=experiment/"録音データ_熱流束"/row["source_experiment_name"]/row["original_source_wav_name"]
        assert p.lp(source).exists(),source
        if source not in cache:cache[source]=wavfile.read(str(p.lp(source)),mmap=False)
        rate,signal=cache[source]
        start=float(row["chunk_start_seconds"]);duration=float(row["chunk_duration_seconds"])
        context_start=max(0,start-1);context_end=min(len(signal)/rate,start+duration+1)
        clip_id=f"{row['source_wav_id']}|{row['chunk_index']}"
        if clip_id not in clips:
            filename=re.sub(r'[^A-Za-z0-9_.-]','_',f"{row['source_wav_id']}_chunk{int(row['chunk_index']):04d}_original_context.wav")
            clip=folder/filename
            if not clip.exists():wavfile.write(str(clip),rate,signal[round(context_start*rate):round(context_end*rate)])
            clips[clip_id]=clip
        records.append({"scope":item["scope"],"selection":item["selection"],"evaluation_noise":item["noise"],"source_wav_id":row["source_wav_id"],"chunk_index":row["chunk_index"],
            "heat_flux_kW_m2":float(row["heat_flux"])/1000,"onb_kW_m2":p.THRESHOLDS[row["source_wav_id"][:8]],
            "original_wav":source.relative_to(ROOT).as_posix(),"target_start_seconds":start,"target_end_seconds":start+duration,
            "audio_clip":clips[clip_id].relative_to(OUT).as_posix(),"clip_start_seconds_in_original":context_start,"target_start_seconds_in_clip":start-context_start,
            "audio_content":"Original recording only; artificial evaluation noise and model highpass/STFT are not applied",
            "diagnostic_seed":item.get("diagnostic_seed",42),"hgb_prediction_kW_m2":item.get("hgb",""),
            **annotations.get((item["scope"],item["noise"],p.identity(item)),{"recording_anomaly_observed":"","synchronized_video_or_log_available":"","bubble_observation":"","notes":""})})
    with (OUT/"human_review_sheet.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (OUT/"audio_review/README.md").write_text("# 原録音の確認用抜粋\n\n追加の人手確認から新しい事実は得られない前提で、既存診断資料として保持する。本人の確認・記入は要求しない。現在の扱いは親フォルダのhuman_review_instructions.mdと、2026-10-06_model_adoption_decisionの採用判断を参照する。\n\n録音内の対象1秒と前後最大1秒を、増幅・正規化・雑音追加せずに抜き出した。対応表は親フォルダのhuman_review_sheet.csv。evaluation_noiseは解析条件で、このWAVへ人工雑音を加えたという意味ではない。モデル入力のhighpass/STFT加工前の音声であり、音だけから気泡発生を確定しない。\n",encoding="utf-8")
    print("Human review sheet:",len(records),"rows;",len(clips),"original-audio excerpts")


if __name__=="__main__":main()

"""Quantify what fixed attenuation changes in the HGB's acoustic inputs."""
import csv
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"code"))
from utils.dataloading.acoustic_summary_features import AcousticFrequency34


def main():
    with (OUT/"representative_chunks.csv").open(encoding="utf-8-sig",newline="") as stream:cases=list(csv.DictReader(stream))
    transformer=AcousticFrequency34();names=transformer.get_feature_names_out()
    records=[]
    for noise in ["clean","-4","-20"]:
        with np.load(OUT/"cache"/f"features_{noise}.npz") as feature_cache:ids=feature_cache["ids"].tolist()
        lookup={idx:i for i,idx in enumerate(ids)};raw=np.load(OUT/"cache"/f"raw_{noise}.npy",mmap_mode="r")
        for case in [r for r in cases if r["noise"]==noise]:
            idx=lookup[f"{case['source_wav_id']}|{int(case['chunk_index'])}"]
            original=np.asarray(raw[idx]);baseline=transformer.transform([original])[0]
            changes={"global_power_x0.1":original*.1}
            for low,high in [(1700,2100),(2100,2500),(2500,2900)]:
                altered=original.copy();altered[:,round(224*low/3000):round(224*high/3000),0]*=.1
                changes[f"freq_{low}_{high}_power_x0.1"]=altered
            for intervention,changed in changes.items():
                f=transformer.transform([changed])[0]
                for name in ["log_power_total","log_ratio_2000_3000_to_1000_2000","log_ratio_2100_2500_to_1000_2000","spectral_centroid_hz","temporal_power_cv"]:
                    j=list(names).index(name)
                    records.append({"scope":case["scope"],"noise":noise,"source_wav_id":case["source_wav_id"],"chunk_index":case["chunk_index"],"intervention":intervention,"feature":name,"original":float(baseline[j]),"changed":float(f[j]),"difference":float(f[j]-baseline[j])})
    with (OUT/"intervention_feature_changes.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print("Intervention feature changes saved:",len(records))


if __name__=="__main__":main()

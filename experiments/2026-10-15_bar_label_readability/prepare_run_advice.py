"""Check proposed June ONB inputs without importing the training entry point."""
from pathlib import Path
from copy import deepcopy
import ast
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"code"))
from utils.experiment.dataset_jobs import build_dataset_jobs
from utils.experiment.learning_policy import normalize_learning_policy, resolve_experiment_names
from utils.experiment.onb_thresholds import onb_threshold_by_experiment, onb_threshold_provenance_by_experiment
from utils.experiment.onb_main_comparison import validate_main_comparison_jobs


def main():
    entry=ROOT/"code/run_ensemble_regression_onb.py"
    before=hashlib.sha256(entry.read_bytes()).hexdigest()
    tree=ast.parse(entry.read_text(encoding="utf-8-sig"))
    assigned=next(n for n in tree.body if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id=="VALIDATION_CONFIG" for t in n.targets))
    raw=assigned.value.args[0]
    config={key.value:ast.literal_eval(value) for key,value in zip(raw.keys,raw.values) if key.value!="thresholds"}
    thresholds=next(value for key,value in zip(raw.keys,raw.values) if key.value=="thresholds")
    config["thresholds"]={key.value:ast.literal_eval(value) for key,value in zip(thresholds.keys,thresholds.values) if not isinstance(value,ast.Call)}
    current_policy=deepcopy(config["learning_policy"])
    current_output=deepcopy(config["output"])
    config["learning_policy"]["evaluation_mode"]="within_wav_chunk"
    config["learning_policy"]["evaluation_settings"]["within_wav_chunk"]["experiment"]="2025.06.11_0.3_2_6.18_0.3_3"
    config["output"]["main_comparison_report"]=True
    config["thresholds"]["by_experiment"]=onb_threshold_by_experiment()
    config["thresholds"]["provenance_by_experiment"]=onb_threshold_provenance_by_experiment()
    names=resolve_experiment_names(config["data"],config["learning_policy"])
    policy=normalize_learning_policy(config["learning_policy"],names,config["run"]["color_channel"])
    assert names==["2025.06.11_0.3_2_6.18_0.3_3"] and policy["training_noise"]=="clean_only"
    assert config["run"]["epochs"]==200 and config["run"]["internal_validation_split"]=="chunk_kfold"
    jobs=build_dataset_jobs(ROOT.joinpath(*config["data"]["experiment_root_parts"]),names,
        config["data"]["max_freq_hz_list"],config["data"]["noise_dir_names"],
        config["data"]["data_source_dir_by_experiment"],"waterflow",config["data"]["chunk_seconds"],
        config["thresholds"]["by_experiment"],"ensemble","202610/15",1,True,policy)
    assert len(jobs)==7
    check_config=deepcopy(config)
    protocol_path=ROOT/config["output"]["main_comparison_protocol"]
    protocol=json.loads(protocol_path.read_text(encoding="utf-8"))
    grid=config["models"]["parameter_sets"]["by_max_freq_hz"]["maxfreq=3kHz"]
    assert all(isinstance(values,list) and len(values)==1 for parameters in grid.values() for values in parameters.values())
    resolved_models={model:{key:values[0] for key,values in parameters.items()} for model,parameters in grid.items()}
    assert resolved_models==protocol["standard_model_parameters"]
    check_config["models"]["parameter_sets"]=[{"models":resolved_models}]
    check_config["models"]["fixed_model_keys"]=config["models"]["enabled_keys"]
    check_config["learning_policy"]=policy
    check_config["data"]["experiment_names"]=names
    check_config["output"]["main_comparison_protocol"]=protocol
    checked=validate_main_comparison_jobs(jobs,check_config)
    assert checked["status"]=="passed"
    assert all(r["source_wavs"]==36 and r["input_chunks"]==2160
               and r["training_chunks"]==1620 and r["evaluation_chunks"]==540 for r in checked["datasets"])
    assert len({r["evaluation_identity_sha256"] for r in checked["datasets"]})==1
    proposal={"date":"2026-10-15","status":"proposal_only_not_applied_to_entrypoint",
              "purpose":"Presentation output using the existing June-2025 main conditions",
              "current_entrypoint_learning_policy":current_policy,
              "current_entrypoint_main_comparison_report":current_output["main_comparison_report"],
              "recommended_validation_config":config,
              "preserve_model_parameters":resolved_models,
              "protocol_note":"Existing v1 standard epoch is 150. This proposal intentionally keeps the latest actual run at 200; standard_configuration=False is expected, not incomplete training.",
              "replot_alternative":"If only typography/colors change, redraw saved predictions/metrics with the current plotter to preserve the analyzed fit and results.",
              "training_or_inference_started":False}
    destination=ROOT/"configs/experiments/2026-10-15_onb_presentation_run_proposal.json"
    destination.write_text(json.dumps(proposal,ensure_ascii=False,indent=2),encoding="utf-8")
    checked.update({"entrypoint_unchanged":before==hashlib.sha256(entry.read_bytes()).hexdigest(),
                    "entrypoint_sha256":before,"entrypoint_imported":False,
                    "new_fits":0,"new_inference":0,"input_arrays_loaded":False,
                    "proposal_applied_to_entrypoint":False})
    assert checked["entrypoint_unchanged"]
    (OUT/"proposed_run_input_check.json").write_text(json.dumps(checked,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"datasets":len(jobs),"source_wavs":36,"train_chunks":1620,"evaluation_chunks":540,
                      "evaluation_identities_match":True,"entrypoint_unchanged":True,"training_started":False}))


if __name__=="__main__":main()

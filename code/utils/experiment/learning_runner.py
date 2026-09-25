"""学習方針によらず同じ指標・説明性・保存処理を使う実行処理。"""

import csv
import gc
import json
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K

from utils.calculation.regression_detection_metrics import RegressionDetectionMetrics
from utils.calculation.prediction_records import (
    build_fold_prediction_rows, load_sample_metadata_without_arrays,
    write_fold_prediction_csv,
)
from utils.config.parameter_sets import parameter_set_tag, resolve_parameter_set
from utils.dataloading.dataloading_and_conversion import DataLoadingConversion
from utils.explainability.training_integration import (
    aggregate_group_mask_comparison, explainability_outputs_complete,
    maybe_explain_trained_model,
)
from utils.experiment.learning_policy import (
    aligned_indices, build_learning_families, checked_metadata, outer_splits,
    experiment_split_kind, targets_from_metadata, wav_groups,
)
from utils.experiment.result_paths import existing_result_run_path, result_run_path, scoped_result_job
from utils.experiment.run_helpers import (
    append_tuning_summary, has_threshold, is_completed_run, json_default, makedirs,
    open_text, path_exists, run_config_digest, run_dir_name, safe_tag,
    set_global_seed, short_digest, windows_long_path, write_run_manifest,
    saved_run_matches_execution,
)
from utils.models.regression.base_regression import RegressionModelMaker
from utils.experiment.acoustic_selection import AcousticTrainingSelector
from utils.training.internal_validation import fit_individual_performance_cv
from utils.training.fitted_artifacts import (
    begin_fitted_state, finish_fitted_state, save_and_verify_model,
)


SUMMARY_METRICS = [
    "r2", "rmse_all", "mae_all", "r2_high", "rmse_high", "mae_high",
    "rmse_onb", "mae_onb", "roc_auc_cont", "pr_auc_cont",
    "accuracy", "precision", "recall", "f1",
]


def write_json(path, value):
    with open_text(path, "w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, default=json_default)


def _remove_previous_summary_rows(path, job, run_dir):
    """同じ保存条件を再計算したときだけ、その条件の古い集計行を置き換える。"""
    if not path_exists(path):
        return
    with open_text(path, "r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        rows = list(reader)
    retained = [row for row in rows if not (
        row.get("run_dir") == run_dir and all(
            str(row.get(key)) == str(job[key])
            for key in ("experiment_name", "max_freq_hz", "noise_dir_name")))]
    if len(retained) != len(rows):
        with open_text(path, "w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(retained)


class ResultRecorder:
    """一つの評価日・周波数・ノイズ条件の予測と指標を記録する。"""

    def __init__(self, job, parameter_set, run_specs, ensemble, config, run_dir, run_hash,
                 param_tag, model_tag, context):
        self.job = {**job, "learning_context": {**context, "evaluation_noise_dir": job["noise_dir_name"]}}
        self.parameter_set, self.specs, self.ensemble = parameter_set, run_specs, ensemble
        self.config, self.run_dir, self.run_hash = config, run_dir, run_hash
        self.param_tag, self.model_tag = param_tag, model_tag
        self.path = result_run_path(job, run_dir)
        self.snr = job["snr_value"]
        self.threshold = job["threshold"]
        self.keys = [spec["key"] for spec in run_specs]
        self.all_keys = self.keys + ensemble.result_keys
        self.labels = {spec["key"]: spec["label"] for spec in run_specs}
        self.labels.update(ensemble.labels)
        self.metrics = RegressionDetectionMetrics()
        self.store = {key: defaultdict(list) for key in self.all_keys}
        self.train_meta = {key: defaultdict(list) for key in self.keys}
        self.split_records = []
        self.band = config["thresholds"]["onb_band_frac"]

    def start(self):
        makedirs(self.path)
        # 中断した実行を完了扱いしないよう、再計算開始時に完了印を外す。
        marker = Path(windows_long_path(self.path / "completed.json"))
        if marker.exists():
            marker.unlink()
        write_run_manifest(self.path, self.job, self.parameter_set, self.specs,
                           self.param_tag, self.model_tag, self.run_hash, self.run_dir,
                           self.config["output"]["run_instance_id"], self.config)
        with open_text(self.path / f"validation_results_{self.snr}.txt", "w", encoding="utf-8") as output:
            output.write("学習外データでの熱流束・ONB評価\n")
            output.write(json.dumps(self.job["learning_context"], ensure_ascii=False, indent=2) + "\n")
            output.write(json.dumps(self.config, ensure_ascii=False, indent=2, default=json_default) + "\n")

    def record_history(self, spec, history, fold, plotter):
        if history is not None:
            for key in ("actual_batch_size", "requested_batch_size", "stopped_by_memory_error", "epochs_completed"):
                self.train_meta[spec["key"]][key].append(history.params.get(key))
        plotter.plot_loss_history(history, self.config["run"]["epochs"], spec["label"],
                                  fold, self.path, self.snr)

    def record_fold(self, fold, indices, metadata, y_true, single_predictions, inner_errors,
                    training_groups, fit_id, plotter, crossfit_fit=None,
                    model_epochs=None):
        outputs = self.ensemble.combine_predictions(single_predictions, inner_errors, fold, crossfit_fit)
        self.ensemble.save_crossfit_fit(self.path, fold, crossfit_fit)
        predictions = self.ensemble.merge_predictions(single_predictions, outputs)
        rows = build_fold_prediction_rows(indices, y_true, predictions, metadata, fold)
        if self.config["output"]["save_fold_predictions"]:
            directory = self.path / "fold_pred"
            makedirs(directory)
            write_fold_prediction_csv(directory / f"pred_f{fold}_{self.snr}.csv", rows, self.all_keys)
        for key, prediction in predictions.items():
            for result in (
                self.metrics.regression_metrics(y_true, prediction, self.threshold, self.band),
                self.metrics.detection_metrics_continuous(y_true, prediction, self.threshold),
                self.metrics.detection_metrics_binary(y_true, prediction, self.threshold),
            ):
                for metric, value in result.items():
                    self.store[key][metric].append(value)
        self.ensemble.record_diagnostics(fold, y_true, single_predictions, outputs, self.threshold, self.band)
        if has_threshold(self.threshold):
            for key in self.all_keys:
                fold_metrics = {
                    metric: values[-1]
                    for metric, values in self.store[key].items()
                }
                plotter.plot_regression_scatter(
                    y_true, predictions[key], targets_from_metadata(metadata),
                    fold_metrics, self.threshold, self.path, self.snr, fold,
                    model_key=key, model_label=self.labels[key],
                )
        self.split_records.append({
            "fold": fold, "fit_id": fit_id,
            "training_wav_groups": sorted(set(training_groups)),
            "evaluation_wav_groups": sorted(set(wav_groups(metadata)[indices])),
            "evaluation_sample_indices": indices.tolist(),
            "n_training_chunks": len(training_groups), "n_evaluation_chunks": len(indices),
            "inner_holdout_errors": inner_errors,
            "model_epochs": model_epochs or {},
        })
        write_json(self.path / "split_manifest.json", {
            "learning_context": self.job["learning_context"], "folds": self.split_records})
        with open_text(self.path / f"validation_results_{self.snr}.txt", "a", encoding="utf-8") as output:
            output.write(f"\nFold {fold} | fit_id={fit_id}\n")
            self.ensemble.write_fold_weights(output, outputs)
            for key in self.all_keys:
                output.write(f"  [{self.labels[key]}] " + " | ".join(
                    f"{metric}={self.store[key][metric][-1]:.6g}" for metric in SUMMARY_METRICS) + "\n")
        return outputs

    def finish(self, plotter, update_noise_trends):
        context = self.job["learning_context"]
        aggregate_group_mask_comparison(self.path, self.config["explainability"], self.keys,
                                        context["outer_folds_per_evaluation_day"])
        plot_keys = self.all_keys
        means, errors = zip(*(self.metrics.mean_se(self.store[key]["r2"]) for key in plot_keys))
        plotter.plot_bar("R2 Score", [self.labels[key] for key in plot_keys], means, errors,
                         self.config["run"]["epochs"], self.path, self.snr)
        for metric, title in (
            ("roc_auc_cont", "ROC-AUC (continuous)"),
            ("pr_auc_cont", "PR-AUC (continuous)"),
        ):
            means, errors = zip(*(self.metrics.mean_se(self.store[key][metric]) for key in plot_keys))
            plotter.plot_bar(title, [self.labels[key] for key in plot_keys], means, errors,
                             self.config["run"]["epochs"], self.path, self.snr)
        with open_text(self.path / f"metrics_summary_{self.snr}.csv", "w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["model"] + [f"{metric}_mean" for metric in SUMMARY_METRICS]
                             + [f"{metric}_se" for metric in SUMMARY_METRICS])
            for key in self.all_keys:
                values = [self.metrics.mean_se(self.store[key][metric]) for metric in SUMMARY_METRICS]
                writer.writerow([self.labels[key]] + [f"{value[0]:.6f}" for value in values]
                                 + [f"{value[1]:.6f}" for value in values])
        with open_text(self.path / f"validation_results_{self.snr}.txt", "a", encoding="utf-8") as output:
            output.write("\nchunk指標: fold平均 ± 標準誤差（1分割の場合は標準誤差を推定できません）\n")
            for key in self.all_keys:
                output.write(f"[{self.labels[key]}]\n")
                for metric in SUMMARY_METRICS:
                    mean, se = self.metrics.mean_se(self.store[key][metric])
                    output.write(f"  {metric}: {mean:.6f} ± {se:.6f}\n")
        base = Path(self.job["save_base_path"])
        summary = base / "tuning_summary.csv"
        if self.config["output"]["save_tuning_summary"]:
            _remove_previous_summary_rows(summary, self.job, self.run_dir)
        _remove_previous_summary_rows(base / "ensemble_presentation_summary.csv", self.job, self.run_dir)
        self.ensemble.save_reports(
            save_path=self.path, base_save_path=base, snr_value=self.snr,
            store=self.store, metrics=self.metrics, summary_metrics=SUMMARY_METRICS, plotter=plotter,
            run_instance_id=self.config["output"]["run_instance_id"], run_hash=self.run_hash,
            run_dir=self.run_dir, job=self.job,
        )
        append_tuning_summary(summary, self.job, self.parameter_set, self.specs, self.store,
                              self.train_meta, SUMMARY_METRICS, self.metrics, self.param_tag,
                              self.run_dir, self.run_hash, self.path, self.all_keys,
                              self.config["output"]["save_tuning_summary"], self.config["output"]["run_instance_id"])
        write_json(self.path / "completed.json", {"run_hash": self.run_hash,
                   "fit_ids": [record["fit_id"] for record in self.split_records]})
        update_noise_trends(plotter, self.job, self.run_dir, self.run_hash, self.keys)
        print(f"評価結果を保存: {self.path}")


def _completed(recorder):
    config, job = recorder.config, recorder.job
    path = existing_result_run_path(job, recorder.run_dir)
    done = is_completed_run(Path(job["save_base_path"]) / "tuning_summary.csv", recorder.run_dir,
                            path, job["snr_value"], config["output"]["resume_completed_runs"],
                            config["output"]["save_tuning_summary"], run_hash=recorder.run_hash)
    if done and config["explainability"].get("retrain_completed_runs_for_xai", False):
        done = explainability_outputs_complete(path, config["explainability"], recorder.keys,
                 job["learning_context"]["outer_folds_per_evaluation_day"],
                 experiment_name=job["experiment_name"], max_freq_name=job["max_freq_hz"],
                 noise_dir_name=job["noise_dir_name"])
    return done


def run_learning_experiments(jobs, policy, config, enabled_specs, parameter_sets,
                             ensemble_manager, trainer, plotter, update_noise_trends):
    """同じ学習済みモデルを必要な評価ノイズへ適用し、完了後に共通出力を保存する。"""
    families = build_learning_families(jobs, policy, config["data"]["experiment_names"])
    selector = AcousticTrainingSelector(config.get("acoustic_selection"), config["thresholds"].get("by_experiment", {}))
    if "acoustic_selection" in config:
        config["acoustic_selection"] = selector.config
    performance_cv = "performance_kfold" in ensemble_manager.selected_strategy_names
    crossfit_cv = any(
        name in {"subset_equal_cv", "crossfit_wav_stack", "crossfit_shrinkage_stack"}
        for name in ensemble_manager.selected_strategy_names
    )
    if performance_cv and "inner_holdout" in ensemble_manager.selected_strategy_names:
        raise ValueError("performance_kfoldとinner_holdoutは重み推定が異なるため同時選択できません。")
    split_kind = experiment_split_kind(policy)
    if selector.enabled and split_kind == "within_day":
        raise ValueError("学習選別は実験日を分離した評価で使用してください。")
    if (selector.enabled and not performance_cv and selector.mode != "peak_height"
            and (crossfit_cv or "inner_holdout" in ensemble_manager.selected_strategy_names)):
        raise ValueError(
            "background_quantile selection must be fitted inside each weight CV fold. "
            "Use fixed peak_height selection with crossfit/inner_holdout."
        )
    loader = DataLoadingConversion()
    reference_metadata = {}
    for family_i, family in enumerate(families, 1):
        context = dict(family["context"])
        fold_count = config["run"]["folds"] if split_kind == "within_day" else 1
        context["outer_folds_per_evaluation_day"] = fold_count
        context["training_datasets"] = [{key: str(job[key]) for key in
                                        ("experiment_name", "data_path", "noise_dir_name", "max_freq_hz")}
                                       for job in family["training_jobs"]]
        print(f"学習計画 {family_i}/{len(families)}: {context}")
        for parameter_i, parameter_set in enumerate(parameter_sets, 1):
            specs = resolve_parameter_set(enabled_specs, parameter_set)
            model_tag = "-".join(spec["key"] for spec in specs)
            if config["run"]["smoke_test"]:
                model_tag = "s_" + model_tag
            param_tag = parameter_set_tag(parameter_set, specs, safe_tag)
            run_hash = run_config_digest(config, parameter_set, specs, model_tag,
                                         config["output"]["save_fold_predictions"])
            ensemble = ensemble_manager.create_run(specs)
            evaluation_jobs = family["evaluation_jobs"]
            if config["output"].get("run_scoped_result_dir", False):
                execution_id = config["output"].get("execution_id")
                evaluation_jobs = [scoped_result_job(
                    job, execution_id, run_hash, config,
                    parameter_index=parameter_i,
                    parameter_count=len(parameter_sets),
                )
                                   for job in evaluation_jobs]
                # 実行・条件の識別は日付フォルダで済むため、末尾のrun階層は作らない。
                run_dir = ""
            else:
                run_dir = run_dir_name(
                    config["run"]["epochs"], param_tag, model_tag,
                    ensemble.enabled, ensemble.strategy_tag,
                    run_hash=run_hash,
                    run_instance_id=config["output"]["run_instance_id"],
                )
            # 既存結果を上書きしないよう、保存先のmanifestと実行条件を確認する。
            saved_manifests = []
            for job in evaluation_jobs:
                manifest_path = existing_result_run_path(job, run_dir) / "run_manifest.json"
                if path_exists(manifest_path):
                    with open_text(manifest_path, "r", encoding="utf-8") as source:
                        saved_manifests.append(json.load(source))
            if saved_manifests:
                compatible = all(saved_run_matches_execution(manifest, config, parameter_set, specs,
                                 model_tag, config["output"]["save_fold_predictions"]) for manifest in saved_manifests)
                hashes = {manifest["run_hash"] for manifest in saved_manifests}
                if compatible and len(hashes) == 1:
                    run_hash = hashes.pop()
                else:
                    raise RuntimeError(
                        f"Result directory hash collision or inconsistent manifests: "
                        f"{result_run_path(evaluation_jobs[0], run_dir)}"
                    )
            recorders = [ResultRecorder(job, parameter_set, specs, ensemble_manager.create_run(specs),
                         config, run_dir, run_hash, param_tag, model_tag, context)
                         for job in evaluation_jobs]
            completion = [_completed(recorder) for recorder in recorders]
            if all(completion):
                for recorder in recorders:
                    update_noise_trends(plotter, recorder.job, run_dir, run_hash, recorder.keys)
                print(f"完了済みのため学習を省略: {run_dir}")
                if not config["run"]["loop_parameter_sets"]:
                    break
                continue
            if any(completion):
                # モデル自体は永続化していない。中断再開でもノイズ間で同一モデルを保証する。
                print("clean学習の一部条件が未完了です。同じモデルで揃えるため、この周波数の全評価ノイズを再計算します。")
            for recorder in recorders:
                recorder.start()
            fit_session_id = uuid4().hex
            train_x, train_y, train_metadata = [], [], []
            for training_job in family["training_jobs"]:
                x, y, metadata = loader.load_npy_data(training_job["data_path"], return_metadata=True)
                train_x.append(x)
                train_y.append(y)
                train_metadata.extend(checked_metadata(metadata, training_job["experiment_name"]))
            x = train_x[0] if len(train_x) == 1 else np.concatenate(train_x)
            y = train_y[0] if len(train_y) == 1 else np.concatenate(train_y)
            del train_x, train_y
            groups = wav_groups(train_metadata)
            metadata_by_noise, splits_by_noise = {}, {}
            for recorder in recorders:
                job = recorder.job
                metadata = checked_metadata(load_sample_metadata_without_arrays(job["data_path"]), job["experiment_name"])
                identity = (job["experiment_name"], job["max_freq_hz"])
                if identity in reference_metadata:
                    aligned_indices(reference_metadata[identity], metadata)
                else:
                    reference_metadata[identity] = metadata
                metadata_by_noise[job["noise_dir_name"]] = metadata
                splits_by_noise[job["noise_dir_name"]] = outer_splits(
                    train_metadata, metadata, fold_count)
            for fold in range(1, fold_count + 1):
                fit_indices = next(iter(splits_by_noise.values()))[fold - 1][0]
                x_fit, y_fit = x[fit_indices], y[fit_indices]
                fit_id = short_digest({"run_hash": run_hash, "context": context, "fold": fold,
                                       "instance": config["output"]["run_instance_id"],
                                       "fit_session": fit_session_id}, length=12)
                set_global_seed(config["run"]["random_seed"] + fold)
                model_epochs = {
                    spec["key"]: int(config["run"]["epochs"])
                    for spec in specs
                }
                if performance_cv:
                    inner_errors, internal_audit = fit_individual_performance_cv(
                        trainer, specs, x_fit, y_fit, [train_metadata[i] for i in fit_indices], selector,
                        config["run"]["folds"], config["run"]["random_seed"],
                        config["features"]["pca_components"], model_epochs)
                    for recorder in recorders:
                        write_json(recorder.path / f"internal_validation_fold{fold}.json", internal_audit)
                retained, selection_audit = selector.select([train_metadata[i] for i in fit_indices])
                for recorder in recorders:
                    write_json(recorder.path / f"training_selection_fold{fold}.json", selection_audit)
                fit_indices = fit_indices[retained]
                x_fit, y_fit = x[fit_indices], y[fit_indices]
                if not performance_cv:
                    inner_errors = ensemble.fit_inner_holdout_errors(
                        trainer, x_fit, y_fit, groups[fit_indices], config["features"]["pca_components"],
                        tuple(x.shape[1:]), model_epochs, fold, fold_count)
                crossfit_fit = ensemble.fit_crossfit_weights(
                    trainer, x_fit, y_fit, groups[fit_indices], config["features"]["pca_components"],
                    tuple(x.shape[1:]), model_epochs, fold, fold_count)
                scaler = MinMaxScaler()
                y_scaled = scaler.fit_transform(y_fit.reshape(-1, 1))
                pca, x_fit_pca = None, None
                if any(spec["kind"] == "sklearn" for spec in specs):
                    x_fit_pca, _, pca = trainer.make_pca(x_fit, [], config["features"]["pca_components"], return_pca=True)
                artifact_directory = artifact_manifest = None
                if config["output"].get("save_fitted_artifacts", False):
                    artifact_directory = recorders[0].path / "fitted_state" / f"fold{fold}"
                    artifact_manifest = begin_fitted_state(
                        artifact_directory,
                        fit_id,
                        fold,
                        config["run"]["random_seed"] + fold,
                        groups[fit_indices],
                        model_epochs,
                        pca,
                        scaler,
                        deterministic_ops=config["run"].get("deterministic_ops", True),
                    )
                    for recorder in recorders:
                        write_json(
                            recorder.path / f"fitted_state_reference_f{fold}.json",
                            {
                                "fit_id": fit_id,
                                "canonical_directory": str(artifact_directory.resolve()),
                            },
                        )
                predictions = {recorder.snr: {} for recorder in recorders}
                for spec in specs:
                    set_global_seed(config["run"]["random_seed"] + fold)
                    model, history = trainer.train_one_model(spec, RegressionModelMaker(tuple(x.shape[1:])),
                                        x_fit, y_scaled, x_fit_pca, model_epochs[spec["key"]])
                    if artifact_directory is not None:
                        save_and_verify_model(
                            artifact_directory,
                            artifact_manifest,
                            spec,
                            model,
                            RegressionModelMaker(tuple(x.shape[1:])),
                            trainer,
                            x_fit[: min(8, len(x_fit))],
                            scaler,
                            pca,
                            verify=config["output"].get("verify_reloaded_artifacts", True),
                        )
                    for recorder in recorders:
                        job = recorder.job
                        indices = splits_by_noise[job["noise_dir_name"]][fold - 1][1]
                        metadata = metadata_by_noise[job["noise_dir_name"]]
                        if len(family["training_jobs"]) == 1 and Path(job["data_path"]) == Path(family["training_jobs"][0]["data_path"]):
                            x_eval, y_eval = x[indices], y[indices]
                        else:
                            x_eval, y_eval = loader.load_npy_data(job["data_path"], sample_indices=indices)
                        if x_eval.shape[1:] != x_fit.shape[1:]:
                            raise ValueError("学習入力と評価入力の形状が一致しません。")
                        x_eval_pca = (trainer.transform_pca(pca, x_eval)
                                      if pca is not None and spec["kind"] == "sklearn" else None)
                        prediction = trainer.predict_one_model(spec, model, x_eval, x_eval_pca, scaler)
                        predictions[recorder.snr][spec["key"]] = prediction
                        recorder.record_history(spec, history, fold, plotter)
                        maybe_explain_trained_model(spec, model, scaler, x_eval, y_eval, prediction,
                            recorder.threshold, recorder.path, fold, job["max_freq_hz"], config["explainability"],
                            pca=pca, experiment_name=job["experiment_name"],
                            noise_dir_name=job["noise_dir_name"])
                        del x_eval, y_eval, x_eval_pca
                    del model, history
                    K.clear_session()
                    gc.collect()
                artifact_outputs = None
                for recorder_i, recorder in enumerate(recorders):
                    metadata = metadata_by_noise[recorder.job["noise_dir_name"]]
                    indices = splits_by_noise[recorder.job["noise_dir_name"]][fold - 1][1]
                    outputs = recorder.record_fold(
                        fold, indices, metadata, targets_from_metadata(metadata)[indices],
                        predictions[recorder.snr], inner_errors, groups[fit_indices], fit_id, plotter,
                        crossfit_fit=crossfit_fit, model_epochs=model_epochs,
                    )
                    if recorder_i == 0:
                        artifact_outputs = outputs
                if artifact_directory is not None:
                    finish_fitted_state(
                        artifact_directory, artifact_manifest, artifact_outputs
                    )
                del x_fit, y_fit, y_scaled, x_fit_pca, pca, predictions, crossfit_fit
                gc.collect()
            for recorder in recorders:
                recorder.finish(plotter, update_noise_trends)
            del x, y, train_metadata
            gc.collect()
            if not config["run"]["loop_parameter_sets"]:
                break

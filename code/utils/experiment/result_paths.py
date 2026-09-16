"""周波数／ノイズの保存階層と、旧階層の結果参照を一元管理する。"""

from pathlib import Path

from utils.experiment.run_helpers import path_exists, short_digest


def scoped_result_job(job, execution_id, run_hash):
    """実行とパラメータ条件ごとに日付フォルダを分ける。"""
    if not execution_id or not run_hash:
        raise ValueError("execution_id and run_hash are required for scoped results")
    base = Path(job["save_base_path"])
    scope = short_digest({"execution_id": execution_id, "run_hash": run_hash}, length=12)
    return {**job, "save_base_path": base.with_name(f"{base.name}__{scope}")}


def result_run_path(job, run_dir):
    return Path(job["save_base_path"]) / job["max_freq_hz"] / job["noise_dir_name"] / run_dir


def existing_result_run_path(job, run_dir):
    current = result_run_path(job, run_dir)
    legacy = Path(job["save_base_path"]) / job["noise_dir_name"] / job["max_freq_hz"] / run_dir
    # 新階層に結果があれば優先し、未移行の既存結果だけ旧階層から読む。
    return current if path_exists(current) or not path_exists(legacy) else legacy


def noise_trend_path(job, run_dir):
    return Path(job["save_base_path"]) / job["max_freq_hz"] / "noise_trends" / run_dir

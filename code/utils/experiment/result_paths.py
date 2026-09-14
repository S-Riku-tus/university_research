"""周波数／ノイズの保存階層と、旧階層の結果参照を一元管理する。"""

from pathlib import Path

from utils.experiment.run_helpers import path_exists


def result_run_path(job, run_dir):
    return Path(job["save_base_path"]) / job["max_freq_hz"] / job["noise_dir_name"] / run_dir


def existing_result_run_path(job, run_dir):
    current = result_run_path(job, run_dir)
    legacy = Path(job["save_base_path"]) / job["noise_dir_name"] / job["max_freq_hz"] / run_dir
    # 新階層に結果があれば優先し、未移行の既存結果だけ旧階層から読む。
    return current if path_exists(current) or not path_exists(legacy) else legacy


def noise_trend_path(job, run_dir):
    return Path(job["save_base_path"]) / job["max_freq_hz"] / "noise_trends" / run_dir

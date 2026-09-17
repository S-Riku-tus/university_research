"""WAVごとに、1秒刻みの正方形パワースペクトル画像を生成する。

対象実験は``PROCESSING_EXPERIMENTS``で選択する。読み込み、
44.1 kHzへのリサンプリング、先頭60秒、500 Hzハイパスは
``2.run_npy_waterflow_2つhighpass.py`` と同じ条件である。このスクリプトは
STFT配列やスペクトログラムではなく、各1秒の線スペクトルPNGを保存する。
"""

from __future__ import annotations

import argparse
from math import gcd
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from utils.dataloading.waterflow_preprocessing import highpass_filter


BASE_EXPERIMENT_DIR = (
    REPO_ROOT
    / "Pool_boiling"
    / "Subcooling_20_degrees"
    / "0.3"
)

# ここで処理する実験を選ぶ。1つだけでも複数でも指定できる。
PROCESSING_EXPERIMENTS = [
    # "2025.06.11_0.3_2",
    "2025.06.18_0.3_3",
    "2025.07.09_0.3_1",
]
RECORDING_DIR_NAME = "録音データ_熱流束"
OUTPUT_DIR_NAME = "power_spectrum_png"

TARGET_SAMPLE_RATE = 44_100
AUDIO_SECONDS_USED = 60
CHUNK_SECONDS = 1.0
MAX_FREQUENCY_HZ = 3_000.0

# 2.run_npy_waterflow_2つhighpass.py と同じフィルタ条件。
FILTER_PASS_HZ = 500
FILTER_STOP_HZ = 400
FILTER_PASS_LOSS_DB = 1
FILTER_STOP_ATTENUATION_DB = 40

# 1秒内でWelch平均を取り、ピークを残しつつ線を読みやすくする。
WELCH_SEGMENT_SAMPLES = 8_192
FIGURE_SIZE_INCHES = (6, 6)
FIGURE_DPI = 200


def pcm_to_float(data: np.ndarray) -> np.ndarray:
    """整数PCMまたは浮動小数点WAVをfloat64へ変換する。"""
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float64, copy=False)
    if data.dtype == np.uint8:
        return (data.astype(np.float64) - 128.0) / 128.0
    info = np.iinfo(data.dtype)
    scale = float(max(abs(info.min), info.max))
    return data.astype(np.float64) / scale


def load_like_npy_preprocessing(file_path: Path) -> tuple[np.ndarray, int]:
    """NPY生成と同様に、モノラル化、変換、切出し、フィルタ処理を行う。"""
    source_sr, data = wavfile.read(file_path, mmap=True)
    data = pcm_to_float(data)
    if data.ndim > 1:
        data = data.mean(axis=1)

    if int(source_sr) != TARGET_SAMPLE_RATE:
        divisor = gcd(int(source_sr), TARGET_SAMPLE_RATE)
        data = signal.resample_poly(
            data,
            TARGET_SAMPLE_RATE // divisor,
            int(source_sr) // divisor,
        )

    sample_limit = int(TARGET_SAMPLE_RATE * AUDIO_SECONDS_USED)
    data = np.asarray(data[:sample_limit], dtype=np.float64)
    data = highpass_filter(
        data,
        TARGET_SAMPLE_RATE,
        FILTER_PASS_HZ,
        FILTER_STOP_HZ,
        FILTER_PASS_LOSS_DB,
        FILTER_STOP_ATTENUATION_DB,
    )
    return data, TARGET_SAMPLE_RATE


def calculate_power_spectrum(
    chunk: np.ndarray,
    sample_rate: int,
    max_frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """指定上限までの片側・線形Welchパワースペクトルを返す。"""
    nperseg = min(WELCH_SEGMENT_SAMPLES, len(chunk))
    frequencies, power = signal.welch(
        chunk,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="spectrum",
    )
    keep = frequencies <= max_frequency_hz
    return frequencies[keep], power[keep]


def save_spectrum_image(
    frequencies: np.ndarray,
    power: np.ndarray,
    save_path: Path,
    max_frequency_hz: float,
) -> None:
    """添付例に近い体裁の正方形スペクトル図を保存する。"""
    # waterflow_preprocessingは研究図用にfont.size=30を設定するため、この図だけ
    # 添付例に近いサイズへ明示的に戻す。
    with plt.rc_context(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.labelsize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    ):
        fig, ax = plt.subplots(
            figsize=FIGURE_SIZE_INCHES,
            dpi=FIGURE_DPI,
            layout="constrained",
        )
        ax.plot(frequencies, power, color="#4C9BD6", linewidth=0.75)
        ax.set_xlim(0, max_frequency_hz)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Power")
        ax.set_xticks(np.arange(0, max_frequency_hz + 1, 500))
        ax.ticklabel_format(
            axis="y", style="sci", scilimits=(0, 0), useMathText=True
        )
        ax.grid(True, color="#b0b0b0", linewidth=0.5, alpha=0.45)
        ax.set_box_aspect(1)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, facecolor="white")
        plt.close(fig)


def save_wav_spectrum_chunks(
    file_path: Path,
    output_dir: Path,
    chunk_seconds: float = CHUNK_SECONDS,
    max_frequency_hz: float = MAX_FREQUENCY_HZ,
    max_chunks: int | None = None,
) -> int:
    """1つのWAVから連続する各秒のスペクトルPNGを生成する。"""
    audio, sample_rate = load_like_npy_preprocessing(file_path)
    chunk_samples = int(round(sample_rate * chunk_seconds))
    if chunk_samples <= 0:
        raise ValueError("chunk_seconds must be positive")

    chunk_count = len(audio) // chunk_samples
    if max_chunks is not None:
        chunk_count = min(chunk_count, max_chunks)

    wav_output_dir = output_dir / file_path.stem
    for chunk_index in range(chunk_count):
        start = chunk_index * chunk_samples
        chunk = audio[start : start + chunk_samples]
        frequencies, power = calculate_power_spectrum(
            chunk, sample_rate, max_frequency_hz
        )
        save_spectrum_image(
            frequencies,
            power,
            wav_output_dir / f"chunk-{chunk_index:04d}.png",
            max_frequency_hz,
        )
    return chunk_count


def sorted_wav_files(input_dir: Path) -> list[Path]:
    """index=N.*.wavを測定index順に並べる。"""
    def sort_key(path: Path) -> tuple[int, float | str]:
        prefix = path.stem.split(".", 1)[0]
        if prefix.startswith("index="):
            try:
                return 0, float(prefix.removeprefix("index="))
            except ValueError:
                pass
        return 1, path.name.lower()

    return sorted(input_dir.glob("*.wav"), key=sort_key)


def condition_dir_name(chunk_seconds: float, max_frequency_hz: float) -> str:
    """保存条件がフォルダ名から分かる短い名前を返す。"""
    chunk_label = f"{chunk_seconds:g}s"
    if max_frequency_hz % 1000 == 0:
        frequency_label = f"{max_frequency_hz / 1000:g}kHz"
    else:
        frequency_label = f"{max_frequency_hz:g}Hz"
    return f"{chunk_label}_maxfreq={frequency_label}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="1秒ごとの正方形0--3 kHzパワースペクトルPNGを生成する。"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="PROCESSING_EXPERIMENTSを一時的に上書きする実験名（複数指定可）",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="任意の単一WAVフォルダを処理する場合だけ指定",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="任意の保存先。複数実験ではこの下に実験名を追加する",
    )
    parser.add_argument("--max-frequency-hz", type=float, default=MAX_FREQUENCY_HZ)
    parser.add_argument("--chunk-seconds", type=float, default=CHUNK_SECONDS)
    parser.add_argument(
        "--max-files", type=int, default=None, help="動作確認用のWAV数上限"
    )
    parser.add_argument(
        "--max-chunks", type=int, default=None, help="WAVごとの画像数上限"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_dir is not None and args.experiments is not None:
        raise ValueError("--input-dir and --experiments cannot be used together")

    if args.input_dir is not None:
        jobs = [(None, args.input_dir)]
    else:
        experiment_names = args.experiments or PROCESSING_EXPERIMENTS
        if not experiment_names:
            raise ValueError("PROCESSING_EXPERIMENTSに処理対象を指定してください")
        jobs = [
            (
                experiment_name,
                BASE_EXPERIMENT_DIR
                / experiment_name
                / RECORDING_DIR_NAME,
            )
            for experiment_name in experiment_names
        ]

    total_images = 0
    for experiment_name, input_dir in jobs:
        if not input_dir.is_dir():
            raise FileNotFoundError(f"WAV folder not found: {input_dir}")

        if args.output_dir is None:
            experiment_dir = input_dir.parent
            output_dir = (
                experiment_dir
                / "data"
                / OUTPUT_DIR_NAME
                / condition_dir_name(args.chunk_seconds, args.max_frequency_hz)
            )
        elif len(jobs) > 1 and experiment_name is not None:
            output_dir = args.output_dir / experiment_name
        else:
            output_dir = args.output_dir

        wav_files = sorted_wav_files(input_dir)
        if args.max_files is not None:
            wav_files = wav_files[: args.max_files]
        if not wav_files:
            raise FileNotFoundError(f"WAV files not found: {input_dir}")

        print(f"Experiment: {experiment_name or 'custom input'}")
        print(f"Input : {input_dir}")
        print(f"Output: {output_dir}")
        for file_number, file_path in enumerate(wav_files, start=1):
            image_count = save_wav_spectrum_chunks(
                file_path,
                output_dir,
                chunk_seconds=args.chunk_seconds,
                max_frequency_hz=args.max_frequency_hz,
                max_chunks=args.max_chunks,
            )
            total_images += image_count
            print(
                f"[{file_number}/{len(wav_files)}] "
                f"{file_path.name}: {image_count} images"
            )
    print(f"Completed: {total_images} images")


if __name__ == "__main__":
    main()

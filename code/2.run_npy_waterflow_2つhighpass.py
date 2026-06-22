import os
import csv
import json
import re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import librosa as lr
import soundfile as sf
import math
import seaborn as sns
from scipy import signal
from skimage.transform import resize

plt.rcParams['font.family'] = 'Times New Roman'

# font of equation
plt.rcParams['mathtext.fontset'] = 'cm'

# direction of x and y-scales
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# showing of minor x and y-scales
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True

# showing of scales in all directions
plt.rcParams["xtick.top"] = True
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["ytick.right"] = True

# font size
plt.rcParams["font.size"] = 30

#######################################################################
# 変数の指定
#######################################################################

# ハイパスフィルタの設定
fp = 500  # 通過域端周波数 (Hz)
fs = 400  # 阻止域端周波数 (Hz)
gpass = 0.00001  # 通過域リップル (dB)
gstop = 0.0001  # 阻止域減衰量 (dB)

sample_number = 672

# ★表示・処理するスペクトログラムの最大周波数 (Hz)★
# この値を変更することで、表示範囲とデータ処理範囲が変わります
# max_freq_hz = 22050
# max_freq_hz = 15000
# max_freq_hz = 10000
# max_freq_hz = 5000
# max_freq_hz = 2000
MAX_FREQ_HZ = [2000, 3000, 5000, 10000, 15000, 22050]
# MAX_FREQ_HZ = [3000]

SAVE_DATE = 20260622
DATASET_VERSION = "exactq"
CHUNK = 1

AUDIO_SAMPLES_USED = 2646000
SAVE_NPY = True
SAVE_SPECTROGRAM_PNG = True
PNG_WITH_AXES = True

# 音声ファイルが格納されたフォルダパス
folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2025.06.18_0.3_3\録音データ_熱流束"
waterflow_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\water_flow\water_flow_125.wav"
experiment_root = os.path.dirname(folder_path)
experiment_name = os.path.basename(experiment_root)
heat_flux_csv_path = os.path.join(
    experiment_root,
    f"実験結果{experiment_name}",
    f"heat_flux_{experiment_name}.csv",
)

# 生成された画像を保存するフォルダのパス
base_npy_save_folder_path = os.path.join(
    experiment_root,
    "data",
    "npy",
    f"waterflow_{SAVE_DATE}_{CHUNK}s_{DATASET_VERSION}",
)
base_png_save_folder_path = os.path.join(
    experiment_root,
    "data",
    "spectrogram_png",
    f"waterflow_{SAVE_DATE}_{CHUNK}s_{DATASET_VERSION}",
)

# SNR値のリスト
SNR_list = [None, 0, -4, -8, -12, -16, -20]

#######################################################################

def sanitize_label_for_filename(label):
    return re.sub(r'[<>:"/\\|?*]', "_", str(label).strip())


def wav_index_from_name(name):
    match = re.search(r"index=(\d+)", os.path.splitext(os.path.basename(name))[0])
    if match:
        return int(match.group(1))
    return float("inf")


def load_heat_flux_labels(csv_path):
    """Read exact heat-flux labels from heat_flux_*.csv, keyed by 1-based wav index."""
    if not os.path.exists(csv_path):
        print(f"Warning: heat flux csv not found. Fallback to wav filename labels: {csv_path}")
        return {}

    labels = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader, start=1):
            if "q" not in row:
                raise ValueError(f"'q' column not found in heat flux csv: {csv_path}")
            labels[index] = sanitize_label_for_filename(row["q"])
    return labels


HEAT_FLUX_LABEL_BY_INDEX = load_heat_flux_labels(heat_flux_csv_path)


def heat_flux_label_from_wav(file_path):
    """Use exact q from heat_flux csv. Fallback keeps old filename-based behavior."""
    text = os.path.splitext(os.path.basename(file_path))[0]
    match = re.search(r"index=(\d+)", text)
    if match and HEAT_FLUX_LABEL_BY_INDEX:
        wav_index = int(match.group(1))
        if wav_index not in HEAT_FLUX_LABEL_BY_INDEX:
            raise ValueError(
                f"wav index {wav_index} is not present in heat flux csv: {heat_flux_csv_path}"
            )
        return HEAT_FLUX_LABEL_BY_INDEX[wav_index]

    dot_index = text.find(".")
    if dot_index != -1:
        return sanitize_label_for_filename(text[dot_index + 1:])
    return sanitize_label_for_filename(text)


# function of Fourier transform
def fourier_transform(data, samplerate):

    # preprocessing by window function
    N = len(data)
    window = np.hanning(N)
    input_data = data * window
    
    # fast Fourier transform
    spectrum = np.fft.fft(input_data)
    
    # calculation of amplitude spectrum
    amplitude = np.sqrt((spectrum.real**2) + (spectrum.imag**2)) / (N / 2)
    
    # generalization
    amplitude = 1 / (sum(window) / N) * amplitude
    
    # truncation at Nyquist frequency
    amplitude = amplitude[:int(len(amplitude)/2)]
    
    return amplitude**2

# Short time Fourier transform
def calc_stft(data, sample_number, samplerate):
    # matrix of stft
    stft = []
    
    for i in range(sample_number):
        b1 = i * ((len(data) - sample_number*2) // sample_number)
        b2 = b1 + sample_number*2
        power = fourier_transform(data[b1:b2], samplerate)
        stft.append(power)
        
    return np.array(stft)

# ハイパスフィルタを適用する関数
def highpass_filter(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2
    wp = fp / fn
    ws = fs / fn
    N, Wn = signal.buttord(wp, ws, gpass, gstop)
    b, a = signal.butter(N, Wn, "high")
    y = signal.filtfilt(b, a, x, axis=0)
    return y

def add_waterflow_noise(y, waterflow_noise, snr):
    y_power = np.mean(y**2)
    waterflow_noise_power = np.mean(waterflow_noise**2)
    if waterflow_noise_power == 0:
        raise ValueError("waterflow_noise_power is zero. Cannot scale noise by SNR.")
    scaled_noise = waterflow_noise * np.sqrt((y_power / 10 ** (snr / 10)) / waterflow_noise_power)

    all_data = y + scaled_noise

    return all_data


def write_preprocess_manifest(
        out_dir, max_freq_hz, samplerate, data_length, fft_window_size,
        max_k, snr_db, chunk_seconds):
    fn = samplerate / 2
    N, Wn = signal.buttord(fp / fn, fs / fn, gpass, gstop)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": os.path.relpath(__file__, os.getcwd()),
        "source_audio_folder": folder_path,
        "waterflow_path": waterflow_path,
        "heat_flux_csv_path": heat_flux_csv_path,
        "label_source": "heat_flux_csv.q keyed by wav index",
        "save_date": SAVE_DATE,
        "dataset_version": DATASET_VERSION,
        "output_formats": {
            "npy": SAVE_NPY,
            "spectrogram_png": SAVE_SPECTROGRAM_PNG,
            "png_with_axes": PNG_WITH_AXES,
        },
        "chunk_seconds": chunk_seconds,
        "samplerate_hz": samplerate,
        "audio_samples_used": AUDIO_SAMPLES_USED,
        "segment_samples": data_length,
        "max_freq_hz": max_freq_hz,
        "max_freq_khz_display": max_freq_hz / 1000.0,
        "max_frequency_bin_index": max_k,
        "stft": {
            "frames": sample_number,
            "n_fft": fft_window_size,
            "window": "hann",
            "hop_length_formula": "(len(chunk) - n_fft) // frames",
            "hop_length_samples": (data_length - fft_window_size) // sample_number,
            "spectrum": "linear power, amplitude**2",
            "resize_shape": [224, 224],
            "resize_library": "skimage.transform.resize",
            "saved_array_axes_before_resize": ["time_frame", "frequency_bin"],
        },
        "highpass_filter": {
            "type": "butterworth",
            "fp_hz": fp,
            "fs_hz": fs,
            "gpass_db": gpass,
            "gstop_db": gstop,
            "buttord_order": int(N),
            "buttord_Wn_normalized": float(Wn),
            "buttord_cutoff_hz": float(Wn * fn),
            "application": "scipy.signal.filtfilt",
        },
        "noise": {
            "snr_list_db": snr_db,
            "formula": "y + waterflow_noise * sqrt((mean(y**2) / 10**(snr/10)) / mean(waterflow_noise**2))",
        },
    }
    manifest_path = os.path.join(out_dir, "preprocess_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def save_spectrogram_chunks_with_snr(
        file_path, waterflow_path, base_npy_save_folder_path,
        base_png_save_folder_path, max_freq_hz, CHUNK, snr_db=SNR_list):
    # y, sr = sf.read(file_path)
    # waterflow_noise, _ = sf.read(waterflow_path)

    y, sr = lr.load(file_path, sr=44100)
    waterflow_noise, _ = lr.load(waterflow_path, sr=sr, mono=False)

    y = highpass_filter(y[:AUDIO_SAMPLES_USED], sr, fp, fs, gpass, gstop)
    if waterflow_noise.ndim > 1:
        waterflow_noise = waterflow_noise[0]
    waterflow_noise = highpass_filter(waterflow_noise[:AUDIO_SAMPLES_USED], sr, fp, fs, gpass, gstop)

    # segment length
    data_length = int(44100 * CHUNK)

    # sample number
    sample_number = 672

    # calc_stft関数内で使用されている FFT のウィンドウサイズは sample_number * 2 です
    fft_window_size = sample_number * 2 # 672 * 2 = 1344

    # ★最大周波数に対応するデータ行数の計算★
    # 周波数ビン k の周波数は k * sr / n_fft
    # max_freq_hz 以下の最大の k を求める: k * sr / n_fft <= max_freq_hz -> k <= max_freq_hz * n_fft / sr
    max_k = int(max_freq_hz * fft_window_size / sr)
    # 実際の上限周波数 (kHz表示用)
    max_freq_khz_display = max_freq_hz / 1000.0
    npy_max_freq_parent_folder = os.path.join(
        base_npy_save_folder_path, f"maxfreq={max_freq_khz_display:.0f}kHz")
    png_max_freq_parent_folder = os.path.join(
        base_png_save_folder_path, f"maxfreq={max_freq_khz_display:.0f}kHz")
    if SAVE_NPY:
        os.makedirs(npy_max_freq_parent_folder, exist_ok=True)
        write_preprocess_manifest(
            npy_max_freq_parent_folder, max_freq_hz, sr, data_length,
            fft_window_size, max_k, snr_db, CHUNK)
    if SAVE_SPECTROGRAM_PNG:
        os.makedirs(png_max_freq_parent_folder, exist_ok=True)
        write_preprocess_manifest(
            png_max_freq_parent_folder, max_freq_hz, sr, data_length,
            fft_window_size, max_k, snr_db, CHUNK)
    extracted_label = heat_flux_label_from_wav(file_path)

    for snr in snr_db:
        # ... (ノイズ付加部分は変更なし)
        if snr is None:
            y_noisy = y
            snr_label = "no_noise"
        else:
             # ノイズ音声の長さを合わせる処理を関数内に移動して汎用性を高める
            repeats = int(np.ceil(len(y) / len(waterflow_noise)))
            current_waterflow_noise = np.tile(waterflow_noise, repeats)[:len(y)]
            y_noisy = add_waterflow_noise(y, current_waterflow_noise, snr)
            snr_label = f"SNR={snr}"

        if SAVE_NPY:
            npy_snr_save_folder_path = os.path.join(
                npy_max_freq_parent_folder, f"heatflux_{snr_label}")
            os.makedirs(npy_snr_save_folder_path, exist_ok=True)
        else:
            npy_snr_save_folder_path = None

        if SAVE_SPECTROGRAM_PNG:
            png_snr_save_folder_path = os.path.join(
                png_max_freq_parent_folder, f"heatflux_{snr_label}")
            os.makedirs(png_snr_save_folder_path, exist_ok=True)
        else:
            png_snr_save_folder_path = None

        print(f"Debug: Type of y_noisy: {type(y_noisy)}")
        if isinstance(y_noisy, np.ndarray):
            print(f"Debug: Shape of y_noisy: {y_noisy.shape}")
        else:
            print(f"Debug: y_noisy is not a numpy array")

        for i in range(0, len(y_noisy), data_length):
            chunk = y_noisy[i:i+data_length]
            if len(chunk) < data_length: # 最後のチャンクが短い場合
                 # 足りない部分をゼロで埋めるか、スキップするか選択
                 # ここではゼロ埋めして固定サイズにする
                 chunk = np.pad(chunk, (0, data_length - len(chunk)), 'constant')

            amplitude = calc_stft(chunk, sample_number, sr)

            # ★最大周波数までのデータで切り取る★
            # 計算した max_k を使用
            amplitude_cropped = amplitude[:, :max_k + 1]

            # ★切り取ったデータをリサイズ★
            resized_amplitude = resize(amplitude_cropped, (224, 224))
            # ★ここまで修正★


            # 横軸（時間）の範囲設定
            time_axis = np.linspace(0, CHUNK, resized_amplitude.shape[1]) # リサイズ後の時間軸のサイズに合わせる

            # 画像として保存
            plt.figure(figsize=(6, 6))

            # ★imshowのextentを修正★
            # リサイズされた (224x...) データに対して、0から one_chunk 秒、0から max_freq_khz_display (kHz) の範囲にマッピング
            plt.imshow(np.rot90(resized_amplitude, k=3), cmap='jet', aspect='auto', origin='lower',
                       extent=[time_axis[0], time_axis[-1], 0, max_freq_khz_display]) # extentの周波数上限を計算値に変更
            # ★ここまで修正★

            if not PNG_WITH_AXES:
                plt.axis('off')


            # 軸ラベルを設定
            tick_positions_sec = [0, CHUNK / 2, CHUNK]
            tick_labels_str = [f"{p:g}" for p in tick_positions_sec]
            plt.xticks(tick_positions_sec, labels=tick_labels_str, fontsize=20)


            # ★Y軸の目盛りの位置とラベルを動的に設定★
            # 0から max_freq_khz_display までの範囲で、適切な間隔で目盛りを生成
            # 例: 0, 1, 2, ... , floor(max_freq_khz_display) までを整数刻みで表示
            # より見やすいように、目盛りの数を調整することも可能
            y_ticks_positions = np.arange(0, max_freq_khz_display + 1, step=1) # 1 kHz 刻み

            # 目盛りの数が多すぎる場合は間隔を調整する例
            if len(y_ticks_positions) > 10: # 例として目盛りが10個を超える場合
                 step = math.ceil(len(y_ticks_positions) / 5) # 約5個になるようにステップを計算
                 y_ticks_positions = np.arange(0, max_freq_khz_display + 1, step=step)


            y_ticks_labels = [f"{int(p)}" for p in y_ticks_positions] # 整数ラベル

            plt.yticks(y_ticks_positions, labels=y_ticks_labels, fontsize=20)
            # ★ここまで修正★


            plt.xlabel("Time s", fontsize=25, labelpad=2)
            plt.ylabel("Frequency kHz", fontsize=25, labelpad=2)
            plt.subplots_adjust(left=0.25, right=0.85, top=0.85, bottom=0.25)


            file_stem = f"{extracted_label}_{i//data_length}"
            if SAVE_NPY:
                npy_save_path = os.path.join(npy_snr_save_folder_path, f"{file_stem}.npy")
                np.save(npy_save_path, resized_amplitude)
            if SAVE_SPECTROGRAM_PNG:
                png_save_path = os.path.join(png_snr_save_folder_path, f"{file_stem}.png")
                pad_inches = 0.02 if PNG_WITH_AXES else 0
                plt.savefig(png_save_path, dpi=150, bbox_inches='tight', pad_inches=pad_inches)
            plt.close()

def main():
    if SAVE_NPY:
        os.makedirs(base_npy_save_folder_path, exist_ok=True)
    if SAVE_SPECTROGRAM_PNG:
        os.makedirs(base_png_save_folder_path, exist_ok=True)
    for max_freq_hz in MAX_FREQ_HZ:
        # フォルダ内の全wavファイルに対してスペクトログラムを生成
        # save_spectrogram_chunks_with_snr 関数に max_freq_hz を渡すように変更
        wav_files = sorted([name for name in os.listdir(folder_path) if name.endswith(".wav")], key=wav_index_from_name)
        len_file = len(wav_files)
        for i, filename in enumerate(wav_files):
            file_path = os.path.join(folder_path, filename)
            save_spectrogram_chunks_with_snr(
                file_path, waterflow_path, base_npy_save_folder_path,
                base_png_save_folder_path, max_freq_hz, CHUNK, snr_db=SNR_list)
            print(f"---------- {i+1} / {len_file} ({filename}) done! ----------")

    print("---------- All done! ----------")


if __name__ == "__main__":
    main()

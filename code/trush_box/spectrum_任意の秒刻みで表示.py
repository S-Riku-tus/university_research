import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy import signal

# ハイパスフィルタを適用する関数
def highpass_filter(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2  # ナイキスト周波数
    wp = fp / fn         # 通過域端周波数の正規化
    ws = fs / fn         # 阻止域端周波数の正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  # フィルタの次数とカットオフ周波数
    b, a = signal.butter(N, Wn, "high")           # ハイパスフィルタの設計
    y = signal.filtfilt(b, a, x)                  # フィルタリング
    return y

# FFTによるパワースペクトル計算
def power_spectrum(time_data, samplerate):
    quantity = len(time_data)
    window_func = np.hanning(quantity)
    windowed_data = time_data * window_func

    # フーリエ変換
    spectrum = np.fft.fft(windowed_data)

    # 窓関数補正
    spectrum = 1/(sum(window_func)/quantity) * spectrum

    # パワースペクトル
    power = (np.abs(spectrum)**2)[:quantity//2]  # 正の周波数成分のみ
    freq = np.fft.fftfreq(quantity, d=1/samplerate)[:quantity//2]
    return freq, power

# 1秒ごとのスペクトル表示
def plot_spectrum_by_second(file_path, save_folder, min_hz=0, max_hz=3000, fp=700, fs=600, gpass=0.00001, gstop=0.0001, one_chunk=1):
    # 音声データ読み込み
    data, samplerate = librosa.load(file_path, sr=44100)

    # ハイパスフィルタを適用
    data = highpass_filter(data, samplerate, fp, fs, gpass, gstop)

    # 出力フォルダ作成
    os.makedirs(save_folder, exist_ok=True)

    # 音声データがステレオの場合はモノラルに変換
    if data.ndim == 2:
        data = data.mean(axis=1)

    # 1秒ごとのデータに分割してスペクトルを計算
    chunk_samples = 4410  # one_chunk * samplerate  # 1秒分のサンプル数
    total_seconds = len(data) // chunk_samples
    data = data[:total_seconds * chunk_samples]

    for i in range(0, len(data), chunk_samples):
        chunk = data[i:i+chunk_samples]
        chunk = highpass_filter(chunk, samplerate, fp, fs, gpass, gstop)

        # スペクトル計算
        freq, power = power_spectrum(chunk, samplerate)

        # 指定範囲内のデータにフィルタリング
        mask = (freq >= min_hz) & (freq <= max_hz)
        freq = freq[mask]
        power = power[mask]

        # プロット
        plt.figure(figsize=(8, 8))
        plt.plot(freq, power, label=f"Second {i+1}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.title(f"Power Spectrum (Second {i+1})")
        plt.grid()
        plt.legend()

        # 保存
        save_path = os.path.join(save_folder, f"spectrum_second_{i//chunk_samples}.png")
        plt.savefig(save_path)
        plt.close()

# フォルダ内のすべてのWAVファイルを処理
def process_all_wav_files(folder_path, save_root_folder, min_hz=0, max_hz=3000, fp=500, fs=400, gpass=0.00001, gstop=0.0001, one_chunk=0.1):
    # フォルダ内のすべてのファイルをループ処理
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)

            # 各ファイル用の保存フォルダを作成
            file_save_folder = os.path.join(save_root_folder, os.path.splitext(filename)[0])
            os.makedirs(file_save_folder, exist_ok=True)

            # スペクトルプロットを生成
            plot_spectrum_by_second(file_path, file_save_folder, min_hz, max_hz, fp, fs, gpass, gstop, one_chunk)

# 実行例
input_folder = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_2\録音データ_熱流束"
save_root_folder = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_2\data\spectrogram\スペクトル0.1秒刻み比較用"
process_all_wav_files(input_folder, save_root_folder)

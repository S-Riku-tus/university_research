import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# 音声ファイルが格納されたフォルダパス  
folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1\filtered_audio"
 
# 生成された画像を保存するフォルダのパス
save_folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1\data\spectrum"

if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

def save_spectrum(file_path, save_folder_path, max_freq=3000):
    # 音声ファイルの読み込み
    signal, sr = librosa.load(file_path, sr=44110)
    
    # FFTの計算
    N = len(signal)  # 信号のサンプル数
    T = 1.0 / sr  # サンプリング間隔

    # ハニング窓を適用
    window = np.hanning(N)
    windowed_signal = signal * window

    amplitude = np.fft.fft(windowed_signal)  # FFTを計算
    f = np.fft.fftfreq(N, T)  # 周波数軸を生成

    # 振幅スペクトルの絶対値を計算（対称性を考慮して前半のみプロット）
    amplitude = np.abs(amplitude[:N // 2])
    f = f[:N // 2]

    # **最小値-最大値の正規化を追加**
    amplitude_min = np.min(amplitude)
    amplitude_max = np.max(amplitude)
    if amplitude_max - amplitude_min > 0:  # ゼロ除算を防ぐためにチェック
        amplitude = (amplitude - amplitude_min) / (amplitude_max - amplitude_min)

    max_freq = min(max_freq, f[-1])

    # スペクトルをプロットして保存
    plt.figure(figsize=(6,6))
    plt.plot(f, amplitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude a.u.')
    plt.title('Frequency Spectrum')

    plt.xlim(0, max_freq)

    # ファイル名を使って保存
    wav_name = os.path.splitext(os.path.basename(file_path))[0]
    save_path = os.path.join(save_folder_path, f"{wav_name}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# 各WAVファイルごとにスペクトル画像を生成
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        save_spectrum(file_path, save_folder_path, max_freq=3000)

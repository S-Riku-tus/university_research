import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt
from numpy import hanning
import librosa

# 音声ファイルが格納されたフォルダパス  
folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\High_speed_compare\Subcooling_20_degrees\2024.9.18_1_0.3\録音データ_tmp"
# 生成されたPSD画像を保存するフォルダのパス
save_folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\High_speed_compare\Subcooling_20_degrees\2024.9.18_1_0.3\data\spectrogram\waterflow\tmp\Spectrum"

# 保存フォルダの準備
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

# ハイパスフィルタの設定
def apply_highpass_filter(signal, sr, cutoff=1000, order=4):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal)

# PSDを保存する関数
def save_psd(file_path, save_folder_path, sr=44100, segment_size=4410, overlap_size=2205, n_segments=1000):
    # 音声ファイルの読み込み
    signal, sr = librosa.load(file_path, sr=sr)
    
    # ハイパスフィルタで前処理
    filtered_signal = apply_highpass_filter(signal, sr, cutoff=900)
    
    # ハニング窓の定義
    window = hanning(segment_size)
    
    # Welch法でPSDを計算
    freqs, psd = welch(filtered_signal, fs=sr, window=window, nperseg=segment_size, noverlap=overlap_size, nfft=segment_size)
    
    # PSDをプロットして保存（線形スケール）
    plt.figure(figsize=(6, 6))
    plt.plot(freqs, psd)  # 線形スケールで表示
    plt.xlim(0, 2e4)
    plt.ylim(0, 5e-9)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (PSD) [a.u./Hz]')  # 単位を追加
    plt.title('Power Spectral Density using Welch’s Method')

    # ファイル名を使って保存
    wav_name = os.path.splitext(os.path.basename(file_path))[0]
    save_path = os.path.join(save_folder_path, f"{wav_name}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# 各WAVファイルごとにPSD画像を生成
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        save_psd(file_path, save_folder_path)

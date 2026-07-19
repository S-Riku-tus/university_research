import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

#######################################################################
#                              変数の指定
#######################################################################

# 水流音のファイルパス
waterflow_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\water_flow\water_flow_125.wav"

# 生成された画像を保存するフォルダのパス
save_folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\waterflow_only"

# チャンクの長さ（秒単位）
one_chunk = 1

#######################################################################

def load_and_stft(y, n_fft=1024, hop_length=256):
    """STFTを適用してスペクトログラムを計算"""
    amplitude, _ = librosa.magphase(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)))
    amplitude_normalized = (amplitude - np.min(amplitude)) / (np.max(amplitude) - np.min(amplitude))
    return amplitude_normalized

def save_waterflow_spectrogram(waterflow_path, save_folder_path, one_chunk=1):
    """水流音のスペクトログラムを生成して保存"""
    # 水流音を読み込み
    y, sr = librosa.load(waterflow_path, sr=44100)
    chunk_samples = one_chunk * sr

    # 保存フォルダが存在しない場合は作成
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # 水流音をチャンクごとに分割し、スペクトログラムを保存
    for i in range(0, len(y), chunk_samples):
        chunk = y[i:i + chunk_samples]

        # スペクトログラムを計算
        amplitude = resize(load_and_stft(chunk), (224, 224))

        # 横軸（時間）と縦軸（周波数）の範囲設定
        time_axis = np.linspace(0, one_chunk, amplitude.shape[1])  # 時間軸（秒）
        freq_axis = np.linspace(0, sr // 2000, amplitude.shape[0])  # 周波数軸（kHz）

        # 画像として保存
        plt.figure(figsize=(6, 6))
        plt.imshow(resize(amplitude, (224, 224)), cmap='jet', aspect='auto', origin='lower',
                    extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
        
        # 軸ラベルを設定
        plt.xticks([0, 0.5, 1.0], labels=["0", "0.5", "1.0"])
        plt.yticks([0, 5, 10, 15, 20], labels=["0", "5", "10", "15", "20"])
        plt.xlabel("Time (s)", fontsize=16)
        plt.ylabel("Frequency (kHz)", fontsize=16)

        save_path = os.path.join(save_folder_path, f"waterflow_{i // chunk_samples}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

# 水流音スペクトログラムを保存
save_waterflow_spectrogram(waterflow_path, save_folder_path, one_chunk=one_chunk)

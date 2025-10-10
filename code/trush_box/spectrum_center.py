import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

def load_and_stft(y, n_fft=768, hop_length=512):
    # STFTを計算して振幅スペクトルを取得する
    amplitude = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    return amplitude

def save_spectrum_center(file_path, save_folder_path, center_width=100):
    y, sr = librosa.load(file_path, sr=44100)

    # STFTを計算して振幅スペクトルを取得
    amplitude = load_and_stft(y)

    # スペクトルの中心部分を抽出
    center_start = amplitude.shape[1] // 2 - center_width // 2
    center_end = center_start + center_width
    spectrum_center = amplitude[:, center_start:center_end]

    # 画像として保存 
    plt.figure(figsize=(6,6))
    plt.imshow(spectrum_center, cmap='jet', aspect='auto', origin='lower')
    plt.axis('off')  # 軸を表示しない

    # 保存するファイルのパスを指定
    save_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_spectrum_center.png"
    save_path = os.path.join(save_folder_path, save_filename)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# 音声ファイルが格納されたフォルダパスと、生成された画像を保存するフォルダのパス
folder_path = r"C:\Users\shiba\研究\6月26日_サブクール度10度_0.3mm\録音データ"
save_folder_path = r"C:\Users\shiba\研究\6月26日_サブクール度10度_0.3mm\スペクトル中心"

# フォルダが存在しない場合は作成
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

# 全ての.wavファイルに対して処理を実行
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        save_spectrum_center(file_path, save_folder_path)

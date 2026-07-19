import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

#######################################################################

#                              変数の指定

#######################################################################

# 音声ファイルが格納されたフォルダパス
folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\High_speed_compare\Subcooling_20_degrees\2024.9.18_1_0.3\録音データ_熱流束"
# 生成された画像を保存するフォルダのパス
save_folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\High_speed_compare\Subcooling_20_degrees\2024.9.18_1_0.3\data\spectrogram\whitenoise\tmp"

# SNRのリスト
SNR_list = [None, 0, -4, -8, -12, -16, -20]

if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

#######################################################################

def load_and_stft(y, n_fft=4410, hop_length=2205):  
    # STFTを計算し、振幅スペクトログラムを計算
    amplitude, phase = librosa.magphase(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)))
    
    # 正規化処理を追加（0から1の範囲にスケーリング）
    amplitude_normalized = (amplitude - np.min(amplitude)) / (np.max(amplitude) - np.min(amplitude))
    
    return amplitude_normalized

# ホワイトノイズを音声に付加する関数
def add_white_noise(y, snr_db):
    signal_power = np.mean(y ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), y.shape)
    return y + noise

def save_spectrogram_chunks(file_path, save_folder_path, one_chunk=1, snr_db=SNR_list):
    y, sr = librosa.load(file_path, sr=44100)
    y = y[0, :]
    chunk_samples = one_chunk * sr

    for snr in snr_db:
        if snr is None:
            # ノイズなしの場合
            y_noisy = y
            snr_label = "no_noise"
        else:
            # ホワイトノイズを付加
            y_noisy = add_white_noise(y, snr)
            snr_label = f"SNR={snr}"

        # 画像を保存するフォルダがない場合は新しく作成
        new_save_folder_path = os.path.join(save_folder_path, f"heatflux_{snr_label}")
        if not os.path.exists(new_save_folder_path):
            os.makedirs(new_save_folder_path)

        for i in range(0, len(y_noisy), chunk_samples):
            chunk = y_noisy[i:i+chunk_samples]

            # STFTを行い、振幅スペクトログラムを得てリサイズ
            amplitude = resize(load_and_stft(chunk), (224, 224))

            # 横軸（時間）と縦軸（周波数）の範囲設定
            time_axis = np.linspace(0, one_chunk, amplitude.shape[1])  # 時間軸（秒）
            freq_axis = np.linspace(0, sr // 2000, amplitude.shape[0])  # 周波数軸（kHz）

            # 画像として保存
            plt.figure(figsize=(6, 6))
            plt.imshow(resize(amplitude, (224, 224)), cmap='jet', aspect='auto', origin='lower',
                       extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
            # plt.axis('off')

            # 軸ラベルを設定
            plt.xticks([0, 0.5, 1.0], labels=["0", "0.5", "1.0"])
            plt.yticks([0, 5, 10, 15, 20], labels=["0", "5", "10", "15", "20"])
            plt.xlabel("Time (s)", fontsize=16)
            plt.ylabel("Frequency (kHz)", fontsize=16)

            # ファイル名からラベルを抽出
            text = os.path.splitext(os.path.basename(file_path))[0]
            dot_index = text.find(".")
            if dot_index != -1:
                extracted_label = text[dot_index+1:]

            # 保存パスを設定し、画像を保存
            save_path = os.path.join(new_save_folder_path, f"{extracted_label}_{i//chunk_samples}.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

# フォルダ内の全wavファイルに対してスペクトログラムを生成
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        save_spectrogram_chunks(file_path, save_folder_path, snr_db=SNR_list)

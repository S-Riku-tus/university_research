import os
import librosa
import numpy as np
from skimage.transform import resize

#######################################################################

#                              変数の指定

#######################################################################

COLOR_CHANNEL = 1

# 音声ファイルが格納されたフォルダパス
folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_2\録音データ_熱流束"
# 生成された画像を保存するフォルダのパス
save_folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_2\data\npy\whitenoise"

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

        # 保存フォルダがない場合は作成
        if COLOR_CHANNEL == 1:
            new_save_folder_path = os.path.join(save_folder_path, f"channel=1/heatflux_{snr_label}")
        elif COLOR_CHANNEL == 3:
            new_save_folder_path = os.path.join(save_folder_path, f"channel=3/heatflux_{snr_label}")
        if not os.path.exists(new_save_folder_path):
            os.makedirs(new_save_folder_path)

        for i in range(0, len(y), chunk_samples):
            chunk = y_noisy[i:i+chunk_samples]
            
            # STFTを行い、振幅スペクトログラムを得てリサイズ
            amplitude = resize(load_and_stft(chunk), (224, 224))
            
            if COLOR_CHANNEL == 1:
                # (224, 224, 1) にするため、amplitudeをそのまま使う
                amplitude = np.expand_dims(amplitude, axis=-1)
            elif COLOR_CHANNEL == 3:
                # 振幅スペクトログラムを3つのチャンネルに複製し、(224, 224, 3) にする
                amplitude = np.stack([amplitude] * 3, axis=-1)  # amplitudeを3つ用意してリストにし、それを3つつなげる

            # ファイル名からラベルを抽出
            text = os.path.splitext(os.path.basename(file_path))[0]
            dot_index = text.find(".")
            if dot_index != -1:
                extracted_label = text[dot_index+1:]
            
            # 保存パスを設定して、npy形式で保存
            save_path = os.path.join(new_save_folder_path, f"{extracted_label}_{i//chunk_samples}.npy")
            np.save(save_path, amplitude)

# フォルダ内の全wavファイルに対してスペクトログラムを生成
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        save_spectrogram_chunks(file_path, save_folder_path)

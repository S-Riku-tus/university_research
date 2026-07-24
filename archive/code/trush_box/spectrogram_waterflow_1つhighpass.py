import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.signal import butter, filtfilt
from scipy import signal

#######################################################################

#                              変数の指定

#######################################################################

# 音声ファイルが格納されたフォルダパス
folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1\録音データ_熱流束"
waterflow_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\water_flow\water_flow_125.wav"
# 生成された画像を保存するフォルダのパス
save_folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1\data\spectrogram\waterflow_highpass1つ"

# SNR値のリスト
SNR_list = [None, 0, -4, -8, -12, -16, -20]

if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)
    
#######################################################################

def load_and_stft(y, n_fft=4410, hop_length=2205):  
    amplitude, _ = librosa.magphase(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)))
    amplitude_normalized = (amplitude - np.min(amplitude)) / (np.max(amplitude) - np.min(amplitude))
    return amplitude_normalized

# 水流音ノイズを音声に付加する関数
def add_waterflow_noise(y, waterflow_noise, snr_db):
    # ノイズ音声の長さが不足する場合のために、ループして音声データを作成
    repeats = int(np.ceil(len(y) / len(waterflow_noise)))
    waterflow_noise = np.tile(waterflow_noise, repeats)[:len(y)]
    
    # SNRに基づきノイズを追加
    signal_power = np.mean(y ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    scaling_factor = np.sqrt(noise_power / np.mean(waterflow_noise ** 2))
    waterflow_noise_scaled = waterflow_noise * scaling_factor
    return y + waterflow_noise_scaled

# ハイパスフィルタを適用する関数 (単一のカットオフ周波数)
def highpass_filter_single_cutoff(x, samplerate, cutoff_freq):
    fn = samplerate / 2  # ナイキスト周波数
    normalized_cutoff = cutoff_freq / fn  # 正規化カットオフ周波数
    b, a = signal.butter(4, normalized_cutoff, btype="high")  # 4次のハイパスフィルタ
    y = signal.filtfilt(b, a, x)  # フィルタリング
    return y

# 修正後のsave_spectrogram_chunks_with_snr関数
def save_spectrogram_chunks_with_snr(file_path, waterflow_path, save_folder_path, one_chunk=1, snr_db=SNR_list):
    y, sr = librosa.load(file_path, sr=44100)
    y = highpass_filter_single_cutoff(y, sr, cutoff_freq=500)  # カットオフ周波数を500Hzに設定
    waterflow_y, _ = librosa.load(waterflow_path, sr=44100)
    chunk_samples = one_chunk * sr

    # 最後のチャンクが61秒目を含む場合の修正
    max_chunks = len(y) // chunk_samples  # 60秒分のみ使用
    y = y[:max_chunks * chunk_samples]

    for snr in snr_db:
        if snr is None:
            y_noisy = y
            snr_label = "no_noise"
        else:
            y_noisy = add_waterflow_noise(y, waterflow_y, snr)
            snr_label = f"SNR={snr}"

        new_save_folder_path = os.path.join(save_folder_path, f"heatflux_{snr_label}")
        if not os.path.exists(new_save_folder_path):
            os.makedirs(new_save_folder_path)

        for i in range(0, len(y_noisy), chunk_samples):
            chunk = y_noisy[i:i+chunk_samples]
            amplitude = load_and_stft(chunk)

            # 横軸（時間）と縦軸（周波数）の範囲設定
            time_axis = np.linspace(0, one_chunk, amplitude.shape[1])  # 時間軸（秒）
            freq_axis = np.linspace(0, sr // 2000, amplitude.shape[0])  # 周波数軸（kHz）

            # 画像として保存
            plt.figure(figsize=(6, 6))
            plt.imshow(resize(amplitude, (224, 224)), cmap='jet', aspect='auto', origin='lower',
                       extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
            plt.axis('off')

            # ファイル名のラベル設定
            text = os.path.splitext(os.path.basename(file_path))[0]
            dot_index = text.find(".")
            extracted_label = text[dot_index + 1:] if dot_index != -1 else text

            save_path = os.path.join(new_save_folder_path, f"{extracted_label}_{i//chunk_samples}.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

# フォルダ内の全wavファイルに対してスペクトログラムを生成
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        save_spectrogram_chunks_with_snr(file_path, waterflow_path, save_folder_path, snr_db=SNR_list)


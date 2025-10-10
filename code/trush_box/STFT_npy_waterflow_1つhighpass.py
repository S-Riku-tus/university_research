import os
import librosa
import numpy as np
from scipy import signal
from skimage.transform import resize

#######################################################################

#                              変数の指定

#######################################################################

# 音声ファイルが格納されたフォルダパス
folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1\録音データ_熱流束"
waterflow_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\water_flow\water_flow_125.wav"
# 生成されたデータを保存するフォルダのパス
save_folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1\data\npy\waterflow_highpass1つ"

# SNR値のリスト
SNR_list = [None, 0, -4, -8, -12, -16, -20]

if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

#######################################################################

def load_and_stft(y, n_fft=4410, hop_length=2205):
    # STFTを計算し、振幅スペクトログラムを得る
    amplitude, _ = librosa.magphase(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)))
    
    # スペクトログラムを0〜1の範囲に正規化
    amplitude_normalized = (amplitude - np.min(amplitude)) / (np.max(amplitude) - np.min(amplitude))
    
    # 224x224にリサイズし、チャンネル次元を追加
    amplitude_resized = resize(amplitude_normalized, (224, 224), anti_aliasing=True)  # リサイズ
    amplitude_resized = np.expand_dims(amplitude_resized, axis=-1)  # チャンネル次元を追加
    
    return amplitude_resized

def add_waterflow_noise(y, waterflow_noise, snr_db):
    repeats = int(np.ceil(len(y) / len(waterflow_noise)))
    waterflow_noise = np.tile(waterflow_noise, repeats)[:len(y)]
    signal_power = np.mean(y ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    scaling_factor = np.sqrt(noise_power / np.mean(waterflow_noise ** 2))
    waterflow_noise_scaled = waterflow_noise * scaling_factor
    return y + waterflow_noise_scaled

def highpass_filter_single_cutoff(x, samplerate, cutoff_freq):
    fn = samplerate / 2
    normalized_cutoff = cutoff_freq / fn
    b, a = signal.butter(4, normalized_cutoff, btype="high")
    y = signal.filtfilt(b, a, x)
    return y

def save_npy_chunks_with_snr(file_path, waterflow_path, save_folder_path, one_chunk=1, snr_db=SNR_list):
    y, sr = librosa.load(file_path, sr=44100)
    y = highpass_filter_single_cutoff(y, sr, cutoff_freq=500)
    waterflow_y, _ = librosa.load(waterflow_path, sr=44100)
    chunk_samples = one_chunk * sr

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

        total_chunks = len(y_noisy) // chunk_samples
        for i in range(total_chunks):
            start_idx = i * chunk_samples
            end_idx = start_idx + chunk_samples
            if end_idx > len(y_noisy):
                break

            chunk = y_noisy[start_idx:end_idx]
            amplitude = load_and_stft(chunk)

            text = os.path.splitext(os.path.basename(file_path))[0]
            dot_index = text.find(".")
            extracted_label = text[dot_index + 1:] if dot_index != -1 else text

            save_path = os.path.join(new_save_folder_path, f"{extracted_label}_{i}.npy")
            np.save(save_path, amplitude)

for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        save_npy_chunks_with_snr(file_path, waterflow_path, save_folder_path, snr_db=SNR_list)

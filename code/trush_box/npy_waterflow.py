import os
import librosa
import numpy as np
import soundfile as sf
from skimage.transform import resize
from scipy import signal

#######################################################################

#                              変数の指定

#######################################################################

COLOR_CHANNEL = 1

# ハイパスフィルタの設定
fp = 500  # 通過域端周波数 (Hz)
fs = 400  # 阻止域端周波数 (Hz)
gpass = 0.00001  # 通過域リップル (dB)
gstop = 0.0001  # 阻止域減衰量 (dB)

# 音声ファイルが格納されたフォルダパス
folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\録音データ_熱流束_合計"
# 生成された画像を保存するフォルダのパス
save_folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_sf_20250408"

# SNRのリスト
SNR_list = [None, 0, -4, -8, -12, -16, -20]
# 水流音ファイルのパス
waterflow_noise_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\water_flow\water_flow_125.wav"

if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

#######################################################################

def load_and_stft(y, n_fft=4410, hop_length=2205):  
    # STFTを計算し、振幅スペクトログラムを計算
    amplitude, _ = librosa.magphase(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)))
    
    # 正規化処理を追加（0から1の範囲にスケーリング）
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

def highpass_filter(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2
    wp = fp / fn
    ws = fs / fn
    N, Wn = signal.buttord(wp, ws, gpass, gstop)
    b, a = signal.butter(N, Wn, "high")
    y = signal.filtfilt(b, a, x)
    return y

def save_spectrogram_chunks(file_path, waterflow_noise_path, save_folder_path, one_chunk=1, snr_db=SNR_list):
    # soundfileで読み込み（この時点では元のサンプリングレート・チャネルが返される）
    y, sr = sf.read(file_path)
    y = highpass_filter(y[:2646000,0], sr, fp, fs, gpass, gstop)
    # y = highpass_filter(y[:,0], sr, fp, fs, gpass, gstop)

    waterflow_y, sr_w = sf.read(waterflow_noise_path)
    waterflow_y = waterflow_y[:2646000,0]

    # librosa.loadでは sr=44100 としてリサンプリングされるため、
    # 同じ結果を得るために、44100Hz以外ならリサンプリングを行う
    target_sr = 44100
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    if sr_w != target_sr:
        waterflow_y = librosa.resample(waterflow_y, orig_sr=sr_w, target_sr=target_sr)

    # 1チャンクあたりのサンプル数
    chunk_samples = one_chunk * sr

    for snr in snr_db:
        if snr is None:
            # ノイズなしの場合
            y_noisy = y
            snr_label = "no_noise"
        else:
            # 水流音ノイズを付加
            y_noisy = add_waterflow_noise(y, waterflow_y, snr)
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
        save_spectrogram_chunks(file_path, waterflow_noise_path, save_folder_path)

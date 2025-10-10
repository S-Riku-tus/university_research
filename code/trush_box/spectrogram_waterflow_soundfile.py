import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from skimage.transform import resize
from scipy.signal import stft  # scipyのSTFTを使用（今回は使っていませんが、インポートは残しています）
from scipy import signal

#######################################################################
#                              変数の指定
#######################################################################

# ハイパスフィルタの設定
fp = 500  # 通過域端周波数 (Hz)
fs = 400  # 阻止域端周波数 (Hz)
gpass = 0.00001  # 通過域リップル (dB)
gstop = 0.0001  # 阻止域減衰量 (dB)

# 音声ファイルが格納されたフォルダパス
folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1\録音データ_熱流束"
waterflow_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\water_flow\water_flow_125.wav"
# 生成された画像を保存するフォルダのパス
save_folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1\data\spectrogram\waterflow_soundfile_highpass_4"
# SNR値のリスト
SNR_list = [None, 0, -4, -8, -12, -16, -20]
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

# matplotlibの設定
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

#######################################################################

def load_and_stft(y, sr, n_fft=4410, hop_length=2205):
    # librosa.stftは入力が1次元（モノラル）であることを前提とするため、
    # モノラルに変換済みであることが必要です。
    amplitude, phase = librosa.magphase(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)))
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

def save_spectrogram_chunks_with_snr(file_path, waterflow_path, save_folder_path, one_chunk=1, snr_db=SNR_list):
    # soundfileで読み込み（この時点では元のサンプリングレート・チャネルが返される）
    y, sr = sf.read(file_path)
    y = highpass_filter(y[:2646000,0], sr, fp, fs, gpass, gstop)
    # y = highpass_filter(y[:,0], sr, fp, fs, gpass, gstop)

    waterflow_y, sr_w = sf.read(waterflow_path)
    waterflow_y = waterflow_y[:2646000,0]
    # waterflow_y = waterflow_y[:2646000,0]
    # waterflow_y = highpass_filter(waterflow_y, sr, fp, fs, gpass, gstop)
    # チャネルが複数（例：ステレオ）の場合は平均してモノラル化する
    # if y.ndim > 1:
    #     # y = np.mean(y, axis=1)
    #     y = y[:, 0]
    # if waterflow_y.ndim > 1:
    #     waterflow_y = np.mean(waterflow_y, axis=1)

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


    # y, sr = librosa.load(file_path, sr=44100)
    # y = highpass_filter(y, sr, fp, fs, gpass, gstop)
    # chunk_samples = one_chunk * sr   #####  librosaを使うと変な感じになる

    # # 水流音ファイルの読み込み
    # waterflow_y, _ = librosa.load(waterflow_path, sr=44100)


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
            amplitude = load_and_stft(chunk, sr=sr)
            # 横軸（時間）と縦軸（周波数）の範囲設定
            time_axis = np.linspace(0, one_chunk, amplitude.shape[1])  # 時間軸（秒）
            freq_axis = np.linspace(0, sr // 2000, amplitude.shape[0])  # 周波数軸（kHz）
            # 画像として保存
            plt.figure(figsize=(6, 6))
            plt.imshow(resize(amplitude, (224, 224)), cmap='jet', aspect='auto', origin='lower',
                       extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
            # plt.axis('off')
            # 軸ラベルを設定（※画像保存のため不要であれば削除可）
            plt.xticks([0, 0.5, 1.0], labels=["0", "0.5", "1.0"])
            plt.yticks([0, 5, 10, 15, 20], labels=["0", "5", "10", "15", "20"])
            plt.xlabel("Time (s)", fontsize=16)
            plt.ylabel("Frequency (kHz)", fontsize=16)
            text = os.path.splitext(os.path.basename(file_path))[0]
            dot_index = text.find(".")
            if dot_index != -1:
                extracted_label = text[dot_index+1:]
            else:
                extracted_label = text
            save_path = os.path.join(new_save_folder_path, f"{extracted_label}_{i//chunk_samples}.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

# フォルダ内の全wavファイルに対してスペクトログラムを生成
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        save_spectrogram_chunks_with_snr(file_path, waterflow_path, save_folder_path, snr_db=SNR_list)
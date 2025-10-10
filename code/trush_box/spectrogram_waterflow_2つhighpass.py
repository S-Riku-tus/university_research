import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.signal import butter, filtfilt
from scipy import signal
import math # 目盛り計算のためにmathモジュールを追加

#######################################################################
# 変数の指定
#######################################################################

# ハイパスフィルタの設定
fp = 1000  # 通過域端周波数 (Hz)
fs = 900  # 阻止域端周波数 (Hz)
gpass = 0.00001  # 通過域リップル (dB)
gstop = 0.0001  # 阻止域減衰量 (dB)

# カットオフ周波数が一つの場合
cutoff_freq = 1000

# ★表示・処理するスペクトログラムの最大周波数 (Hz)★
# この値を変更することで、表示範囲とデータ処理範囲が変わります
max_freq_hz = 22050 # 例: 5000 Hz = 5 kHz
# max_freq_hz = 10000 # 例: 10000 Hz = 10 kHz
# max_freq_hz = 2000 # 例: 2000 Hz = 2 kHz


# 音声ファイルが格納されたフォルダパス
folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1\録音データ_熱流束"
waterflow_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\water_flow\water_flow_125.wav"

# 生成された画像を保存するフォルダのパス
save_folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\spectrogram_5_02_最終_cutoff=1"

if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

# ハイパスのカットオフ周波数の数
HIGHPASS_FILTER_NUM = "1"

# SNR値のリスト
SNR_list = [None, 0, -4, -8, -12, -16, -20]

# matplotlibの設定
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
#######################################################################

def load_and_stft(y, n_fft=4410, hop_length=2205):
    # amplitude, phase = librosa.magphase(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))) # absは不要
    amplitude = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    # 正規化は必須ではないが、元のコードに合わせる
    amplitude_normalized = (amplitude - np.min(amplitude)) / (np.max(amplitude) - np.min(amplitude) + 1e-8) # ゼロ除算防止
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

# ハイパスフィルタを適用する関数
def highpass_filter(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2  # ナイキスト周波数
    wp = fp / fn         # 通過域端周波数の正規化
    ws = fs / fn         # 阻止域端周波数の正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  # フィルタの次数とカットオフ周波数
    b, a = signal.butter(N, Wn, "high")           # ハイパスフィルタの設計
    y = signal.filtfilt(b, a, x)                  # フィルタリング
    return y

def highpass_filter_single_cutoff(x, samplerate, cutoff_freq):
    fn = samplerate / 2  # ナイキスト周波数
    normalized_cutoff = cutoff_freq / fn  # 正規化カットオフ周波数
    b, a = signal.butter(4, normalized_cutoff, btype="high")  # 4次のハイパスフィルタ
    y = signal.filtfilt(b, a, x)  # フィルタリング
    return y


def save_spectrogram_chunks_with_snr(file_path, waterflow_path, save_folder_path, one_chunk=0.1, snr_db=SNR_list, max_freq_hz=5000): # max_freq_hz を引数として受け取る

    y, sr = librosa.load(file_path, sr=44100, mono=False)
    waterflow_noise, _ = librosa.load(waterflow_path, sr=44100, mono=False)

    # ハイパスフィルタ適用 (変更なし)
    if HIGHPASS_FILTER_NUM == "1":
        # mono=Falseでロードしている場合、フィルタは各チャンネルに適用する必要があるか、
        # 使用するチャンネルを選択する必要がある。元のコードは最初のチャンネルを使用
        if y.ndim > 1: y = y[0, :]
        if waterflow_noise.ndim > 1: waterflow_noise = waterflow_noise[0, :]

        # データの長さを揃える (元のコードにあるスライス:2646000は特定ファイルの長さに依存する可能性)
        # 適切には、処理対象のデータ長を取得し、ノイズをそれに合わせるべき
        # ここでは元のコードのスライスをそのまま使用するが、注意が必要
        process_length = min(len(y), len(waterflow_noise), 2646000 if len(y) >= 2646000 and len(waterflow_noise) >= 2646000 else min(len(y), len(waterflow_noise)))

        y = highpass_filter_single_cutoff(y[:process_length], sr, cutoff_freq)
        waterflow_noise = highpass_filter_single_cutoff(waterflow_noise[:process_length], sr, cutoff_freq)

    elif HIGHPASS_FILTER_NUM == "2":
         if y.ndim > 1: y = y[0, :]
         if waterflow_noise.ndim > 1: waterflow_noise = waterflow_noise[0, :]
         process_length = min(len(y), len(waterflow_noise), 2646000 if len(y) >= 2646000 and len(waterflow_noise) >= 2646000 else min(len(y), len(waterflow_noise)))

         y = highpass_filter(y[:process_length], sr, fp, fs, gpass, gstop)
         waterflow_noise = highpass_filter(waterflow_noise[:process_length], sr, fp, fs, gpass, gstop)
    else:
         # フィルタなしの場合もチャンネルと長さを揃える
         if y.ndim > 1: y = y[0, :]
         if waterflow_noise.ndim > 1: waterflow_noise = waterflow_noise[0, :]
         process_length = min(len(y), len(waterflow_noise), 2646000 if len(y) >= 2646000 and len(waterflow_noise) >= 2646000 else min(len(y), len(waterflow_noise)))
         y = y[:process_length]
         waterflow_noise = waterflow_noise[:process_length]


    chunk_samples = int(one_chunk * sr)
    n_fft = 4410 # STFTで使用するFFTサイズ

    # ★最大周波数に対応するデータ行数の計算★
    # 周波数ビン k の周波数は k * sr / n_fft
    # max_freq_hz 以下の最大の k を求める: k * sr / n_fft <= max_freq_hz -> k <= max_freq_hz * n_fft / sr
    max_k = int(max_freq_hz * n_fft / sr)
    # 実際の上限周波数 (kHz表示用)
    max_freq_khz_display = max_freq_hz / 1000.0


    for snr in snr_db:
        # ... (ノイズ付加部分は変更なし)
        if snr is None:
            y_noisy = y
            snr_label = "no_noise"
        else:
             # ノイズ音声の長さを合わせる処理を関数内に移動して汎用性を高める
            repeats = int(np.ceil(len(y) / len(waterflow_noise)))
            current_waterflow_noise = np.tile(waterflow_noise, repeats)[:len(y)]
            y_noisy = add_waterflow_noise(y, current_waterflow_noise, snr)
            snr_label = f"SNR={snr}"


        # SNRごとの保存フォルダを作成 (変更なし)
        new_save_folder_path = os.path.join(save_folder_path, f"heatflux_{snr_label}_maxfreq={max_freq_khz_display:.0f}kHz") # フォルダ名に最大周波数を追加
        if not os.path.exists(new_save_folder_path):
            os.makedirs(new_save_folder_path)

        for i in range(0, len(y_noisy), chunk_samples):
            chunk = y_noisy[i:i+chunk_samples]
            if len(chunk) < chunk_samples: # 最後のチャンクが短い場合
                 # 足りない部分をゼロで埋めるか、スキップするか選択
                 # ここではゼロ埋めして固定サイズにする
                 chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), 'constant')


            amplitude = load_and_stft(chunk, n_fft=n_fft)

            # ★最大周波数までのデータで切り取る★
            # 計算した max_k を使用
            amplitude_cropped = amplitude[:max_k + 1, :]

            # ★切り取ったデータをリサイズ★
            resized_amplitude = resize(amplitude_cropped, (224, 224))
            # ★ここまで修正★


            # 横軸（時間）の範囲設定
            time_axis = np.linspace(0, one_chunk, resized_amplitude.shape[1]) # リサイズ後の時間軸のサイズに合わせる

            # 画像として保存
            plt.figure(figsize=(6, 6))

            # ★imshowのextentを修正★
            # リサイズされた (224x...) データに対して、0から one_chunk 秒、0から max_freq_khz_display (kHz) の範囲にマッピング
            plt.imshow(resized_amplitude, cmap='jet', aspect='auto', origin='lower',
                       extent=[time_axis[0], time_axis[-1], 0, max_freq_khz_display]) # extentの周波数上限を計算値に変更
            # ★ここまで修正★

            # plt.axis('off') # 必要に応じてコメントアウト
            # plt.ylim(...) # extentで範囲を指定するため不要


            # 軸ラベルを設定 (変更なし)
            tick_positions_sec = [0, one_chunk / 2, one_chunk]
            tick_labels_str = [f"{p:g}" for p in tick_positions_sec]
            plt.xticks(tick_positions_sec, labels=tick_labels_str, fontsize=20)


            # ★Y軸の目盛りの位置とラベルを動的に設定★
            # 0から max_freq_khz_display までの範囲で、適切な間隔で目盛りを生成
            # 例: 0, 1, 2, ... , floor(max_freq_khz_display) までを整数刻みで表示
            # より見やすいように、目盛りの数を調整することも可能
            y_ticks_positions = np.arange(0, max_freq_khz_display + 1, step=1) # 1 kHz 刻み

            # 目盛りの数が多すぎる場合は間隔を調整する例
            if len(y_ticks_positions) > 10: # 例として目盛りが10個を超える場合
                 step = math.ceil(len(y_ticks_positions) / 5) # 約5個になるようにステップを計算
                 y_ticks_positions = np.arange(0, max_freq_khz_display + 1, step=step)


            y_ticks_labels = [f"{int(p)}" for p in y_ticks_positions] # 整数ラベル

            plt.yticks(y_ticks_positions, labels=y_ticks_labels, fontsize=20)
            # ★ここまで修正★


            plt.xlabel("Time s", fontsize=25, labelpad=2)
            plt.ylabel("Frequency kHz", fontsize=25, labelpad=2)
            plt.subplots_adjust(left=0.25, right=0.85, top=0.85, bottom=0.25)


            # ファイル名生成、保存処理 (変更なし)
            text = os.path.splitext(os.path.basename(file_path))[0]
            dot_index = text.find(".")
            if dot_index != -1:
                extracted_label = text[dot_index+1:]
            else:
                extracted_label = text

            save_path = os.path.join(new_save_folder_path, f"{extracted_label}_{i//chunk_samples}.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()


# フォルダ内の全wavファイルに対してスペクトログラムを生成 (変更なし)
# save_spectrogram_chunks_with_snr 関数に max_freq_hz を渡すように変更
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        save_spectrogram_chunks_with_snr(file_path, waterflow_path, save_folder_path,
                                         one_chunk=0.1, snr_db=SNR_list,
                                         max_freq_hz=max_freq_hz) # ★ここで max_freq_hz を渡す★

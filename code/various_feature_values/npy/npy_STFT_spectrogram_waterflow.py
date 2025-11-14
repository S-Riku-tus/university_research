import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import soundfile as sf
import math
import seaborn as sns
from scipy import signal
from skimage.transform import resize

plt.rcParams['font.family'] = 'Times New Roman'

# font of equation
plt.rcParams['mathtext.fontset'] = 'cm'

# direction of x and y-scales
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# showing of minor x and y-scales
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True

# showing of scales in all directions
plt.rcParams["xtick.top"] = True
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["ytick.right"] = True

# font size
plt.rcParams["font.size"] = 30

#######################################################################
# 変数の指定
#######################################################################

# ハイパスフィルタの設定
fp = 1000  # 通過域端周波数 (Hz)
fs = 900  # 阻止域端周波数 (Hz)
gpass = 0.00001  # 通過域リップル (dB)
gstop = 0.0001  # 阻止域減衰量 (dB)

sample_number = 672

# ★表示・処理するスペクトログラムの最大周波数 (Hz)★
# この値を変更することで、表示範囲とデータ処理範囲が変わります
# max_freq_hz = 22050
# max_freq_hz = 15000
# max_freq_hz = 10000
# max_freq_hz = 5000
# max_freq_hz = 2000
MAX_FREQ_HZ = [2000, 5000, 10000, 15000, 22050]

SAVE_DATE = 20251114

CHUNK = 1

# 音声ファイルが格納されたフォルダパス
folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\録音データ_熱流束_合計"
waterflow_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\water_flow\water_flow_125.wav"

# 生成された画像を保存するフォルダのパス
base_save_folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_" + f"{SAVE_DATE}_" + f"{CHUNK}s"

if not os.path.exists(base_save_folder_path):
    os.makedirs(base_save_folder_path)

# SNR値のリスト
SNR_list = [None, 0, -4, -8, -12, -16, -20]

#######################################################################

# function of Fourier transform
def fourier_transform(data, samplerate):

    # preprocessing by window function
    N = len(data)
    window = np.hanning(N)
    input_data = data * window
    
    # fast Fourier transform
    spectrum = np.fft.fft(input_data)
    
    # calculation of amplitude spectrum
    amplitude = np.sqrt((spectrum.real**2) + (spectrum.imag**2)) / (N / 2)
    
    # generalization
    amplitude = 1 / (sum(window) / N) * amplitude
    
    # truncation at Nyquist frequency
    amplitude = amplitude[:int(len(amplitude)/2)]
    
    return amplitude**2

# Short time Fourier transform
def calc_stft(data, sample_number, samplerate):
    # matrix of stft
    stft = []
    
    for i in range(sample_number):
        b1 = i * ((len(data) - sample_number*2) // sample_number)
        b2 = b1 + sample_number*2
        power = fourier_transform(data[b1:b2], samplerate)
        stft.append(power)
        
    return np.array(stft)

# ハイパスフィルタを適用する関数
def highpass_filter(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2
    wp = fp / fn
    ws = fs / fn
    N, Wn = signal.buttord(wp, ws, gpass, gstop)
    b, a = signal.butter(N, Wn, "high")
    y = signal.filtfilt(b, a, x, axis=0)
    return y

def add_waterflow_noise(y, waterflow_noise, snr):
    y_power = np.mean(y**2)
    waterflow_noise_power = np.mean(waterflow_noise**2)
    scaled_noise_power = waterflow_noise * np.sqrt((y_power / 10 ** (snr / 10)) / waterflow_noise_power)

    all_data = y_power + scaled_noise_power

    return all_data


def save_spectrogram_chunks_with_snr(file_path, waterflow_path, base_save_folder_path, max_freq_hz, CHUNK, snr_db=SNR_list):
    y, sr = sf.read(file_path)
    waterflow_noise, _ = sf.read(waterflow_path)

    y = highpass_filter(y[:2646000,0], sr, fp, fs, gpass, gstop)
    waterflow_noise = highpass_filter(waterflow_noise[:2646000,0], sr, fp, fs, gpass, gstop)

    # segment length
    data_length = int(44100 * CHUNK)

    # sample number
    sample_number = 672

    # calc_stft関数内で使用されている FFT のウィンドウサイズは sample_number * 2 です
    fft_window_size = sample_number * 2 # 672 * 2 = 1344

    # ★最大周波数に対応するデータ行数の計算★
    # 周波数ビン k の周波数は k * sr / n_fft
    # max_freq_hz 以下の最大の k を求める: k * sr / n_fft <= max_freq_hz -> k <= max_freq_hz * n_fft / sr
    max_k = int(max_freq_hz * fft_window_size / sr)
    # 実際の上限周波数 (kHz表示用)
    max_freq_khz_display = max_freq_hz / 1000.0
    max_freq_parent_folder = os.path.join(base_save_folder_path, f"maxfreq={max_freq_khz_display:.0f}kHz")
    if not os.path.exists(max_freq_parent_folder):
        os.makedirs(max_freq_parent_folder)

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

        snr_save_folder_path = os.path.join(max_freq_parent_folder, f"heatflux_{snr_label}") # パス生成方法を変更
        if not os.path.exists(snr_save_folder_path):
            os.makedirs(snr_save_folder_path)

        print(f"Debug: Type of y_noisy: {type(y_noisy)}")
        if isinstance(y_noisy, np.ndarray):
            print(f"Debug: Shape of y_noisy: {y_noisy.shape}")
        else:
            print(f"Debug: y_noisy is not a numpy array")

        for i in range(0, len(y_noisy), data_length):
            chunk = y_noisy[i:i+data_length]
            if len(chunk) < data_length: # 最後のチャンクが短い場合
                 # 足りない部分をゼロで埋めるか、スキップするか選択
                 # ここではゼロ埋めして固定サイズにする
                 chunk = np.pad(chunk, (0, data_length - len(chunk)), 'constant')

            amplitude = calc_stft(chunk, sample_number, sr)

            # ★最大周波数までのデータで切り取る★
            # 計算した max_k を使用
            amplitude_cropped = amplitude[:, :max_k + 1]

            # ★切り取ったデータをリサイズ★
            resized_amplitude = resize(amplitude_cropped, (224, 224))
            # ★ここまで修正★


            # 横軸（時間）の範囲設定
            time_axis = np.linspace(0, CHUNK, resized_amplitude.shape[1]) # リサイズ後の時間軸のサイズに合わせる

            # 画像として保存
            plt.figure(figsize=(6, 6))

            # ★imshowのextentを修正★
            # リサイズされた (224x...) データに対して、0から one_chunk 秒、0から max_freq_khz_display (kHz) の範囲にマッピング
            plt.imshow(np.rot90(resized_amplitude, k=3), cmap='jet', aspect='auto', origin='lower',
                       extent=[time_axis[0], time_axis[-1], 0, max_freq_khz_display]) # extentの周波数上限を計算値に変更
            # ★ここまで修正★

            plt.axis('off')


            # 軸ラベルを設定
            tick_positions_sec = [0, CHUNK / 2, CHUNK]
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

            save_path = os.path.join(snr_save_folder_path, f"{extracted_label}_{i//data_length}.npy")
            np.save(save_path, resized_amplitude)
            # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()


for max_freq_hz in MAX_FREQ_HZ:
    # フォルダ内の全wavファイルに対してスペクトログラムを生成
    # save_spectrogram_chunks_with_snr 関数に max_freq_hz を渡すように変更
    for i, filename in enumerate(os.listdir(folder_path)):
        if filename.endswith(".wav"):
            len_file = len(os.listdir(folder_path))
            file_path = os.path.join(folder_path, filename)
            save_spectrogram_chunks_with_snr(file_path, waterflow_path, base_save_folder_path, max_freq_hz,
                                            CHUNK, snr_db=SNR_list) # ★ここで max_freq_hz を渡す★
            print(f"---------- {i+1} / {len_file} ({filename}) done! ----------")

print("---------- All done! ----------")

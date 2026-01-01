import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy import signal

#######################################################################
# グラフのスタイル設定
#######################################################################
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams["xtick.top"] = True
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["font.size"] = 30

#######################################################################
# 変数の指定
#######################################################################

# ハイパスフィルタの設定
fp = 500       # 通過域端周波数 (Hz)
fs = 400       # 阻止域端周波数 (Hz)
gpass = 0.00001 # 通過域リップル (dB)
gstop = 0.0001  # 阻止域減衰量 (dB)

# ★表示・処理する最大周波数のリスト★
MAX_FREQ_HZ = [3000]

# 切り出す時間幅 (秒)
CHUNK = 1

SAVE_DATE = 20251211

# SNR値のリスト (Noneはノイズなし)
SNR_list = [None, 0, -4, -8, -12, -16, -20]

# 音声ファイルが格納されたフォルダパス
folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2025.06.11_0.3_2\録音データ_熱流束"

# 水流動音（ノイズ）のパス
waterflow_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\water_flow\water_flow_125.wav"

# 生成された画像を保存するベースフォルダ
base_save_folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2025.06.11_0.3_2\data\spectrum\waterflow_" + f"{SAVE_DATE}_" + f"{CHUNK}s"

#######################################################################

# ハイパスフィルタを適用する関数
def highpass_filter(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2
    wp = fp / fn
    ws = fs / fn
    N, Wn = signal.buttord(wp, ws, gpass, gstop)
    b, a = signal.butter(N, Wn, "high")
    y = signal.filtfilt(b, a, x)
    return y

# 水流動音を付加する関数
def add_waterflow_noise(y, waterflow_noise, snr):
    y_power = np.mean(y**2)
    waterflow_noise_power = np.mean(waterflow_noise**2)
    
    # ノイズが無音の場合のゼロ除算回避
    if waterflow_noise_power == 0:
        return y
        
    scaled_noise_power = waterflow_noise * np.sqrt((y_power / 10 ** (snr / 10)) / waterflow_noise_power)
    all_data = y + scaled_noise_power
    return all_data

# FFTによるパワースペクトル計算
def power_spectrum(time_data, samplerate):
    quantity = len(time_data)
    window_func = np.hanning(quantity)
    windowed_data = time_data * window_func

    # フーリエ変換
    spectrum = np.fft.fft(windowed_data)

    # 窓関数補正
    spectrum = 1/(sum(window_func)/quantity) * spectrum

    # パワースペクトル
    power = (np.abs(spectrum)**2)[:quantity//2]
    freq = np.fft.fftfreq(quantity, d=1/samplerate)[:quantity//2]
    return freq, power

# チャンクごとのスペクトルを保存する関数
def save_spectrum_chunks(file_path, waterflow_path, base_save_folder_path, max_freq_hz, CHUNK, snr_list):
    
    # ★重要修正: 高周波成分を守るため sr=None に戻しました★
    y, sr = librosa.load(file_path, sr=44100)

    # 2. 信号のSRに合わせてノイズを読み込む（アップサンプリング）
    waterflow_noise, _ = librosa.load(waterflow_path, sr=sr, mono=False)
    
    print(f"Processing: {os.path.basename(file_path)} | SR: {sr} Hz")

    # 3. ハイパスフィルタ適用 (信号・ノイズ両方)
    y = highpass_filter(y, sr, fp, fs, gpass, gstop)
    waterflow_noise = highpass_filter(waterflow_noise[0], sr, fp, fs, gpass, gstop)

    # 音声データがステレオの場合はモノラルに変換
    if y.ndim == 2:
        y = y.mean(axis=1)

    # 4. SNRごとのループ処理
    for snr in snr_list:
        
        # ノイズ付加処理
        if snr is None:
            y_processed = y
            snr_label = "no_noise"
        else:
            repeats = int(np.ceil(len(y) / len(waterflow_noise)))
            current_waterflow_noise = np.tile(waterflow_noise, repeats)[:len(y)]
            y_processed = add_waterflow_noise(y, current_waterflow_noise, snr)
            snr_label = f"SNR={snr}"

        # -------------------------------------------------------------------
        # ★フォルダ構成の修正部分: ベース > SNR > 周波数 > ファイル名 ★
        # -------------------------------------------------------------------
        
        # 周波数ラベルの作成
        max_freq_khz_display = max_freq_hz / 1000.0
        freq_label = f"maxfreq={max_freq_khz_display:.0f}kHz"
        
        # ファイル名ラベルの作成
        file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]

        # フォルダパスの結合 (順序変更)
        save_folder = os.path.join(base_save_folder_path, snr_label, freq_label, file_name_no_ext)
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # -------------------------------------------------------------------

        # 5. チャンク処理
        chunk_samples = int(sr * CHUNK)
        total_chunks = len(y_processed) // chunk_samples
        data_to_process = y_processed[:total_chunks * chunk_samples]

        for i in range(0, len(data_to_process), chunk_samples):
            chunk = data_to_process[i:i+chunk_samples]
            
            # スペクトル計算
            freq, power = power_spectrum(chunk, sr)

            # 指定範囲内のデータにフィルタリング (表示用)
            mask = (freq >= 0) & (freq <= max_freq_hz)
            freq_masked = freq[mask]
            power_masked = power[mask]

            # プロット
            plt.figure(figsize=(10, 8))
            plt.plot(freq_masked, power_masked, color='blue', linewidth=2)
            
            plt.xlabel("Frequency (Hz)", labelpad=10)
            plt.ylabel("Power", labelpad=10)
            
            # グリッド設定
            plt.grid(which='major', linestyle='-', linewidth=0.5)
            plt.grid(which='minor', linestyle=':', linewidth=0.5)
            
            # X軸の範囲を max_freq_hz に合わせる
            plt.xlim(0, max_freq_hz)

            plt.tight_layout()

            # 保存
            current_time = i / sr
            save_path = os.path.join(save_folder, f"spectrum_{current_time:.2f}s.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

#######################################################################
# メイン実行処理
#######################################################################

if not os.path.exists(base_save_folder_path):
    os.makedirs(base_save_folder_path)

print("Starting processing...")

# 指定された最大周波数のリストでループ
for max_freq_hz in MAX_FREQ_HZ:
    print(f"\nProcessing for Max Frequency: {max_freq_hz} Hz")
    
    # フォルダ内の全wavファイルリストを取得
    wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    len_file = len(wav_files)

    for i, filename in enumerate(wav_files):
        file_path = os.path.join(folder_path, filename)
        
        # スペクトル生成関数の呼び出し
        save_spectrum_chunks(file_path, waterflow_path, base_save_folder_path, max_freq_hz, CHUNK, SNR_list)
        
        print(f"---------- {i+1} / {len_file} ({filename}) done! ----------")

print("\n---------- All done! ----------")
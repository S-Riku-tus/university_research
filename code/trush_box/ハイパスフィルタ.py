import os
import librosa
import numpy as np
from scipy.io.wavfile import write
from scipy import signal

#######################################################################
#                              変数の指定
#######################################################################

# 音声ファイルが格納されたフォルダパス
input_folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_2\録音データ_熱流束"
# フィルタ後のWAVファイルを保存するフォルダパス
output_folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_2\録音データ_ハイパス"

# ハイパスフィルタの設定
fp = 500  # 通過域端周波数 (Hz)
fs = 400  # 阻止域端周波数 (Hz)
gpass = 0.00001  # 通過域リップル (dB)
gstop = 0.0001  # 阻止域減衰量 (dB)

#######################################################################

# ハイパスフィルタを適用する関数
def highpass_filter(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2  # ナイキスト周波数
    wp = fp / fn         # 通過域端周波数の正規化
    ws = fs / fn         # 阻止域端周波数の正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  # フィルタの次数とカットオフ周波数
    b, a = signal.butter(N, Wn, "high")           # ハイパスフィルタの設計
    y = signal.filtfilt(b, a, x)                  # フィルタリング
    return y

# フォルダ内の全WAVファイルに対して処理を実行
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

for filename in os.listdir(input_folder_path):
    if filename.endswith(".wav"):
        # 音声データを読み込み
        file_path = os.path.join(input_folder_path, filename)
        y, sr = librosa.load(file_path, sr=44100, mono=False)  # ステレオ対応で読み込み
        y = y[0, :]  # ステレオの場合、片方のチャンネルを使用
        
        # ハイパスフィルタを適用
        y_filtered = highpass_filter(y, sr, fp, fs, gpass, gstop)
        
        # フィルタ後の音声データをWAVファイルとして保存
        output_file_path = os.path.join(output_folder_path, f"filtered_{filename}")
        write(output_file_path, sr, (y_filtered * 32767).astype(np.int16))  # PCM 16-bit形式で保存
        print(f"Filtered file saved: {output_file_path}")

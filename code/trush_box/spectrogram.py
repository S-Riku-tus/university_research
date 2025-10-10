import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


# 音声ファイルが格納されたフォルダパス  
folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Subcooling_20_degrees\0.3\2024.9.18\録音データ"

# 生成された画像を保存するフォルダのパス
save_folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Subcooling_20_degrees\0.3\2024.9.18"


def load_and_stft(y, n_fft=1024, hop_length=256):  # ステップサイズ(オーバーラップ率)とは、フレームの重なり率のこと

    # STFTを計算したのち、振幅スペクトログラムを計算(np.abs())
    # y...1次元np配列の音声信号, n_fft...FFTを計算するためのウィンドウサイズ, hop_length...各フレーム間のステップサイズ
    amplitude = (np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)))
    return amplitude
    

# one_chunk...音声データを区切る1区間の時間(秒)
# librosa()で読み込んだyは、モノラル(mono=True)の時は一次元のnp配列(サンプル数,)
# chunk_samples...1区間に含まれるサンプル数
def save_spectrogram_chunks(file_path, save_folder_path, one_chank=1):
    y, sr = librosa.load(file_path, sr=44100)
    chunk_samples = one_chank * sr

    # 画像を保存するフォルダがない場合は新しく作成
    new_save_folder_path = os.path.join(save_folder_path, f"spectrogram")
    if not os.path.exists(new_save_folder_path):
        os.makedirs(new_save_folder_path)

    # resize(x, y) で、画像x(2次元配列)をyピクセルに変換
    for i in range(0, len(y), chunk_samples):
        chunk = y[i:i+chunk_samples]

        # 今、chunkが音声データのyの値の一部となっている
        # amplitude = resize(load_and_stft(chunk), (224, 224))
        
        # 画像として保存 
        plt.figure(figsize=(6,6))
        plt.imshow(load_and_stft(chunk), cmap='jet', aspect='auto', origin='lower')
        plt.axis('off')  # 軸を表示しない

        # 保存するファイルのパスを指定
        # os.path.splitext...拡張子を除いたファイルパスと、拡張子を分割し、タプルで返す(第一引数[0]がファイルパス)
        # os.path.basename...フルパスから末尾のファイル名と拡張子を抽出
        save_path = os.path.join(new_save_folder_path, f"{os.path.splitext(os.path.basename(file_path))[0]}_{i//chunk_samples}.png")
        # bbox_inches='tight', pad_inches=0...周囲の余白を最小限にする、余白を0インチにする
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()


# for文ですべてのファイルをリストアップし、filename.endswithで、その中から拡張子が'.wav'のものに対して関数を呼び出し
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        save_spectrogram_chunks(file_path, save_folder_path)
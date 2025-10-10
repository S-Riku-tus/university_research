import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage.transform import resize

def load_and_wavelet(y, wavelet='db1', level=5):
    # ウェーブレット変換を適用
    coeffs = pywt.wavedec(y, wavelet, level=level)
    # 各レベルの詳細係数を連結
    coeff_array, coeff_slices = pywt.coeffs_to_array(coeffs)
    amplitude = np.abs(coeff_array)
    return amplitude

def save_wavelet_chunks(file_path, save_folder_path, one_chunk=1):
    y, sr = librosa.load(file_path)
    chunk_samples = one_chunk * sr

    # 画像を保存するフォルダがない場合は新しく作成
    new_save_folder_path = os.path.join(save_folder_path, "ウェーブレット変換")
    if not os.path.exists(new_save_folder_path):
        os.makedirs(new_save_folder_path)

    for i in range(0, len(y), chunk_samples):
        chunk = y[i:i+chunk_samples]
        
        # 今、chunkが音声データのyの値の一部となっている
        amplitude = resize(load_and_wavelet(chunk), (224, 224))
        
        # 画像として保存 
        plt.figure(figsize=(6,6))
        plt.imshow(amplitude, cmap='jet', aspect='auto', origin='lower')
        plt.axis('off')  # 軸を表示しない

        # 保存するファイルのパスを指定
        save_path = os.path.join(new_save_folder_path, f"{os.path.splitext(os.path.basename(file_path))[0]}_{i//chunk_samples}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

folder_path = r"C:\Users\shiba\研究\6月26日_サブクール度10度_0.3mm\録音データ"  # 音声ファイルが格納されたフォルダパス
save_folder_path = r"C:\Users\shiba\研究\6月26日_サブクール度10度_0.3mm"  # 生成された画像を保存するフォルダのパス

# for文ですべてのファイルをリストアップし、filename.endswithで、その中から拡張子が'.wav'のものに対して関数を呼び出し
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        save_wavelet_chunks(file_path, save_folder_path)

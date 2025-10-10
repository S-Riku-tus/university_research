import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

def load_and_stft(y, sr, n_fft=1024, hop_length=256):
    # STFTを計算したのち、振幅スペクトログラムを計算(np.abs())
    # y...1次元np配列の音声信号, n_fft...FFTを計算するためのウィンドウサイズ, hop_length...各フレーム間のステップサイズ
    amplitude = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    # dB表記に直す
    amplitude_db = librosa.amplitude_to_db(amplitude, ref=np.max)
    return amplitude_db

def save_spectrogram_chunks(file_path, save_folder_path, one_chunk=1):
    y, sr = librosa.load(file_path)
    chunk_samples = one_chunk * sr

    new_save_folder_path = os.path.join(save_folder_path, "スペクトログラム_dB")
    if not os.path.exists(new_save_folder_path):
        os.makedirs(new_save_folder_path)

    for i in range(0, len(y), chunk_samples):
        chunk = y[i:i+chunk_samples]
        if len(chunk) < chunk_samples:
            break
        
        amplitude_db = resize(load_and_stft(chunk, sr), (224, 224))
        
        plt.figure(figsize=(6, 6))
        plt.imshow(amplitude_db, cmap='jet', aspect='auto', origin='lower')
        plt.axis('off')
        
        save_path = os.path.join(new_save_folder_path, f"{os.path.splitext(os.path.basename(file_path))[0]}_{i//chunk_samples}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

folder_path = r"C:\Users\shiba\研究\6月26日_サブクール度10度_0.3mm\録音データ"
save_folder_path = r"C:\Users\shiba\研究\6月26日_サブクール度10度_0.3mm"

for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        save_spectrogram_chunks(file_path, save_folder_path)

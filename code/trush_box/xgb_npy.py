import os
import librosa
import numpy as np

#######################################################################

#                              変数の指定

#######################################################################

# 音声ファイルが格納されたフォルダパス
folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Subcooling_20_degrees\0.3\2024.9.18\録音データ_熱流束"
# 生成されたデータを保存するフォルダのパス
save_folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Subcooling_20_degrees\0.3\2024.9.18\data"

#######################################################################

def load_and_stft(y, n_fft=1024, hop_length=256):  
    # STFTを計算し、振幅スペクトログラムを計算
    amplitude, phase = librosa.magphase(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)))
    
    # 正規化処理を追加（0から1の範囲にスケーリング）
    amplitude_normalized = (amplitude - np.min(amplitude)) / (np.max(amplitude) - np.min(amplitude))
    
    return amplitude_normalized

def save_spectrogram_chunks(file_path, save_folder_path, one_chunk=1):
    y, sr = librosa.load(file_path, sr=44100)
    chunk_samples = one_chunk * sr

    # 保存フォルダがない場合は作成
    new_save_folder_path = os.path.join(save_folder_path, "heatflux_xgboost_npy")
    if not os.path.exists(new_save_folder_path):
        os.makedirs(new_save_folder_path)

    # チャンク数を計算（全体の長さに基づいて）
    num_chunks = len(y) // chunk_samples

    for i in range(num_chunks):  # 完全なチャンク数分だけループ
        start_index = i * chunk_samples
        chunk = y[start_index:start_index + chunk_samples]
        
        # STFTを行い、振幅スペクトログラムを得る
        amplitude = load_and_stft(chunk)
        
        # (224, 224) のスペクトログラムを1次元に変換（平坦化）
        amplitude_flatten = amplitude.flatten()  # (n,) の一次元配列に変換
        
        # ファイル名からラベルを抽出
        text = os.path.splitext(os.path.basename(file_path))[0]
        dot_index = text.find(".")
        if dot_index != -1:
            extracted_label = text[dot_index+1:]
        
        # 保存パスを設定して、npy形式で保存（1次元データ）
        save_path = os.path.join(new_save_folder_path, f"{extracted_label}_{i+1}.npy")
        np.save(save_path, amplitude_flatten)

# フォルダ内の全wavファイルに対してスペクトログラムを生成し、1次元化して保存
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        save_spectrogram_chunks(file_path, save_folder_path)

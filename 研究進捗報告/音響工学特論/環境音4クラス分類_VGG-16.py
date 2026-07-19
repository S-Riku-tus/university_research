import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.utils import to_categorical
from datetime import datetime  # タイムスタンプの取得
from keras.applications import VGG16

#######################################################################

#                         スペクトログラムの作成

#######################################################################

# STFTを計算して振幅スペクトログラムを得る
def load_and_stft(y, n_fft=1024, hop_length=256):
    amplitude = (np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)))**0.3 * 100
    return amplitude

# ノイズを付加
def add_noise(y, noise_factor=0.01):
    noise = np.random.randn(len(y))
    augmented_y = y + noise_factor * noise
    return augmented_y

# 時間シフト
def time_shift(y, sr, shift_max_ms=100):
    shift = np.random.randint(sr * shift_max_ms / 1000)
    return np.roll(y, shift)

# スペクトラムのランダムシフト
def random_shift_spectrum(spec):
    rows, cols = spec.shape
    rand_shift = np.random.randint(-10, 10)
    return np.roll(spec, rand_shift, axis=1)

# 音声データを区切り、スペクトログラムを生成
def create_spectrogram_chunks(y, sr, chunk_duration=1):
    chunk_samples = chunk_duration * sr
    spectrograms = []

    for i in range(0, len(y), chunk_samples):
        chunk = y[i:i+chunk_samples]
        if len(chunk) == chunk_samples:
            amplitude = resize(load_and_stft(chunk), (224, 224))
            spectrograms.append(amplitude)

    return spectrograms

#######################################################################

#                         データセットの作成

#######################################################################

def create_dataset(folder_path):
    spectrograms = []
    y_labels = []
    label_dict = {}  # ラベルを数値に変換する辞書
    label_index = 0
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            y, sr = librosa.load(file_path)
            
            # ノイズを付加した音声データの生成
            y_noisy = add_noise(y)
            
            # 時間シフトした音声データの生成
            y_time_shifted = time_shift(y, sr)
            
            # 各拡張データに対してスペクトログラムを生成
            for augmented_y in [y, y_noisy, y_time_shifted]:
                specs = create_spectrogram_chunks(augmented_y, sr)
                # スペクトラムのランダムシフトを適用
                specs = [random_shift_spectrum(spec) for spec in specs]
                spectrograms.extend(specs)
                label = os.path.splitext(file_name)[0]  # ファイル名をラベルとして使用
                if label not in label_dict:
                    label_dict[label] = label_index
                    label_index += 1
                y_labels.extend([label_dict[label]] * len(specs))
    
    X = np.array([spec[..., np.newaxis] for spec in spectrograms])  # 4次元テンソルに変換
    y = np.array(y_labels)
    
    # データの正規化
    X = (X - np.mean(X)) / np.std(X)
    
    return X, y, label_dict

#######################################################################

#                         VGG-16のモデルの作成

#######################################################################

def create_vgg16(input_shape, num_classes):
    vgg_model = VGG16(weights=None, include_top=False, input_shape=input_shape)
    for layer in vgg_model.layers:
        layer.trainable = False  # VGG16の重みを固定
    
    x = vgg_model.output
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=vgg_model.input, outputs=predictions)
    return model

#######################################################################

# フォルダパスの指定
folder_path = r"C:\Users\Casper3\Python\Ueki\shibasaki\研究"

# データセットの作成
X, y, label_dict = create_dataset(folder_path)
y = to_categorical(y, num_classes=len(label_dict))

# トレーニングとテストデータに分割
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

input_shape = (224, 224, 1)
num_classes = len(label_dict)
model = create_vgg16(input_shape, num_classes)
adam_optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# ReduceLROnPlateau コールバックの設定
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# トレーニング中に使用するコールバックをリストにまとめる
callbacks = [reduce_lr]

# モデルのトレーニング
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), callbacks=callbacks)

# モデルの評価
score = model.evaluate(X_val, y_val, verbose=0)
print(f'Validation loss: {score[0]} / Validation accuracy: {score[1]}')

#######################################################################

#              accuracyとlossのグラフをプロットして保存

#######################################################################

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_folder = r"C:\Users\Casper3\Python\Ueki\shibasaki\研究\グラフ_vgg16"
os.makedirs(output_folder, exist_ok=True)

# accuracyのグラフ
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
accuracy_plot_path = os.path.join(output_folder, f'accuracy_plot_{timestamp}.png')
plt.savefig(accuracy_plot_path, bbox_inches='tight')
plt.close()

# lossのグラフ
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
loss_plot_path = os.path.join(output_folder, f'loss_plot_{timestamp}.png')
plt.savefig(loss_plot_path, bbox_inches='tight')
plt.close()

print(f'Plots saved at: {accuracy_plot_path} and {loss_plot_path}')

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from PIL import Image

# STFTを計算して振幅スペクトログラムを得る
def load_and_stft(y, n_fft=1024, hop_length=256):
    amplitude = (np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)))**0.3*50
    return amplitude


# 音声データを区切り、スペクトログラムを生成
def create_spectrogram_chunks(y, sr, chunk_duration=1, target_size=(224, 224)):
    chunk_samples = chunk_duration * sr
    spectrograms = []

    for i in range(0, len(y), chunk_samples):
        chunk = y[i:i+chunk_samples]
        if len(chunk) == chunk_samples:
            # STFTを計算して振幅スペクトログラムを得る
            amplitude = load_and_stft(chunk)
            
            # スペクトログラムを画像として扱う
            img = Image.fromarray(amplitude)
            
            # 画像を指定サイズにリサイズする
            resized_img = img.resize(target_size, resample=Image.BILINEAR)
            
            # リサイズした画像を numpy 配列に変換してリストに追加
            resized_amplitude = np.array(resized_img)
            spectrograms.append(resized_amplitude)

    return spectrograms


# データセットの作成
def create_dataset(folder_path):
    spectrograms = []
    y_labels = []
    label_dict = {}  # ラベルを数値に変換する辞書
    label_index = 0
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            y, sr = librosa.load(file_path)
            specs = create_spectrogram_chunks(y, sr)
            spectrograms.extend(specs)
            label = os.path.splitext(file_name)[0]  # ファイル名をラベルとして使用
            if label not in label_dict:
                label_dict[label] = label_index
                label_index += 1
            y_labels.extend([label_dict[label]] * len(specs))
    
    X = np.array([spec[..., np.newaxis] for spec in spectrograms])  # 4次元テンソルに変換
    y = np.array(y_labels)
    
    return X, y, label_dict

# フォルダパスの指定
folder_path = r"C:\Users\shiba\研究"

# データセットの作成
X, y, label_dict = create_dataset(folder_path)
y = to_categorical(y, num_classes=len(label_dict))

# トレーニングとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# AlexNetの構築
def create_alexnet(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

input_shape = (224, 224, 1)
num_classes = len(label_dict)
model = create_alexnet(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# モデルのトレーニング
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# モデルの評価
score = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# テストデータで予測
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# 混同行列の計算
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# 混同行列の可視化
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_dict.keys(), yticklabels=label_dict.keys())
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

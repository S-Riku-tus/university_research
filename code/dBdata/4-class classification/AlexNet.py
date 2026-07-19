import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import random

#######################################################################

#                              変数の指定

#######################################################################

# 訓練時のバッチサイズとエポック数
BATCH_SIZE = 32
#EPOCH_NUM = 200
EPOCH_NUM = 200
# 学習率
LEARNING_RATE = 0.001

# 訓練：検証(検証データの割合)
VAL_RATE = 0.2

# 特徴量データの読み込み
TRAIN_PATH = r"C:\Users\Casper4\Python\Ueki\shibasaki\研究\6月26日_サブクール度10度_0.3mm\スペクトログラム_dB"

# val-lossグラフの保存先フォルダ
SAVE_PATH = r"C:\Users\Casper4\Python\Ueki\shibasaki\研究\6月26日_サブクール度10度_0.3mm\training_results\AlexNet"


#######################################################################

#                       データの読み込みと形の変換

 #######################################################################

def load_images_from_folder(folder_path):
    x, y = [], []
    class_names = ['boiling1', 'boiling2', 'boiling3', 'not_boiling']
    image_files = {class_name: [] for class_name in class_names}

    # 指定した特徴量の保存されているフォルダの中からそれぞれのファイルを読み込む
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):  # 末尾が".png"となっていたら...
            for class_name in class_names:
                if filename.startswith(class_name):
                    image_files[class_name].append(filename)
                    break

    min_images = min(len(image_files[class_name]) for class_name in class_names)

    selected_images = {class_name: random.sample(image_files[class_name], min_images) for class_name in class_names}

    print("boiling1_images length:", len(image_files['boiling1']))
    print("boiling2_images length:", len(image_files['boiling2']))
    print("boiling3_images length:", len(image_files['boiling3']))
    print("not_boiling_images length:", len(image_files['not_boiling']))

    for class_idx, class_name in enumerate(class_names):
        for filename in selected_images[class_name]:
            image = load_img(os.path.join(folder_path, filename), target_size=(224, 224))
            x.append(img_to_array(image))
            y.append(class_idx)

    x = np.array(x)
    y = np.array(y)
    x = x.astype('float32')
    x = x / 255.0
    y = to_categorical(y, num_classes=len(class_names))
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    return x, y, class_names

#######################################################################

#                         AlexNetのモデルの作成

#######################################################################

def alexnet(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_shape, padding="valid"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
    model.add(Conv2D(256, (5, 5), strides=(1, 1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model

#######################################################################

#                         val-lossのグラフの保存

#######################################################################

def plot_training_results(history, output_folder, base_filename='', learning_rate=LEARNING_RATE, epochs=EPOCH_NUM, batch_size=BATCH_SIZE):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # 連番のファイル名生成
    base_path = os.path.join(output_folder, base_filename)
    num = 1
    while os.path.exists(f"{base_path}lr{learning_rate}_ep{epochs}_bs{batch_size}_{num}.png"):
        num += 1
    save_path = f"{base_path}lr{learning_rate}_ep{epochs}_bs{batch_size}_{num}.png"
    
    plt.savefig(save_path)
    plt.show()

#######################################################################

#                           混同行列の保存

#######################################################################

def save_confusion_matrix(model, x_val, y_val, class_names, output_folder, learning_rate=LEARNING_RATE, epochs=EPOCH_NUM, batch_size=BATCH_SIZE):
    # モデルの予測を取得
    y_pred = model.predict(x_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)

    # 混同行列を計算
    cm = confusion_matrix(y_true_classes, y_pred_classes, labels=np.arange(len(class_names)))

    # ヒートマップとして混同行列をプロット
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    os.makedirs(output_folder, exist_ok=True)
    
    num = 1
    base_filename = 'confusion_matrix'
    while os.path.exists(os.path.join(output_folder, f"{base_filename}_lr{learning_rate}_ep{epochs}_bs{batch_size}_{num}.png")):
        num += 1
    plt.savefig(os.path.join(output_folder, f'{base_filename}_lr{learning_rate}_ep{epochs}_bs{batch_size}_{num}.png'))
    plt.show()

#######################################################################

#                                実行部

#######################################################################

# 訓練データのロード
x, y, class_names = load_images_from_folder(TRAIN_PATH)
print("x shape:", x.shape)
print("y shape:", y.shape)

# 訓練データと検証データに分割
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=VAL_RATE, random_state=42)

#######################################################################

# モデルの作成
model = alexnet(input_shape=(224, 224, 3), num_classes=4)
model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

# TensorBoardコールバックの設定
tensorboard_callback = TensorBoard(log_dir='./logs')
# 検証損失（または指定されたモニタリング指標）が改善されなくなったときに学習率を動的に減少させる
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# モデルのトレーニング
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCH_NUM,
    validation_data=(x_val, y_val),
    callbacks=[tensorboard_callback, reduce_lr]
)

# 学習曲線をプロットしてファイルに保存する
plot_training_results(history, SAVE_PATH, base_filename='', learning_rate=LEARNING_RATE, epochs=EPOCH_NUM, batch_size=BATCH_SIZE)

# 混同行列を保存
save_confusion_matrix(model, x_val, y_val, class_names, SAVE_PATH, learning_rate=LEARNING_RATE, epochs=EPOCH_NUM, batch_size=BATCH_SIZE)

import os
import numpy as np
import time
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, r2_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt

#######################################################################

#                              変数の指定

#######################################################################

BATCH_SIZES = [64]
EPOCH_NUMS = [1000]
LEARNING_RATES = [0.0001]

DATA_PATH = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\サブクール度10度\2024.10.3_0.3\スペクトログラム\RGB_熱流束_SNR=0"
SAVE_PATH = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\サブクール度20度\2024.9.18_0.3\regression_result\RGB\AlexNet\SNR=0_non_threshold"

#######################################################################

#                         データの読み込みと形の変換

#######################################################################

def load_images_from_folder(folder_path):
    x, y = [], []
    print("読み込みスタート")
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            heat_flux = float(filename.split('_')[0])
            image = load_img(os.path.join(folder_path, filename), target_size=(224, 224))
            x.append(img_to_array(image))
            y.append(heat_flux)

    x = np.array(x)
    y = np.array(y)
    x = x.astype('float32')
    x = x / 255.0
    return x, y

#######################################################################

#                         AlexNetのモデルの作成

#######################################################################

def alexnet_regression(input_shape):
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
    model.add(Dense(1, activation='linear'))
    return model

#######################################################################

#                                実行部

#######################################################################

def main():
    start_time = time.time()
    x, y = load_images_from_folder(DATA_PATH)
    load_time = time.time() - start_time

    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print(f"データの読み込み時間: {load_time:.2f} 秒")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1

    for train_index, val_index in kf.split(x):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        for lr in LEARNING_RATES:
            for batch_size in BATCH_SIZES:
                for epoch_num in EPOCH_NUMS:
                    print(f"Fold {fold} - Learning Rate: {lr}, Batch Size: {batch_size}, Epochs: {epoch_num}")

                    model = alexnet_regression(input_shape=(224, 224, 3))
                    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

                    start_time = time.time()
                    history = model.fit(x_train, y_train, epochs=epoch_num, batch_size=batch_size, verbose=1, validation_data=(x_val, y_val))
                    train_time = time.time() - start_time

                    print(f"モデルの訓練時間: {train_time:.2f} 秒")

                    y_val_pred = model.predict(x_val)

                    # 回帰分析結果のプロット
                    plt.figure()
                    plt.scatter(y_val, y_val_pred)
                    plt.xlabel('True Heat Flux')
                    plt.ylabel('Predicted Heat Flux')
                    plt.title(f'Regression Analysis: Fold {fold}, LR={lr}, BS={batch_size}, E={epoch_num}')
                    plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--', linewidth=2)
                    plt.tick_params(direction='in', which='both')
                    r2 = r2_score(y_val, y_val_pred)
                    plt.text(0.5, 0.1, f'R^2 Score: {r2:.4f}', ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
                    plt.savefig(os.path.join(SAVE_PATH, f'regression_analysis_lr{lr}_bs{batch_size}_ep{epoch_num}_fold{fold}.png'))
                    plt.close()

                    print(f'R^2 Score: {r2}')

                    # ROC曲線の作成（閾値なし）
                    fpr, tpr, _thresholds = roc_curve(y_val, y_val_pred)  # 閾値を設けず連続値を使用
                    roc_auc = auc(fpr, tpr)

                    # ROC曲線のプロットと保存
                    plt.figure()
                    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
                    plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.0])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve: Fold {fold}, LR={lr}, BS={batch_size}, E={epoch_num}')
                    plt.legend(loc="lower right")
                    plt.savefig(os.path.join(SAVE_PATH, f'roc_curve_lr{lr}_bs{batch_size}_ep{epoch_num}_fold{fold}.png'))
                    plt.close()

        fold += 1

if __name__ == "__main__":
    main()

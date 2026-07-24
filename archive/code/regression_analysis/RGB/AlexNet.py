import os
import numpy as np
import time
import csv
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

# 訓練時のバッチサイズとエポック数のリスト
BATCH_SIZES = [12, 48]
EPOCH_NUMS = [1000]
# 学習率
LEARNING_RATES = [0.0001, 0.0005]
# 閾値
THRESHOLD = 558802  # TODO: ここの値が、csvファイルで求めた沸騰開始点での熱流束となれば、沸騰-非沸騰分類モデルでのROC曲線が書けるのではないか？

COLOR_CHANNEL = 3

# データフォルダ
DATA_PATH = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Subcooling_20_degrees\0.3\2024.9.18\data\spectrogram\heatflux_no_noise"
# regression_resultとROC曲線の保存先フォルダ
SAVE_PATH = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Subcooling_20_degrees\0.3\2024.9.18\regression_result\AlexNet\heatflux_no_noise"

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# CSVファイルのパス
CSV_FILE_PATH = os.path.join(SAVE_PATH, "train_times.csv")

# CSVファイルにヘッダーを書き込む（もしファイルが存在しなければ）
if not os.path.exists(CSV_FILE_PATH):
    with open(CSV_FILE_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['learning_rate', 'batch_size', 'epochs', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5'])

#######################################################################

#                       データの読み込みと形の変換

#######################################################################

def load_images_from_folder(folder_path):
    x, y = [], []
    print("読み込みスタート")
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            # ファイル名から熱流束の値を取得する
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
    model.add(Dense(1, activation='linear'))  # 回帰のための線形活性化関数
    return model

#######################################################################

#                                実行部

#######################################################################

def main():
    # データのロード
    start_time = time.time()
    x, y = load_images_from_folder(DATA_PATH)
    load_time = time.time() - start_time

    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print(f"データの読み込み時間: {load_time:.2f} 秒")

    # 5分割交差検証の準備
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1

    for train_index, val_index in kf.split(x):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        for lr in LEARNING_RATES:
            for batch_size in BATCH_SIZES:
                for epoch_num in EPOCH_NUMS:
                    print(f"Fold {fold} - Learning Rate: {lr}, Batch Size: {batch_size}, Epochs: {epoch_num}")

                    # AlexNetモデルの作成とコンパイル
                    model = alexnet_regression(input_shape=(224, 224, COLOR_CHANNEL))
                    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
                    
                    # モデルの訓練
                    start_time = time.time()
                    history = model.fit(x_train, y_train, epochs=epoch_num, batch_size=batch_size, verbose=1, 
                                        validation_data=(x_val, y_val))
                    training_time = time.time() - start_time

                    print(f"モデルの訓練時間: {training_time:.2f} 秒")

                    ##############################################################################

                    # CSVファイルを読み込んで、同じパラメータがあるか確認
                    found = False
                    rows = []

                    with open(CSV_FILE_PATH, mode='r', newline='') as file:
                        reader = csv.reader(file)
                        next(reader)  # ヘッダーをスキップ
                        for row in reader:
                            if (float(row[0]) == lr and
                                int(row[1]) == batch_size and
                                int(row[2]) == epoch_num):
                                # 既存のパラメータと一致した場合
                                # 現在のフォールドの列に計算時間を追加
                                row[2 + fold] = str(float(row[2 + fold]) + training_time) if row[2 + fold] else str(training_time)
                                found = True
                            rows.append(row)

                    # 見つからなかった場合、新しい行を追加
                    if not found:
                        # fold_1 から fold_5 までの列を空で初期化
                        new_row = [lr, batch_size, epoch_num] + [''] * 5
                        # 現在のフォールドに計算時間を追加
                        new_row[2 + fold] = str(training_time)
                        rows.append(new_row)

                    # ファイルを上書きして結果を保存
                    with open(CSV_FILE_PATH, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['learning_rate', 'batch_size', 'epochs', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5'])
                        writer.writerows(rows)  # データ行を書き込む

                    ##############################################################################

                    # 検証データに対する予測
                    y_val_pred = model.predict(x_val)

                    # 損失関数のプロットと保存
                    plt.figure()
                    plt.plot(history.history['loss'], label='Training Loss')
                    plt.plot(history.history['val_loss'], label='Validation Loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.title(f'Loss Curve: Fold {fold}, LR={lr}, BS={batch_size}, E={epoch_num}')
                    plt.legend()
                    plt.savefig(os.path.join(SAVE_PATH, f'loss_curve_lr{lr}_bs{batch_size}_ep{epoch_num}_ts{THRESHOLD}_fold{fold}.png'))
                    plt.close()

                    print(f'Loss curve saved for Fold {fold}, LR={lr}, BS={batch_size}, E={epoch_num}')

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
                    plt.savefig(os.path.join(SAVE_PATH, f'regression_analysis_lr{lr}_bs{batch_size}_ep{epoch_num}_ts{THRESHOLD}_snr{0}_fold{fold}.png'))
                    plt.close()

                    print(f'R^2 Score: {r2}')

                    # 予測を分類（閾値を超えているかどうか）
                    y_val_pred_binary = (y_val_pred > THRESHOLD).astype(int)
                    y_val_binary = (y_val > THRESHOLD).astype(int)

                    # ROC曲線の作成
                    fpr, tpr, _thresholds = roc_curve(y_val_binary, y_val_pred_binary)
                    roc_auc = auc(fpr, tpr)

                    # ROC曲線のプロットと保存
                    plt.figure()
                    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
                    plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.0])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'Receiver Operating Characteristic: Fold {fold}, LR={lr}, BS={batch_size}, E={epoch_num}')
                    plt.legend(loc="lower right")
                    plt.savefig(os.path.join(SAVE_PATH, f'roc_curve_lr{lr}_bs{batch_size}_ep{epoch_num}_ts{THRESHOLD}_snr{0}_fold{fold}.png'))
                    plt.close()

        fold += 1

if __name__ == "__main__":
    main()

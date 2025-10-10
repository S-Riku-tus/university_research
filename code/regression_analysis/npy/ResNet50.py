import os
import numpy as np
import time
import csv
from sklearn.metrics import roc_curve, auc, r2_score
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

#######################################################################

#                              変数の指定

#######################################################################

# 訓練時のバッチサイズとエポック数のリスト
BATCH_SIZES = [128]
EPOCH_NUMS = [1000]
# 学習率
LEARNING_RATES = [0.00001, 0.00005]
# 閾値
THRESHOLD = 558802  # TODO: ここの値が、csvファイルで求めた沸騰開始点での熱流束となれば、沸騰-非沸騰分類モデルでのROC曲線が書けるのではないか？

COLOR_CHANNEL = 1

# データフォルダ
DATA_PATH = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Subcooling_20_degrees\0.3\2024.9.18\data\heatflux_npy_channel=1"

if COLOR_CHANNEL == 1:
    # regression_resultとROC曲線の保存先フォルダ
    SAVE_PATH = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Subcooling_20_degrees\0.3\2024.9.18\regression_result\npy\ResNet50\channel=1\best"
elif COLOR_CHANNEL == 3:
    SAVE_PATH = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Subcooling_20_degrees\0.3\2024.9.18\regression_result\npy\ResNet50\channel=3"

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
        if filename.endswith(".npy"):
            # ファイル名から熱流束の値を取得する
            heat_flux = float(filename.split('_')[0])
            data = np.load(os.path.join(folder_path, filename))
            x.append(data)
            y.append(heat_flux)

    x = np.array(x)
    y = np.array(y)
    x = x.astype('float32')
    # x = x / 255.0

    return x, y

#######################################################################

#                         ResNet50のモデルの作成

#######################################################################

def resnet50_regression(input_shape):
    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='linear')(x)  # 回帰のための線形活性化関数
    model = Model(inputs=base_model.input, outputs=x)
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

    # AUCの値を保存しておく
    auc_scores_per_param = []

    for lr in LEARNING_RATES:
        for batch_size in BATCH_SIZES:
            for epoch_num in EPOCH_NUMS:
                print(f"Learning Rate: {lr}, Batch Size: {batch_size}, Epochs: {epoch_num}")

                # 5分割交差検証の準備
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                fold = 1
                auc_scores = []  # 各パラメータセットのAUCスコア

                plt.figure()  # 各パラメータセットごとにROC曲線を描画する

                for train_index, val_index in kf.split(x):
                    x_train, x_val = x[train_index], x[val_index]
                    y_train, y_val = y[train_index], y[val_index]

                    # AlexNetモデルの作成とコンパイル
                    model = resnet50_regression(input_shape=(224, 224, COLOR_CHANNEL))
                    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

                    # モデルの訓練
                    start_time = time.time()
                    history = model.fit(x_train, y_train, epochs=epoch_num, batch_size=batch_size, verbose=1, validation_data=(x_val, y_val))
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
                    plt.savefig(os.path.join(SAVE_PATH, f'regression_analysis_lr{lr}_bs{batch_size}_ep{epoch_num}_ts{THRESHOLD}_fold{fold}.png'))
                    plt.close()

                    print(f'R^2 Score: {r2}')

                    # 予測を分類（閾値を超えているかどうか）
                    y_val_pred_binary = (y_val_pred > THRESHOLD).astype(int)
                    y_val_binary = (y_val > THRESHOLD).astype(int)

                    # ROC曲線の作成
                    fpr, tpr, _thresholds = roc_curve(y_val_binary, y_val_pred_binary)
                    roc_auc = auc(fpr, tpr)
                    auc_scores.append(roc_auc)

                    # 各foldのROC曲線をプロット
                    plt.plot(fpr, tpr, label=f'Fold {fold} (AUC = {roc_auc:.4f})')
                    fold += 1

                # AUCの平均と標準誤差を計算
                mean_auc = np.mean(auc_scores)
                std_auc = np.std(auc_scores, ddof=1)
                se_auc = std_auc / np.sqrt(len(auc_scores))  # 標準誤差

                # 95%信頼区間を計算 (平均 ± 1.96 * 標準誤差)
                confidence_interval = 1.96 * se_auc

                auc_scores_per_param.append(mean_auc)

                # パラメータセットごとのROC曲線のグラフを保存
                plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve: LR={lr}, BS={batch_size}, E={epoch_num}')
                plt.legend(title=f'Mean AUC = {mean_auc:.4f}±{confidence_interval:.4f}%', loc="lower right")
                plt.savefig(os.path.join(SAVE_PATH, f'roc_curve_lr{lr}_bs{batch_size}_ep{epoch_num}_ts{THRESHOLD}.png'))
                plt.close()

                print(f'Combined ROC curve saved for LR={lr}, BS={batch_size}, E={epoch_num}.')


if __name__ == '__main__':
    main()

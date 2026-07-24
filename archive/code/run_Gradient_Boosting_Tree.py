import os
import numpy as np
import time
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, r2_score, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt

#######################################################################

#                              変数の指定

#######################################################################


BATCH_SIZES = [16, 24, 64]
EPOCH_NUMS = [300, 500, 1000]
LEARNING_RATES = [0.3, 0.1, 0.03, 0.01]
# 閾値
THRESHOLD = 558802  # TODO: ここの値が、csvファイルで求めた沸騰開始点での熱流束となれば、沸騰-非沸騰分類モデルでのROC曲線が書けるのではないか？

COLOR_CHANNEL = 1

# データフォルダ
DATA_PATH = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Subcooling_20_degrees\0.3\2024.9.18\data\heatflux_xgboost_npy"

if COLOR_CHANNEL == 1:
    # regression_resultとROC曲線の保存先フォルダ
    SAVE_PATH = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Subcooling_20_degrees\0.3\2024.9.18\regression_result\npy\xgboost\channel=1"
elif COLOR_CHANNEL == 3:
    SAVE_PATH = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Subcooling_20_degrees\0.3\2024.9.18\regression_result\npy\xgboost\channel=3"

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

#######################################################################

#                       データの読み込みと形の変換

#######################################################################

def load_images_from_folder(folder_path):
    x, y = [], []
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            heat_flux = float(filename.split('_')[0])
            data = np.load(os.path.join(folder_path, filename))

            x.append(data)
            y.append(heat_flux)
    x = np.array(x).astype('float32') / 255.0
    y = np.array(y)
    return x, y

#######################################################################

#                                実行部

#######################################################################

# データの読み込み
start_time = time.time()
x, y = load_images_from_folder(DATA_PATH)
x = x.reshape(x.shape[0], -1)  # XGBoost用に1次元配列に変換
load_time = time.time() - start_time
print("x shape:", x.shape)
print("y shape:", y.shape)
print(f"データの読み込み時間: {load_time:.2f} 秒")

# 5分割交差検証
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

for BATCH_SIZE in BATCH_SIZES:
    for lr in LEARNING_RATES:
        for epoch_num in EPOCH_NUMS:
            fold = 1
            for train_index, val_index in kf.split(x):
                print(f"Learning Rate: {lr}, Batch Size: {BATCH_SIZE}, Epochs: {epoch_num} - Fold {fold}")
                x_train, x_val = x[train_index], x[val_index]
                y_train, y_val = y[train_index], y[val_index]

                print("x_train shape:", x_train.shape)
                print("x_val shape:", x_val.shape)
                
                # XGBoost用データセットの作成
                train_data = xgb.DMatrix(x_train, label=y_train)
                eval_data = xgb.DMatrix(x_val, label=y_val)
                
                # XGBoostのパラメータ
                xgb_params = {
                    'learning_rate': lr,
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                }
                evals = [(train_data, 'train'), (eval_data, 'eval')]
                
                # モデルの学習
                start_time = time.time()
                gbm = xgb.train(
                    xgb_params,
                    train_data,
                    num_boost_round=epoch_num,
                    early_stopping_rounds=10,
                    evals=evals,
                )
                train_time = time.time() - start_time

                print(f"モデルの訓練時間: {train_time:.2f} 秒")
                
                # 検証データに対する予測
                y_val_pred = gbm.predict(xgb.DMatrix(x_val))
                
                # # 損失関数のプロットと保存
                # plt.figure()
                # plt.plot(history.history['loss'], label='Training Loss')
                # plt.plot(history.history['val_loss'], label='Validation Loss')
                # plt.xlabel('Epochs')
                # plt.ylabel('Loss')
                # plt.title(f'Loss Curve: Fold {fold}, LR={lr}, BS={BATCH_SIZE}, E={epoch_num}')
                # plt.legend()
                # plt.savefig(os.path.join(SAVE_PATH, f'loss_curve_lr{lr}_bs{BATCH_SIZE}_ep{epoch_num}_ts{THRESHOLD}_fold{fold}.png'))
                # plt.close()

                # 回帰分析の結果のプロット
                plt.figure()
                plt.scatter(y_val, y_val_pred)
                plt.xlabel('True Heat Flux')
                plt.ylabel('Predicted Heat Flux')
                plt.title(f'Regression Analysis: Fold {fold}, LR={lr}, BS={BATCH_SIZE}, E={epoch_num}')
                plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--', linewidth=2)
                plt.tick_params(direction='in', which='both')
                r2 = r2_score(y_val, y_val_pred)
                plt.text(0.5, 0.1, f'R^2 Score: {r2:.4f}', ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
                plt.savefig(os.path.join(SAVE_PATH, f'regression_analysis_lr{lr}_bs{BATCH_SIZE}_ep{epoch_num}_fold_{fold}.png'))
                plt.close()
                
                # 予測を分類（閾値を超えているかどうか）
                y_val_pred_binary = (y_val_pred > THRESHOLD).astype(int)
                y_val_binary = (y_val > THRESHOLD).astype(int)

                # ROC曲線の作成
                fpr, tpr, _ = roc_curve(y_val_binary, y_val_pred_binary)
                roc_auc = auc(fpr, tpr)
                
                # ROC曲線のプロット
                plt.figure()
                plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.4f})')
                plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2)
                # plt.xlim([0.0, 1.0])  # これを入れると、軸と曲線のグラフが重なるので、消したほうが良い？
                # plt.ylim([0.0, 1.0])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve: Fold {fold}, LR={lr}, BS={BATCH_SIZE}, E={epoch_num}')
                plt.legend(loc='lower right')
                plt.savefig(os.path.join(SAVE_PATH, f'roc_curve_lr{lr}_bs{BATCH_SIZE}_ep{epoch_num}_fold_{fold}.png'))
                plt.close()

                fold += 1

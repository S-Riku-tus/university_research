import os
import numpy as np
import time
import csv
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, r2_score
from tensorflow.keras.optimizers import Adam
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
from matplotlib.ticker import ScalarFormatter
from datetime import datetime
from utils.models.regression import RegressionModelMaker
from utils.dataloading.dataloading_and_conversion import DataLoadingConversion
from utils.calculation.calc_r2_auc import AUCorR2Calculation

#######  読んでよかった記事  #######
#######  https://atmarkit.itmedia.co.jp/ait/articles/2112/02/news016.html ・・・アンサンブル学習の手法のいろいろ
#######  https://qiita.com/eureka-ai/items/6c55e3b6d9617ae58afa  ・・・つよつよAIエンジニアになろう
#######  https://speakerdeck.com/moepy_stats/social-implementation-of-machine-learning  ・・・機械学習を「社会実装」するということ
#######  https://speakerdeck.com/shibuiwilliam/ji-jie-xue-xi-woshi-yong-hua-suruenziniaringusukiru  ・・・機械学習を実用化するエンジニアリングスキル



#######################################################################

#                                 入力部

#######################################################################

# 訓練時のバッチサイズとエポック数のリスト
BATCH_SIZES = {"AlexNet": 12, "ResNet50": 12, "VGG16": 32}
EPOCH_NUM = 300
# 学習率
LEARNING_RATE = {"AlexNet": 0.0001, "ResNet50": 0.00005, "VGG16": 0.00005}
# 閾値
threshold_list = [260675.103239721, 353145.749413166, 264418.48987934]  # TODO: ここの値が、csvファイルで求めた沸騰開始点での熱流束となれば、沸騰-非沸騰分類モデルでのROC曲線が書けるのではないか？
THRESHOLD = sum(threshold_list) / len(threshold_list)

# フォールド数
DIVISIONS = 5
# チャンネル数
COLOR_CHANNEL = 1

# アンサンブルの組み合わせ方（0・・単純平均 ｜ 1・・重み付き平均 ｜ 2・・最小値）
ENSEMBLE_METHOD = 1
# ブートストラップサンプルするか
BOOTSTRAP_SAMPLING = False

#  ホワイトノイズ ( = 0)か水流動音 ( = 1)か #
NOISE = 1

# 保存したモデルの重みを用いるかどうか
PREVIOUS_MODEL = False

# ハイパス通した録音データかどうか
HIGHPASS = True

#### データフォルダの設定 ####
noise = "whitenoise" if NOISE == 0 else "waterflow"
highpass = "_highpass_0.5s" if HIGHPASS else ""  ###########################1つって入れた！！！！！！！
highpass = "_sf_20250408_0.5s" if HIGHPASS else "" # 20250408限定
noise = noise + highpass
BASE_DATA_PATH = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy" + "\\" + noise + f"\\channel={COLOR_CHANNEL}\\"
DATA_PATH = [BASE_DATA_PATH + "heatflux_no_noise",
             BASE_DATA_PATH + "heatflux_SNR=0",
             BASE_DATA_PATH + "heatflux_SNR=-4",
             BASE_DATA_PATH + "heatflux_SNR=-8",
             BASE_DATA_PATH + "heatflux_SNR=-12",
             BASE_DATA_PATH + "heatflux_SNR=-16",
             BASE_DATA_PATH + "heatflux_SNR=-20"]

# DATA_PATH = [BASE_DATA_PATH + "tmp_no_noise",]

#### regression_resultとROC曲線の保存先フォルダ ####
BASE_SAVE_PATH = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\regression_result\npy\ensemble" + "\\channel=" + str(COLOR_CHANNEL)

if ENSEMBLE_METHOD == 0:
    SAVE_PATH = os.path.join(BASE_SAVE_PATH, f"average")
elif ENSEMBLE_METHOD == 1:
    SAVE_PATH = os.path.join(BASE_SAVE_PATH, f"weight_average_highpass_20250408\\100%")
elif ENSEMBLE_METHOD == 2:
    SAVE_PATH = os.path.join(BASE_SAVE_PATH, f"min")

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# matplotlibの設定
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

#######################################################################

#                                実行部

#######################################################################

def main():
    for data_path in DATA_PATH:
        # SNR値をパスから抽出
        if "no_noise" in data_path:
            snr_value = "no_noise"
        else:
            snr_value = data_path.split("SNR=")[-1]  # "SNR=" の後の部分を取得

        # データのロード
        start_time = time.time()
        data_loading = DataLoadingConversion()
        x, y = data_loading.load_images_from_folder(data_path)
        load_time = time.time() - start_time

        print("x shape:", x.shape)
        print("y shape:", y.shape)
        print(f"データの読み込み時間: {load_time:.2f} 秒")

        # 分割交差検証の準備
        kf = KFold(n_splits=DIVISIONS, shuffle=True, random_state=42)

        # R^2 scoreとAUCの値を保存するリスト
        r2_scores, alexnet_r2_scores, resnet50_r2_scores, vgg16_r2_scores, ensemble_r2_scores = [], [], [], [], []
        alexnet_auc_scores, resnet50_auc_scores, vgg16_auc_scores, ensemble_auc_scores  = [], [], [], []
        accuracy_scores, precision_scores, recall_scores, f1_scores, auc_scores = [], [], [], [], []

        plt.figure()  # 各パラメータセットごとにROC曲線を描画する

        fold = 1
        output_file = os.path.join(SAVE_PATH, f'validation_results_{snr_value}.txt')
        # テキストファイルの作成・追記モードで開く
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write("K-fold Cross-Validation Results\n")
            f.write("="*30 + "\n")

            for train_index, val_index in kf.split(x):
                x_train, x_val = x[train_index], x[val_index]
                y_train, y_val = y[train_index], y[val_index]

                # 各モデルの作成
                regressionmodelmaker = RegressionModelMaker((224, 224, COLOR_CHANNEL))
                alexnet_model = regressionmodelmaker.alexnet()
                resnet50_model = regressionmodelmaker.resnet50()
                vgg16_model = regressionmodelmaker.vgg16()

                # モデルのコンパイル
                alexnet_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE['AlexNet']), loss='mean_squared_error')
                resnet50_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE['ResNet50']), loss='mean_squared_error')
                vgg16_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE['VGG16']), loss='mean_squared_error')

                if BOOTSTRAP_SAMPLING:
                    # 各モデルに対して異なるブートストラップサンプルを作成
                    x_train, y_train = resample(x_train, y_train, random_state=fold * 1)
                    x_train, y_train = resample(x_train, y_train, random_state=fold * 2)
                    x_train, y_train = resample(x_train, y_train, random_state=fold * 3)

                # 各モデルの訓練
                save_dir_1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(BASE_SAVE_PATH))))
                # save_dir = os.path.join(save_dir_1, f"all_weights\\{snr_value}")
                save_dir = os.path.join(save_dir_1, f"all_weights_20250408\\{snr_value}")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                if not PREVIOUS_MODEL:
                    print(f"AlexNet Model Start : Fold {fold} / {DIVISIONS}")
                    alexnet_model.fit(x_train, y_train, batch_size=BATCH_SIZES["AlexNet"], epochs=EPOCH_NUM, verbose=1)
                    print(f"ResNet50 Model Start : Fold {fold} / {DIVISIONS}")
                    resnet50_model.fit(x_train, y_train, batch_size=BATCH_SIZES['ResNet50'], epochs=EPOCH_NUM, verbose=1)
                    print(f"VGG16 Model Start : Fold {fold} / {DIVISIONS}")
                    vgg16_model.fit(x_train, y_train, batch_size=BATCH_SIZES['VGG16'], epochs=EPOCH_NUM, verbose=1)

                    # モデルの重みの保存
                    alexnet_weights_path = os.path.join(save_dir, f"AlexNet_fold{fold}_{snr_value}.h5")
                    alexnet_model.save_weights(alexnet_weights_path)
                    resnet50_weights_path = os.path.join(save_dir, f"resnet50_fold{fold}_{snr_value}.h5")
                    resnet50_model.save_weights(resnet50_weights_path)
                    vgg16_weights_path = os.path.join(save_dir, f"vgg16_fold{fold}_{snr_value}.h5")
                    vgg16_model.save_weights(vgg16_weights_path)
                else:
                    alexnet_model.load_weights(os.path.join(save_dir, f"AlexNet_fold{fold}_{snr_value}.h5"))
                    resnet50_model.load_weights(os.path.join(save_dir, f"resnet50_fold{fold}_{snr_value}.h5"))
                    vgg16_model.load_weights(os.path.join(save_dir, f"vgg16_fold{fold}_{snr_value}.h5"))

                # 各モデルの予測
                alexnet_pred = alexnet_model.predict(x_val)
                resnet50_pred = resnet50_model.predict(x_val)
                vgg16_pred = vgg16_model.predict(x_val)
                predictions = [alexnet_pred, resnet50_pred, vgg16_pred]

                # 回帰分析の結果とR^2 scoreの計算
                calc = AUCorR2Calculation()
                alexnet_error = calc.calc_r2_score(y_val, alexnet_pred, alexnet_r2_scores, fold, "AlexNet")
                resnet50_error = calc.calc_r2_score(y_val, resnet50_pred, resnet50_r2_scores, fold, "ResNet50")
                vgg16_error = calc.calc_r2_score(y_val, vgg16_pred, vgg16_r2_scores, fold, "VGG16")

                #######################################################################
                ################## 単純平均か重み付き平均か最小値を選択 ##################

                if ENSEMBLE_METHOD == 0:
                    # アンサンブルの単純平均
                    ensemble_pred = (alexnet_pred + resnet50_pred + vgg16_pred) / 3
                elif ENSEMBLE_METHOD == 1:
                    # アンサンブルの重み付き平均
                    # 重みの計算（誤差率の逆数を使用)
                    errors = [alexnet_error, resnet50_error, vgg16_error]
                    weights = [1 / error for error in errors]
                    weights = weights / np.sum(weights)  # 正規化して合計を1に
                    ensemble_pred = sum(w * pred for w, pred in zip(weights, predictions))
                elif ENSEMBLE_METHOD == 2:
                    # アンサンブルの最小値を選択
                    ensemble_pred = np.min(predictions, axis=0)  # 各サンプルごとに最小値を選択

                #######################################################################

                # バイナリラベルの生成
                y_binary = (ensemble_pred >= THRESHOLD).astype(int)
                y_val_binary = (y_val >= THRESHOLD).astype(int)

                # 評価指標を計算
                r2 = r2_score(y_val, ensemble_pred)
                accuracy = accuracy_score(y_val_binary, y_binary)
                precision = precision_score(y_val_binary, y_binary)
                recall = recall_score(y_val_binary, y_binary)
                f1 = f1_score(y_val_binary, y_binary)

                fpr, tpr, _ = roc_curve(y_val_binary, y_binary)
                roc = auc(fpr, tpr)

                # 評価指標をリストに保存
                r2_scores.append(r2)
                accuracy_scores.append(accuracy)
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
                auc_scores.append(roc)

                # 結果をファイルに追記
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"Recorded at: {current_time}\n")
                f.write(f"Fold {fold} Results:\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1 Score: {f1:.4f}\n")
                f.write(f"R² Score: {r2:.4f}\n")
                f.write(f"AUC Score: {roc:.4f}\n")
                f.write("-"*30 + "\n")

                ensemble_error = calc.calc_r2_score(y_val, ensemble_pred, ensemble_r2_scores, fold, "Ensemble")
                ensemble_r2 = 1 - ensemble_error

                # AUC=1の熱流束を求める
                thresholds = np.linspace(min(ensemble_pred), max(ensemble_pred), 100000)
                optimal_threshold = None

                for threshold in thresholds:
                    threshold = threshold[0]
                    y_binary = (ensemble_pred >= threshold).astype(int)
                    y_binary = y_binary.flatten()  # (245, 1) -> (245,)

                    y_val_true_binary = (y_val >= threshold).astype(int)
                    y_val_false_binary = (y_val < threshold).astype(int)
                    tpr = np.sum((y_binary == 1) & (y_val >= threshold)) / np.sum(y_val >= threshold)
                    fpr = np.sum((y_binary == 1) & (y_val < threshold)) / np.sum(y_val < threshold)

                    if fpr == 0 and tpr == 1:  # 100%分類性能の条件
                        optimal_threshold = threshold
                        break

                # 各分割の回帰分析結果をプロット
                plt.figure(figsize=(12, 9))
                plt.scatter(y_val, ensemble_pred, label='Data', alpha=0.6)
                plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
                # plt.title(f'Regression Analysis(SNR = {snr_value}) - Split {split_idx}', fontsize=30)
                plt.xlabel('True Heat Flux MW/m²', fontsize=40)
                plt.ylabel('Predicted Heat Flux MW/m²', fontsize=40)
                # オーダー表記を非表示にする
                ax = plt.gca()
                ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style="plain", axis="both")  # 科学表記を無効にする
                ax.xaxis.offsetText.set_visible(False)  # デフォルトのオフセットを非表示
                ax.yaxis.offsetText.set_visible(False)
                ax.xaxis.get_offset_text().set_text('×10⁶')  # x軸の単位を明示
                ax.yaxis.get_offset_text().set_text('×10⁶')  # y軸の単位を明示

                # 閾値を描画
                plt.axvline(x=THRESHOLD, color='k', linestyle='dashed', label=f'Threshold (Boiling Point):\n{THRESHOLD / 1e6:.4f} MW/m²')
                plt.axhline(y=THRESHOLD, color='k', linestyle='dashed', label=None)


                #XXX: これは修論でやる予定でやんす
                # # 閾値を水平線として描画 -> 2024/12/24/02:02.02 なんでこんな時間に、、きつい
                # if optimal_threshold is not None:
                #     plt.axhline(optimal_threshold, color='gray', linestyle='dashed',
                #                 label=f'AUC=1 Threshold: {optimal_threshold  / 1e6:.2f}')
                #     print("AUC=1の閾値を描画しました。")


                # # 100%分類性能の評価
                # blocks = []
                # current_block = [0]

                # for i in range(1, len(y_val)):
                #     if y_val[i] == y_val[i - 1]:
                #         current_block.append(i)
                #     else:
                #         blocks.append(current_block)
                #         current_block = [i]
                # blocks.append(current_block)

                # # 塊の先頭の y_val 値を取得し、blocks を小さい順に並べ替える
                # block_labels = [y_val[block[0]] for block in blocks]  # 各ブロックの代表ラベル
                # sorted_indices = np.argsort(block_labels)  # 小さい順のインデックス
                # sorted_blocks = [blocks[i] for i in sorted_indices]  # 並べ替えた blocks

                # # print("並べ替え後の blocks:", sorted_blocks)

                # # ensemble_pred の順序も blocks に合わせて並べ替え
                # sorted_ensemble_pred = np.concatenate([ensemble_pred[block] for block in sorted_blocks])

                # # 並べ替え後の y_val と ensemble_pred を確認
                # sorted_y_val = np.concatenate([y_val[block] for block in sorted_blocks])

                # # 1. すべての塊のフラグと最小値を計算
                # block_flags = []  # 各塊がすべて閾値を超えているか否かを保持
                # block_min_values = []  # 各塊の最小値を保持

                # for block in sorted_blocks:
                #     block_preds = ensemble_pred[block]  # 各塊のデータ
                #     block_preds = block_preds.flatten()
                #     block_flags.append(np.all(block_preds > THRESHOLD))  # すべて閾値を超えているか
                #     block_min_values.append(np.min(block_preds))  # 塊内の最小値
                #     print(block_preds)


                # print(block_flags)
                # print(block_min_values)
                # # 2. 条件を満たす塊を探索
                # for i, min_value in enumerate(block_min_values):
                #     if not block_flags[i]:  # 現在の塊が条件を満たさない場合はスキップ
                #         continue

                #     # 現在の塊より上のすべての塊が条件を満たしているか確認
                #     if all(block_flags[i:]):  # 以降のすべてが条件を満たしている場合
                #         plt.axvline(x=min_value, linestyle='dashdot',
                #                     color='green', label=f"100% Classification Threshold:\n{min_value / 1e6:.4f} MW/m²")
                #         print("100%分類の閾値を描画しました。")
                #         break


                # 100%分類性能の評価
                blocks = []
                current_block = [0]

                for i in range(1, len(y_val)):
                    if y_val[i] == y_val[i - 1]:
                        current_block.append(i)
                    else:
                        blocks.append(current_block)
                        current_block = [i]
                blocks.append(current_block)

                # 塊の先頭の y_val 値を取得し、blocks を小さい順に並べ替える
                block_labels = [y_val[block[0]] for block in blocks]  # 各ブロックの代表ラベル
                sorted_indices = np.argsort(block_labels)  # 小さい順のインデックス
                sorted_blocks = [blocks[i] for i in sorted_indices]  # 並べ替えた blocks

                # ensemble_pred の順序も blocks に合わせて並べ替え
                sorted_ensemble_pred = np.concatenate([ensemble_pred[block] for block in sorted_blocks])

                # 並べ替え後の y_val と ensemble_pred を確認
                sorted_y_val = np.concatenate([y_val[block] for block in sorted_blocks])

                # 1. すべての塊のフラグと最小値を計算
                block_flags = []  # 各塊がすべて閾値を超えているか否かを保持
                block_val_values = []  # 各塊のval値を保持
                block_min_preds = []  # 各塊の最小の予測値を保持

                for block in sorted_blocks:
                    block_preds = ensemble_pred[block]  # 各塊のデータ
                    block_preds = block_preds.flatten()
                    block_flags.append(np.all(block_preds > THRESHOLD))  # すべて閾値を超えているか
                    block_val_values.append(y_val[block[0]])  # 塊内の代表val値を保持
                    block_min_preds.append(np.min(block_preds))  # 塊内の最小予測値を保持

                # 各塊の最小予測値をプリント
                for i, min_pred in enumerate(block_min_preds):
                    print(f"塊 {i + 1} の最小予測値: {min_pred}")

                print(block_flags)
                print(block_val_values)


                # 2. 条件を満たす塊を探索
                for i, val_value in enumerate(block_val_values):

                    # if val_value < THRESHOLD:
                    #     continue

                    if not block_flags[i]:  # 現在の塊が条件を満たさない場合はスキップ
                        continue

                    # 現在の塊より上のすべての塊が条件を満たしているか確認
                    if all(block_flags[i:]):  # 以降のすべてが条件を満たしている場合
                        plt.axvline(x=val_value, linestyle='dashdot',
                                    color='green', label=f"100% Classification Threshold:\n{val_value / 1e6:.4f} MW/m²")
                        print("100%分類の閾値を描画しました。")
                        break



                plt.text(0.75, 0.1, f'R² Score: {r2:.4f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=40)    
                legend = plt.legend(loc=(0.007, 0.72), fontsize=20)
                frame = legend.get_frame()
                frame.set_edgecolor('black')
                frame.set_linewidth(0.7)
                frame.set_alpha(None)  

                # 軸ラベルを 10⁶ の単位で調整
                xticks = np.arange(0, 1.3e6, step=2e5)  # x軸の範囲を1.2e6までに設定
                yticks = np.arange(0, 1.3e6, step=2e5)  # y軸の範囲を1.2e6までに設定
                plt.xticks(xticks, fontsize=24, labels=[f'{x/1e6:.1f}' for x in xticks])  # ラベルを10⁶単位で表示
                plt.yticks(yticks, fontsize=24, labels=[f'{x/1e6:.1f}' for x in yticks])  # 同様にy軸も調整
                plt.tick_params(axis='both', labelsize=30)

                # 分割ごとの結果を保存
                output_path = os.path.join(SAVE_PATH, f't_regression_split_{snr_value}_{fold}.png')
                plt.savefig(output_path)
                plt.close()

                fold += 1

            # 各評価指標の標準誤差を計算
            r2_scores = np.array(r2_scores)
            accuracy_scores = np.array(accuracy_scores)
            precision_scores = np.array(precision_scores)
            recall_scores = np.array(recall_scores)
            f1_scores = np.array(f1_scores)
            auc_scores = np.array(auc_scores)

            r2_std_error = np.std(r2_scores) / np.sqrt(len(r2_scores))
            accuracy_std_error = np.std(accuracy_scores) / np.sqrt(len(accuracy_scores))
            precision_std_error = np.std(precision_scores) / np.sqrt(len(precision_scores))
            recall_std_error = np.std(recall_scores) / np.sqrt(len(recall_scores))
            f1_std_error = np.std(f1_scores) / np.sqrt(len(f1_scores))
            auc_std_error = np.std(auc_scores) / np.sqrt(len(auc_scores))

            # 平均結果をファイルに追記
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Recorded at: {current_time}\n")
            f.write("\nAverage Results:\n")
            f.write(f"Average Accuracy: {np.mean(accuracy_scores):.4f} ± {accuracy_std_error:.4f}\n")
            f.write(f"Average Precision: {np.mean(precision_scores):.4f} ± {precision_std_error:.4f}\n")
            f.write(f"Average Recall: {np.mean(recall_scores):.4f} ± {recall_std_error:.4f}\n")
            f.write(f"Average F1 Score: {np.mean(f1_scores):.4f} ± {f1_std_error:.4f}\n")
            f.write(f"Average R² Score: {np.mean(r2_scores):.4f} ± {r2_std_error:.4f}\n")
            f.write(f"Average AUC Score: {np.mean(auc_scores):.4f} ± {auc_std_error:.4f}\n")
            f.write("="*30 + "\n\n")


if __name__ == '__main__':
    main()

import os
import numpy as np
import time
import csv
import joblib
from pathlib import Path
from sklearn.model_selection import KFold
import gc
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, roc_curve, auc, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import ScalarFormatter
from datetime import datetime
from utils.models.regression.base_regression import RegressionModelMaker
from utils.models.regression.new_regression import RegressionModelMaker1106
from utils.models.regression.swin_transformer import SwinTransformerModelMaker
from utils.dataloading.dataloading_and_conversion import DataLoadingConversion
from utils.calculation.calc_r2_auc import AUCorR2Calculation

#######  読んでよかった記事  #######
#######  https://atmarkit.itmedia.co.jp/ait/articles/2112/02/news016.html ・・・アンサンブル学習の手法のいろいろ
#######  https://qiita.com/eureka-ai/items/6c55e3b6d9617ae58afa  ・・・つよつよAIエンジニアになろう
#######  https://speakerdeck.com/moepy_stats/social-implementation-of-machine-learning  ・・・機械学習を「社会実装」するということ
#######  https://speakerdeck.com/shibuiwilliam/ji-jie-xue-xi-woshi-yong-hua-suruenziniaringusukiru  ・・・機械学習を実用化するエンジニアリングスキル



#######################################################################

#                              変数の指定

#######################################################################

# 訓練時のバッチサイズとエポック数のリスト
# BATCH_SIZES = {"AlexNet": 12, "ResNet50": 12, "VGG16": 16}  # これは卒論時の値
BATCH_SIZES_ALL = [48]
EPOCH_NUM = 200
# 学習率
# 今のところ一番良いやつ
LEARNING_RATE_ALL = [0.005]

# 閾値
# threshold_list = [260675.103239721, 353145.749413166, 264418.48987934]  # TODO: ここの値が、csvファイルで求めた沸騰開始点での熱流束となれば、沸騰-非沸騰分類モデルでのROC曲線が書けるのではないか？
threshold_list = [275174.6641]
THRESHOLD = sum(threshold_list) / len(threshold_list)

# パラメータをループさせて検証するかどうか
FLG_ROOP = True

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

SAVE_DATE = "20260128"
# SAVE_DATE = "cnn+tra系_tune"

# 使用するデータの日付
DATA_DATE = "20251219"

# 周波数解析のパラメータ
CHUNK = 1
max_freq_hz = "maxfreq=22kHz"
# max_freq_hz = "maxfreq=15kHz"
# max_freq_hz = "maxfreq=10kHz"
# max_freq_hz = "maxfreq=5kHz"
# max_freq_hz = "maxfreq=3kHz"
# max_freq_hz = "maxfreq=2kHz"
# max_freq_hz = ["maxfreq=22kHz",
#                "maxfreq=3kHz"]

#### データフォルダの設定 ####
noise = "whitenoise" if NOISE == 0 else "waterflow"
highpass = f"_{DATA_DATE}_{CHUNK}s"
noise = noise + highpass

BASE_PATH = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2025.07.09_0.3_1"
base_path = Path(BASE_PATH)

BASE_DATA_PATH = base_path / "data" / "npy" / noise / str(max_freq_hz)
DATA_PATH = [
    BASE_DATA_PATH / "heatflux_no_noise",
    # BASE_DATA_PATH / "heatflux_SNR=0",
    # BASE_DATA_PATH / "heatflux_SNR=-4",
    # BASE_DATA_PATH / "heatflux_SNR=-8",
    # BASE_DATA_PATH / "heatflux_SNR=-12",
    # BASE_DATA_PATH / "heatflux_SNR=-16",
    # BASE_DATA_PATH / "heatflux_SNR=-20"
]

#### regression_resultとROC曲線の保存先フォルダ ####
BASE_SAVE_PATH = base_path / "regression_result" / "npy" / "ensemble"

# matplotlibの設定
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


#######################################################################

#                    回帰分析とROC曲線の描画する関数

#######################################################################

def save_combined_r2_auc(valuation, valuation_indicators_scores, error_list, model_name):
    """
    5分割交差検証のROC曲線の平均AUCと95%信頼区間を計算し、グラフを保存
    """
    mean = np.mean(valuation_indicators_scores)
    std_valuation = np.std(valuation_indicators_scores, ddof=1)
    se_valuation = std_valuation / np.sqrt(len(valuation_indicators_scores))
    confidence_interval = 1.96 * se_valuation
    
    # 信頼区間を計算
    upper_bound = mean + confidence_interval
    lower_bound = mean - confidence_interval
    error_list.append(upper_bound - mean)
    print('Error list size : ', len(error_list))
    print(f'upper_bound : {upper_bound} | Lower bound : {lower_bound} | Confidence interval : {confidence_interval:.4f}')
    if valuation == "r2":
        print(f'Model : {model_name} | Combined Regression saved for Mean R^2 score = {mean:.4f} ± {confidence_interval:.4f}%')
    else:
        print(f'Model : {model_name} | Combined ROC curve saved for Mean AUC = {mean:.4f} ± {confidence_interval:.4f}%')

    return f'{mean:.4f} ± {confidence_interval:.4f}', mean, error_list

def plot_model_variables(valuation, str_valuation_list, valuation_list, error_list, epochs, save_path, snr_value):
    if valuation == "r2":
        print("R^2 score List:", valuation_list)
        print("String R^2 score List:", str_valuation_list)
        print("R^2 score List Length:", len(valuation_list))
        print("String R^2 score List Length:", len(str_valuation_list))

        plt.figure(figsize=(8, 6))
        # models = ['AlexNet', 'ResNet50', 'VGG16', 'Ensemble']
        models = ['AlexNet', 'Alex+Tf AP', 'Alex+Tf GAP', 'Ensemble']
        
        # AUCリストの長さをチェック
        if len(valuation_list) != len(models) or len(str_valuation_list) != len(models):
            raise ValueError("Length of r2_list or str_r2_list does not match the number of models.")

        # プロットの処理
        bars = plt.bar(models, valuation_list, color=['c', 'cadetblue', 'skyblue', 'dodgerblue'], yerr=error_list, capsize=5, width=0.5)
        plt.ylim(0.0, 1.05)
        # plt.title(f'Comparison of R^2 score for each model (Epochs: {epochs}, SNR: {snr_value})', fontsize=15, pad=8)
        plt.ylabel('R² Score', fontsize=23)

        # 軸のフォントサイズ
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=18)

        # 各棒の中に精度の数値を表示
        for i, (auc, str_auc) in enumerate(zip(valuation_list, str_valuation_list)):
            plt.text(i, 0.03, str_auc, ha='center', va='bottom', fontsize=25, color='black', rotation=90)

        # 画像を保存
        output_path_base = os.path.join(save_path, "roc_results")
        if not os.path.exists(output_path_base):
            os.makedirs(output_path_base)
        # output_path = os.path.join(output_path_base, f'Comparison_of_R^2 score_ep{epochs}_SNR={snr_value}.png')
        output_path = os.path.join(output_path_base, f'R^2_ep{epochs}_SNR={snr_value}.png')
        plt.savefig(output_path)
        plt.close()
    else:
        print("AUC List:", valuation_list)
        print("String AUC List:", str_valuation_list)
        print("AUC List Length:", len(valuation_list))
        print("String AUC List Length:", len(str_valuation_list))

        plt.figure(figsize=(8, 6))
        # models = ['AlexNet', 'ResNet50', 'VGG16', 'Ensemble']
        models = ['AlexNet', 'Alex+Tf AP', 'Alex+Tf GAP', 'Ensemble']
        
        # AUCリストの長さをチェック
        if len(valuation_list) != len(models) or len(str_valuation_list) != len(models):
            raise ValueError("Length of auc_list or str_auc_list does not match the number of models.")

        # プロットの処理
        bars = plt.bar(models, valuation_list, color=['c', 'cadetblue', 'skyblue', 'dodgerblue'], yerr=error_list, capsize=5, width=0.5)
        plt.ylim(0.0, 1.05)
        # plt.title(f'Comparison of AUC for each model (Epochs: {epochs}, SNR: {snr_value})', fontsize=16, pad=8)
        plt.ylabel('AUC', fontsize=20)

        # 軸のフォントサイズ
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=18)

        # 各棒の中に精度の数値を表示
        for i, (auc, str_auc) in enumerate(zip(valuation_list, str_valuation_list)):
            plt.text(i, 0.03, str_auc, ha='center', va='bottom', fontsize=25, color='black', rotation=90)

        # 画像を保存
        output_path_base = os.path.join(save_path, "roc_results")
        if not os.path.exists(output_path_base):
            os.makedirs(output_path_base)
        # output_path = os.path.join(output_path_base, f'Comparison_of_AUC_ep{epochs}_SNR={snr_value}.png')
        output_path = os.path.join(output_path_base, f'AUC_ep{epochs}_SNR={snr_value}.png')
        plt.savefig(output_path)
        plt.close()


# 損失関数
def plot_loss_history(history, epochs, model_name, fold, save_path, snr_value):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    # もし検証データセットの損失も記録している場合（model.fitのvalidation_data引数を使用した場合）
    # if 'val_loss' in history.history:
    #     plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss History (Epochs: {epochs}, Fold: {fold}, SNR: {snr_value})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.legend()
    plt.grid(True)

    output_path_base = os.path.join(save_path, "loss_histories")
    if not os.path.exists(output_path_base):
        os.makedirs(output_path_base, exist_ok=True) # exist_ok=True を追加

    # output_path = os.path.join(output_path_base, f'loss_history_{model_name}_ep{epochs}_fold{fold}_SNR={snr_value}.png')
    output_path = os.path.join(output_path_base, f'ep{epochs}_fold{fold}_SNR={snr_value}.png')
    plt.savefig(output_path)
    plt.close()

def _mean_se(arr):
    arr = np.asarray(arr, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


#######################################################################

#                                実行部

#######################################################################

def main():
    for data_path in DATA_PATH:
        # --- メモリ管理: 前のループのゴミを掃除 ---
        K.clear_session()
        gc.collect()

        # SNR値をパスから抽出
        if "no_noise" in str(data_path):
            snr_value = "no_noise"
        else:
            snr_value = str(data_path).split("SNR=")[-1]  # "SNR=" の後の部分を取得

        print(f"\n{'='*40}")
        print(f"データセット読み込み開始: {snr_value}")

        # データのロード
        start_time = time.time()
        data_loading = DataLoadingConversion()
        if COLOR_CHANNEL == 1:
            x, y = data_loading.load_npy_data(data_path)
        else:
            x, y = data_loading.load_image_data(data_path)
        load_time = time.time() - start_time

        print("x shape:", x.shape)
        print("y shape:", y.shape)
        print(f"データの読み込み時間: {load_time:.2f} 秒")

        # X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        for all_bs in BATCH_SIZES_ALL:
            for all_lr in LEARNING_RATE_ALL:
                # LEARNING_RATE = {"AlexNet": 0.05, "ResNet50": 0.0005, "VGG16": 0.0005}
                # LEARNING_RATE = {"AlexNet": 0.001, "ResNet50": 0.005, "VGG16": 0.0001}
                # BATCH_SIZES = {"AlexNet": 12, "ResNet50": 12, "VGG16": 12}
                LEARNING_RATE = {"AlexNet": all_lr, "ResNet50": all_lr, "VGG16": all_lr}
                BATCH_SIZES = {"AlexNet": all_bs, "ResNet50": all_bs, "VGG16": all_bs}

                noise_dir_name = os.path.basename(data_path)

                if ENSEMBLE_METHOD == 0:
                    SAVE_PATH = os.path.join(BASE_SAVE_PATH, noise_dir_name, max_freq_hz, f"{SAVE_DATE}_ep{EPOCH_NUM}_chu{CHUNK}", f"average")
                elif ENSEMBLE_METHOD == 1:
                    # SAVE_PATH = os.path.join(BASE_SAVE_PATH, f"{SAVE_DATE}_ep{EPOCH_NUM}_chu{CHUNK}_bsAl{bsare}_Re{bsres}_Vg{bsvgg}_lrAl{lrale}_Re{lrres}_Vg{lrvgg}", f"weight_average")
                    # SAVE_PATH = os.path.join(BASE_SAVE_PATH, noise_dir_name, f"pre_{SAVE_DATE}_ep{EPOCH_NUM}_bsAl{bsare}_Re{bsres}_Vg{bsvgg}_lrAl{lrale}_Re{lrres}_Vg{lrvgg}")
                    SAVE_PATH = os.path.join(BASE_SAVE_PATH, noise_dir_name, max_freq_hz, f"pre_{SAVE_DATE}_ep{EPOCH_NUM}_bs{BATCH_SIZES['AlexNet']}_lr{LEARNING_RATE['AlexNet']}")
                elif ENSEMBLE_METHOD == 2:
                    SAVE_PATH = os.path.join(BASE_SAVE_PATH, noise_dir_name, max_freq_hz, f"{SAVE_DATE}_ep{EPOCH_NUM}_chu{CHUNK}", f"min")

                if not os.path.exists(SAVE_PATH):
                    os.makedirs(SAVE_PATH, exist_ok=True)

                # 分割交差検証の準備
                kf = KFold(n_splits=DIVISIONS, shuffle=True, random_state=42)

                # R^2 scoreとAUCの値を保存するリスト
                alexnet_r2_scores, resnet50_r2_scores, vgg16_r2_scores, ensemble_r2_scores = [], [], [], []
                alexnet_auc_scores, resnet50_auc_scores, vgg16_auc_scores, ensemble_auc_scores  = [], [], [], []
                accuracy_scores, precision_scores, recall_scores, f1_scores, auc_scores = [], [], [], [], []
                r2_scores, r2_high_scores = [], []
                rmse_all_scores, rmse_high_scores = [], []
                mae_all_scores,  mae_high_scores  = [], []

                # R^2 scoreとAUCエラーバー用のリスト
                r2_error_list = []
                auc_error_list = []

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


                        # 1. スケーラーを訓練データ（y_train）にのみ適合させる
                        scaler = MinMaxScaler()
                        # reshape(-1, 1) は、yをscikit-learnが要求する2D配列に変換するため
                        y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))

                        # 2. 訓練データで学習したスケーラーを検証データ（y_val）に適用する
                        y_val_scaled = scaler.transform(y_val.reshape(-1, 1))

                        # === Random Forest用のデータ加工 ===
                        # 画像データ (N, 224, 224, 1) を (N, 50176) に平坦化する
                        # 224 * 224 * 1 = 50,176特徴量
                        x_train_flat = x_train.reshape(x_train.shape[0], -1)
                        x_val_flat = x_val.reshape(x_val.shape[0], -1)

                        # 各モデルの作成
                        regressionmodelmaker = RegressionModelMaker((224, 224, COLOR_CHANNEL))
                
                        # alexnet_model = regressionmodelmaker.alexnet()
                        resnet50_model = regressionmodelmaker.cnn_transformer_v1()
                        vgg16_model = regressionmodelmaker.cnn_transformer_v2()
                        rf_model = regressionmodelmaker.random_forest()

                        # alexnet_model.compile(optimizer=SGD(learning_rate=LEARNING_RATE['AlexNet'], momentum=0.9, clipnorm=1.0), loss='mean_squared_error')
                        resnet50_model.compile(optimizer=SGD(learning_rate=LEARNING_RATE['ResNet50'], momentum=0.9, clipnorm=1.0), loss='mean_squared_error')
                        vgg16_model.compile(optimizer=SGD(learning_rate=LEARNING_RATE['VGG16'], momentum=0.9, clipnorm=1.0), loss='mean_squared_error')

                        if BOOTSTRAP_SAMPLING:
                            # 各モデルに対して異なるブートストラップサンプルを作成
                            x_train, y_train = resample(x_train, y_train, random_state=fold * 1)
                            x_train, y_train = resample(x_train, y_train, random_state=fold * 2)
                            x_train, y_train = resample(x_train, y_train, random_state=fold * 3)

                        # 各モデルの訓練
                        save_dir_1 = os.path.dirname(os.path.dirname(os.path.dirname(BASE_SAVE_PATH)))
                        save_dir = os.path.join(save_dir_1, "all_weights", f"{SAVE_DATE}_epoch{EPOCH_NUM}_chunk{CHUNK}", f"{snr_value}")
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        if not PREVIOUS_MODEL:
                            # print(f"AlexNet Model Start : Fold {fold} / {DIVISIONS}")
                            # alexnet_history = alexnet_model.fit(x_train, y_train_scaled, batch_size=BATCH_SIZES["AlexNet"], epochs=EPOCH_NUM, verbose=1)
                            print(f"ResNet50 Model Start : Fold {fold} / {DIVISIONS}")
                            resnet50_history = resnet50_model.fit(x_train, y_train_scaled, batch_size=BATCH_SIZES['ResNet50'], epochs=EPOCH_NUM, verbose=1)
                            print(f"VGG16 Model Start : Fold {fold} / {DIVISIONS}")
                            vgg16_history = vgg16_model.fit(x_train, y_train_scaled, batch_size=BATCH_SIZES['VGG16'], epochs=EPOCH_NUM, verbose=1)
                            print(f"Random Forest Model Start : Fold {fold}")
                            # RFは平坦化したデータ(x_train_flat)と、1次元化したラベル(ravel)を使うのが一般的
                            rf_model.fit(x_train_flat, y_train_scaled.ravel())

                            # モデルの重みの保存
                            # alexnet_weights_path = os.path.join(save_dir, f"AlexNet_fold{fold}_{snr_value}.weights.h5")
                            # alexnet_model.save_weights(alexnet_weights_path)
                            # resnet50_weights_path = os.path.join(save_dir, f"resnet50_fold{fold}_{snr_value}.weights.h5")
                            # resnet50_model.save_weights(resnet50_weights_path)
                            # vgg16_weights_path = os.path.join(save_dir, f"vgg16_fold{fold}_{snr_value}.weights.h5")
                            # vgg16_model.save_weights(vgg16_weights_path)

                            # rf_save_path = os.path.join(save_dir, f"RandomForest_fold{fold}_{snr_value}.joblib")
                            # joblib.dump(alexnet_model, rf_save_path)
                        else:
                            # alexnet_model.load_weights(os.path.join(save_dir, f"AlexNet_fold{fold}_{snr_value}.weights.h5"))
                            resnet50_model.load_weights(os.path.join(save_dir, f"resnet50_fold{fold}_{snr_value}.weights.h5"))
                            vgg16_model.load_weights(os.path.join(save_dir, f"vgg16_fold{fold}_{snr_value}.weights.h5"))

                        # 各モデルの損失履歴をプロットして保存
                        if not PREVIOUS_MODEL: # 以前のモデルをロードした場合は損失履歴がないためスキップ
                            # plot_loss_history(alexnet_history, EPOCH_NUM, "AlexNet", fold, SAVE_PATH, snr_value)
                            plot_loss_history(resnet50_history, EPOCH_NUM, "ResNet50", fold, SAVE_PATH, snr_value)
                            plot_loss_history(vgg16_history, EPOCH_NUM, "VGG16", fold, SAVE_PATH, snr_value)

                        # 各モデルの予測
                        # alexnet_pred = alexnet_model.predict(x_val)
                        resnet50_pred = resnet50_model.predict(x_val)
                        vgg16_pred = vgg16_model.predict(x_val)
                        rf_pred = rf_model.predict(x_val_flat).reshape(-1, 1)
                        # predictions = [alexnet_pred, resnet50_pred, vgg16_pred]
                        predictions = [rf_pred, resnet50_pred, vgg16_pred]

                        # 予測結果と正解データを元のスケールに戻す
                        # alexnet_pred = scaler.inverse_transform(alexnet_pred)
                        resnet50_pred = scaler.inverse_transform(resnet50_pred)
                        vgg16_pred = scaler.inverse_transform(vgg16_pred)
                        rf_pred = scaler.inverse_transform(rf_pred)
                        # predictions = [alexnet_pred, resnet50_pred, vgg16_pred]
                        predictions = [rf_pred, resnet50_pred, vgg16_pred]

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
                            weights = []
                            for error in errors:
                                if np.isinf(error) or np.isnan(error) or error <= 0:
                                    weights.append(1e-6) # 非常に小さい重みを与える
                                else:
                                    weights.append(1 / error)

                            # すべての重みがゼロに近い場合、正規化でゼロ除算を避ける
                            if np.sum(weights) == 0:
                                weights = [1.0/len(errors)] * len(errors) # 各モデルに均等な重み
                            else:
                                weights = np.array(weights) / np.sum(weights)  # 正規化して合計を1に
                            
                            ensemble_pred = sum(w * pred for w, pred in zip(weights, predictions))
                        elif ENSEMBLE_METHOD == 2:
                            # アンサンブルの最小値を選択
                            ensemble_pred = np.min(predictions, axis=0)  # 各サンプルごとに最小値を選択

                        #######################################################################

                        ensemble_error = calc.calc_r2_score(y_val, ensemble_pred, ensemble_r2_scores, fold, "Ensemble")

                        # 各モデルの出力が閾値より高いか判定
                        alexnet_pred_binary = (alexnet_pred > THRESHOLD).astype(int)
                        resnet50_pred_binary = (resnet50_pred > THRESHOLD).astype(int)
                        vgg16_pred_binary = (vgg16_pred > THRESHOLD).astype(int)

                        #######################################################################
                        ####################### 多数決するかしないかを選択 ######################

                        # ensemble_pred_binary = (alexnet_pred_binary + resnet50_pred_binary + vgg16_pred_binary) >= 2  # 2以上なら「沸騰」と判断
                        # バイナリラベルの生成
                        y_binary = (ensemble_pred >= THRESHOLD).astype(int)
                        y_val_binary = (y_val >= THRESHOLD).astype(int)

                        #######################################################################

                        alexnet_binary = resnet50_binary = vgg16_binary = ensemble_binary = y_val_binary

                        # ROC曲線のプロットとAUCの計算
                        calc.calc_roc_curve(alexnet_binary, alexnet_pred_binary, alexnet_auc_scores, fold, "AlexNet")
                        calc.calc_roc_curve(resnet50_binary, resnet50_pred_binary, resnet50_auc_scores, fold, "ResNet50")
                        calc.calc_roc_curve(vgg16_binary, vgg16_pred_binary, vgg16_auc_scores, fold, "VGG16")
                        calc.calc_roc_curve(ensemble_binary, y_binary, ensemble_auc_scores, fold, "Ensemble")

                        # 評価指標を計算
                        r2 = r2_score(y_val, ensemble_pred)  # 全体 R^2
                        # --- 追加：閾値以上のみの R^2 ---
                        mask_low  = (y_val < THRESHOLD).ravel()
                        mask_high = (y_val >= THRESHOLD).ravel()
                        y_val_high = y_val[mask_high].ravel()
                        y_pred = np.asarray(ensemble_pred).ravel()
                        ensemble_pred_high = np.asarray(ensemble_pred).ravel()[mask_high]
                        r2_high = r2_score(y_val_high, ensemble_pred_high)

                        # --- 追加：R²(High) が伸びない理由を数値で確認 ---
                        y_all   = np.asarray(y_val).ravel()
                        yhat_all= np.asarray(ensemble_pred).ravel()
                        m_high  = (y_all >= THRESHOLD)
                        m_low   = ~m_high
                        # ==== RMSE / MAE（All と High）====
                        rmse_all  = np.sqrt(mean_squared_error(y_all, yhat_all))
                        rmse_high = np.sqrt(mean_squared_error(y_all[m_high], yhat_all[m_high]))
                        mae_all   = mean_absolute_error(y_all, yhat_all)
                        mae_high  = mean_absolute_error(y_all[m_high], yhat_all[m_high])
                        # ===================================

                        # R²(High) の分母が縮む影響（global mean vs subset mean）
                        SSE_h = np.sum((y_all[m_high] - yhat_all[m_high])**2)
                        SST_h_global = np.sum((y_all[m_high] - y_all.mean())**2)
                        SST_h_subset = np.sum((y_all[m_high] - y_all[m_high].mean())**2)
                        R2_high_global = 1 - SSE_h / (SST_h_global + 1e-12)
                        R2_high_subset = 1 - SSE_h / (SST_h_subset + 1e-12)  # = r2_high と一致
                        SST_shrink = (SST_h_subset + 1e-12) / (SST_h_global + 1e-12)

                        # 高域の線形キャリブ（傾き・切片・バイアス）
                        from sklearn.linear_model import LinearRegression
                        reg_high = LinearRegression().fit(y_all[m_high].reshape(-1,1), yhat_all[m_high])
                        slope_high, intercept_high = reg_high.coef_[0], reg_high.intercept_
                        bias_high  = (yhat_all[m_high] - y_all[m_high]).mean()

                        # low の分散（真値 vs 予測）
                        std_true_low = np.std(y_all[m_low])
                        std_pred_low = np.std(yhat_all[m_low])
                        # ---------------------------------------------------


                        # 平均は全体平均（R²定義に合わせる）
                        y_mean = np.mean(y_val)

                        SSE_all  = np.sum((y_val - y_pred)**2)
                        SST_all  = np.sum((y_val - y_mean)**2)

                        SSE_low  = np.sum((y_val[mask_low]  - y_pred[mask_low])**2)
                        SSE_high = np.sum((y_val[mask_high] - y_pred[mask_high])**2)

                        SST_low  = np.sum((y_val[mask_low]  - y_mean)**2)
                        SST_high = np.sum((y_val[mask_high] - y_mean)**2)

                        print(
                            f"n_low={mask_low.sum()}, n_high={mask_high.sum()} | "
                            f"SSE_low/SSE_all={SSE_low/SSE_all:.3f}, SST_low/SST_all={SST_low/SST_all:.3f}"
                        )


                        # ---------------------------------
                        accuracy = accuracy_score(y_val_binary, y_binary)
                        precision = precision_score(y_val_binary, y_binary, zero_division=0)
                        recall = recall_score(y_val_binary, y_binary, zero_division=0)
                        f1 = f1_score(y_val_binary, y_binary, zero_division=0)

                        fpr, tpr, _ = roc_curve(y_val_binary, y_binary)
                        roc = auc(fpr, tpr)

                        # 評価指標をリストに保存
                        r2_scores.append(r2)
                        r2_high_scores.append(r2_high)
                        accuracy_scores.append(accuracy)
                        precision_scores.append(precision)
                        recall_scores.append(recall)
                        f1_scores.append(f1)
                        auc_scores.append(roc)
                        # ---- 集計リストへ ----
                        rmse_all_scores.append(rmse_all)
                        rmse_high_scores.append(rmse_high)
                        mae_all_scores.append(mae_all)
                        mae_high_scores.append(mae_high)


                        # 結果をファイルに追記
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"Recorded at: {current_time}\n")
                        f.write(f"Fold {fold} Results:\n")
                        f.write(f"Accuracy: {accuracy:.4f}\n")
                        f.write(f"Precision: {precision:.4f}\n")
                        f.write(f"Recall: {recall:.4f}\n")
                        f.write(f"F1 Score: {f1:.4f}\n")
                        f.write(f"R² Score: {r2:.4f}\n")
                        f.write(f"R² Score (y >= THRESHOLD): {r2_high:.4f}\n")
                        f.write(f"R² High (global mean): {R2_high_global:.4f}\n")
                        f.write(f"R² High (subset mean): {R2_high_subset:.4f}  (SST_shrink={SST_shrink:.3f})\n")
                        f.write(f"High calib: slope={slope_high:.3f}, intercept={intercept_high:.3g}, bias={bias_high:.3g}\n")
                        f.write(f"Low stds: true={std_true_low:.4g}, pred={std_pred_low:.4g}\n")
                        f.write(f"AUC Score: {roc:.4f}\n")
                        f.write(f"RMSE (All/High): {rmse_all:.4f} / {rmse_high:.4f}\n")
                        f.write(f"MAE  (All/High): {mae_all:.4f} / {mae_high:.4f}\n")
                        f.write("-"*30 + "\n")

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

                        plt.text(0.72, 0.10, f'R² All :  {r2:.4f}\nR² High: {r2_high:.4f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=40)   
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
                        output_path_base = os.path.join(SAVE_PATH, "regression_results")
                        if not os.path.exists(output_path_base):
                            os.makedirs(output_path_base, exist_ok=True)
                        output_path = os.path.join(output_path_base, f'regression_split_{snr_value}_{fold}.png')
                        plt.savefig(output_path)
                        plt.close()

                        fold += 1

                        # =========================================================
                        # ループの最後でメモリを確実に解放する処理
                        # =========================================================
                        
                        # 1. 重い変数を明示的に削除する
                        #    (変数が存在する場合のみ削除するように try-except または if で囲むと安全ですが、
                        #     このフローなら確実に生成されているため del でOKです)
                        del x_train, x_val, y_train, y_val, y_train_scaled, y_val_scaled
                        del alexnet_model, resnet50_model, vgg16_model
                        del alexnet_history, resnet50_history, vgg16_history
                        del predictions, ensemble_pred
                        # 必要であれば以下も削除
                        del alexnet_pred, resnet50_pred, vgg16_pred
                        
                        # 2. TensorFlowのセッションをクリアする
                        K.clear_session()
                        
                        # 3. ガベージコレクションを強制実行する(参照を切ったメモリ領域を即座にOSに返却させる）
                        gc.collect()

                    # 各モデルの回帰分析とR^2 scoreを保存
                    str_r2_alexnet, r2_alexnet, r2_error_list = save_combined_r2_auc("r2", alexnet_r2_scores, r2_error_list, "AlexNet")
                    str_r2_resnet, r2_resnet, r2_error_list = save_combined_r2_auc("r2", resnet50_r2_scores, r2_error_list, "ResNet50")
                    str_r2_vgg16, r2_vgg16, r2_error_list = save_combined_r2_auc("r2", vgg16_r2_scores, r2_error_list, "VGG16")
                    str_r2_ensemble, r2_ensemble, r2_error_list = save_combined_r2_auc("r2", ensemble_r2_scores, r2_error_list, "Ensemble")

                    # 各モデルのROC曲線とAUCを保存
                    str_auc_alexnet, auc_alexnet, auc_error_list = save_combined_r2_auc("auc", alexnet_auc_scores, auc_error_list, "AlexNet")
                    str_auc_resnet, auc_resnet, auc_error_list = save_combined_r2_auc("auc", resnet50_auc_scores, auc_error_list, "ResNet50")
                    str_auc_vgg16, auc_vgg16, auc_error_list = save_combined_r2_auc("auc", vgg16_auc_scores, auc_error_list, "VGG16")
                    str_auc_ensemble, auc_ensemble, auc_error_list = save_combined_r2_auc("auc", ensemble_auc_scores, auc_error_list, "Ensemble")

                    # 各モデルのR^2 scoreのリスト
                    str_r2_list = [str_r2_alexnet, str_r2_resnet, str_r2_vgg16, str_r2_ensemble]
                    r2_list = [r2_alexnet, r2_resnet, r2_vgg16, r2_ensemble]

                    # 各モデルのAUCのリスト
                    str_auc_list = [str_auc_alexnet, str_auc_resnet, str_auc_vgg16, str_auc_ensemble]
                    auc_list = [auc_alexnet, auc_resnet, auc_vgg16, auc_ensemble]

                    # 関数を呼び出して棒グラフを描画
                    plot_model_variables("r2", str_r2_list, r2_list, r2_error_list, EPOCH_NUM, SAVE_PATH, snr_value)
                    plot_model_variables("auc", str_auc_list, auc_list, auc_error_list, EPOCH_NUM, SAVE_PATH, snr_value)

                    r2_std_error = np.std(r2_scores) / np.sqrt(len(r2_scores))
                    accuracy_std_error = np.std(accuracy_scores) / np.sqrt(len(accuracy_scores))
                    precision_std_error = np.std(precision_scores) / np.sqrt(len(precision_scores))
                    recall_std_error = np.std(recall_scores) / np.sqrt(len(recall_scores))
                    f1_std_error = np.std(f1_scores) / np.sqrt(len(f1_scores))
                    auc_std_error = np.std(auc_scores) / np.sqrt(len(auc_scores))

                    # --- 追加：閾値以上 R^2 の平均±SE ---
                    r2_high_mean = np.mean(r2_high_scores)
                    r2_high_se   = np.std(r2_high_scores) / np.sqrt(len(r2_high_scores))
                    # -------------------------------------

                    rmse_all_mean,  rmse_all_se  = _mean_se(rmse_all_scores)
                    rmse_high_mean, rmse_high_se = _mean_se(rmse_high_scores)
                    mae_all_mean,   mae_all_se   = _mean_se(mae_all_scores)
                    mae_high_mean,  mae_high_se  = _mean_se(mae_high_scores)

                    # 平均結果をファイルに追記
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"Recorded at: {current_time}\n")
                    f.write("\nAverage Results:\n")
                    f.write(f"Average Accuracy: {np.mean(accuracy_scores):.4f} ± {accuracy_std_error:.4f}\n")
                    f.write(f"Average Precision: {np.mean(precision_scores):.4f} ± {precision_std_error:.4f}\n")
                    f.write(f"Average Recall: {np.mean(recall_scores):.4f} ± {recall_std_error:.4f}\n")
                    f.write(f"Average F1 Score: {np.mean(f1_scores):.4f} ± {f1_std_error:.4f}\n")
                    f.write(f"Average R² Score: {np.mean(r2_scores):.4f} ± {r2_std_error:.4f}\n")
                    f.write(f"Average R² Score (y >= THRESHOLD): {r2_high_mean:.4f} ± {r2_high_se:.4f}\n")
                    f.write(f"Average AUC Score: {np.mean(auc_scores):.4f} ± {auc_std_error:.4f}\n")
                    f.write(f"Average RMSE (All/High): {rmse_all_mean:.4f} ± {rmse_all_se:.4f} / "
                            f"{rmse_high_mean:.4f} ± {rmse_high_se:.4f}\n")
                    f.write(f"Average MAE  (All/High): {mae_all_mean:.4f} ± {mae_all_se:.4f} / "
                            f"{mae_high_mean:.4f} ± {mae_high_se:.4f}\n")
                    f.write("="*30 + "\n\n")

                if not FLG_ROOP:
                    break
            if not FLG_ROOP:
                break
        del x, y
        gc.collect()

if __name__ == '__main__':
    main()

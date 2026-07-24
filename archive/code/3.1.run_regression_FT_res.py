import os
import numpy as np
import time
import csv
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, r2_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import ResNet50
from sklearn.utils import resample
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D, Input
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Assuming these utility classes are in the specified paths and will be used as is,
# or their relevant parts are incorporated/modified.
# For RegressionModelMaker, we are replacing its ResNet50 part.
# from utils.models.regression import RegressionModelMaker # This will be replaced for ResNet50
from utils.dataloading.dataloading_and_conversion import DataLoadingConversion
from utils.calculation.calc_r2_auc import AUCorR2Calculation

#######  読んでよかった記事  #######
#######  https://atmarkit.itmedia.co.jp/ait/articles/2112/02/news016.html ・・・アンサンブル学習の手法のいろいろ
#######  https://qiita.com/eureka-ai/items/6c55e3b6d9617ae58afa  ・・・つよつよAIエンジニアになろう
#######  https://speakerdeck.com/moepy_stats/social-implementation-of-machine-learning  ・・・機械学習を「社会実装」するということ
#######  https://speakerdeck.com/shibuiwilliam/ji-jie-xue-xi-woshi-yong-hua-suruenziniaringusukiru  ・・・機械学習を実用化するエンジニアリングスキル



#######################################################################

#                         変数の指定

#######################################################################

# 訓練時のバッチサイズとエポック数のリスト
BATCH_SIZES = {"ResNet50": 12} # Only ResNet50
EPOCH_NUM = 100
# 学習率
LEARNING_RATE = {"ResNet50": 0.00005} # Only ResNet50
# 閾値
threshold_list = [260675.103239721, 353145.749413166, 264418.48987934]  # TODO: ここの値が、csvファイルで求めた沸騰開始点での熱流束となれば、沸騰-非沸騰分類モデルでのROC曲線が書けるのではないか？
THRESHOLD = sum(threshold_list) / len(threshold_list)

# フォールド数
DIVISIONS = 2
# チャンネル数
COLOR_CHANNEL = 1 # Set to 1 or 3. If 1, the model will adapt it. If 3, uses directly.

# アンサンブルの組み合わせ方（0・・単純平均 ｜ 1・・重み付き平均 ｜ 2・・最小値）
# ENSEMBLE_METHOD = 1 # No longer needed as we only have one model
# ブートストラップサンプルするか
BOOTSTRAP_SAMPLING = False

#  ホワイトノイズ ( = 0)か水流動音 ( = 1)か #
NOISE = 1

# 保存したモデルの重みを用いるかどうか
PREVIOUS_MODEL = False
SAVE_DATE = 20250523

# ハイパス通した録音データかどうか
HIGHPASS = True

#### データフォルダの設定 ####
noise = "whitenoise" if NOISE == 0 else "waterflow"
highpass = "_highpass_0.5s" if HIGHPASS else ""
highpass = "_sf_20250408_0.5s" if HIGHPASS else "" # 20250408限定
noise = noise + highpass
BASE_DATA_PATH = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy" + "\\" + noise + f"\\channel={COLOR_CHANNEL}\\"
DATA_PATH = [BASE_DATA_PATH + "heatflux_no_noise",]
            #  BASE_DATA_PATH + "heatflux_SNR=0",
            #  BASE_DATA_PATH + "heatflux_SNR=-4",
            #  BASE_DATA_PATH + "heatflux_SNR=-8",
            #  BASE_DATA_PATH + "heatflux_SNR=-12",
            #  BASE_DATA_PATH + "heatflux_SNR=-16",
            #  BASE_DATA_PATH + "heatflux_SNR=-20"]

#### regression_resultとROC曲線の保存先フォルダ ####
# BASE_SAVE_PATH = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\regression_result\npy\ensemble" + "\\channel=" + str(COLOR_CHANNEL) # Ensemble part of path might need change
BASE_SAVE_PATH = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\regression_result\npy\resnet50_finetuned" + "\\channel=" + str(COLOR_CHANNEL)


# Adjusted SAVE_PATH for single model, ensemble method is irrelevant
# if ENSEMBLE_METHOD == 0:
#     SAVE_PATH = os.path.join(BASE_SAVE_PATH, f"average")
# elif ENSEMBLE_METHOD == 1:
#     SAVE_PATH = os.path.join(BASE_SAVE_PATH, f"weight_average_highpass_20250408\\roc")
# elif ENSEMBLE_METHOD == 2:
#     SAVE_PATH = os.path.join(BASE_SAVE_PATH, f"min")
SAVE_PATH = os.path.join(BASE_SAVE_PATH, f"resnet50_results_highpass_{SAVE_DATE}\\roc")


if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# matplotlibの設定
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

#######################################################################

#                   データの読み込みと形の変換 (from original)

#######################################################################

# This function is now part of DataLoadingConversion class, assuming it works as intended.
# def load_images_from_folder(folder_path):
#     x, y = [], []
#     print("読み込みスタート")
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".npy"):
#             heat_flux = float(filename.split('_')[0])
#             data = np.load(os.path.join(folder_path, filename))
#             x.append(data)
#             y.append(heat_flux)
#     x = np.array(x)
#     y = np.array(y)
#     x = x.astype('float32')
#     return x, y

#######################################################################

#                     ResNet50 Fine-tuning Model

#######################################################################

def create_resnet50_finetuned_model(input_shape=(224, 224, 1)):
    """
    Creates a ResNet50 model for regression, fine-tuned on ImageNet.
    Handles 1-channel or 3-channel input.
    """
    inputs = Input(shape=input_shape)
    
    current_channels = input_shape[-1]
    
    if current_channels == 1:
        # Adapt 1-channel input to 3-channel for ResNet50 base
        x = Conv2D(3, (1, 1), padding='same', name='conv_1_to_3_channels')(inputs) # 1x1 conv to make 3 channels
    elif current_channels == 3:
        x = inputs
    else:
        raise ValueError(f"Unsupported number of channels: {current_channels}. Must be 1 or 3.")

    # Base ResNet50 model with ImageNet weights, input_shape must be (H, W, 3)
    # The input_tensor argument is not needed if we pass the processed tensor `x` directly to the base model call.
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(input_shape[0], input_shape[1], 3))
    
    # Freeze the layers of the base model if desired (common in fine-tuning)
    # for layer in resnet_base.layers:
    #     layer.trainable = False 
    # Or fine-tune some top layers:
    # for layer in resnet_base.layers[:-10]: # Example: freeze all but last 10
    #    layer.trainable = False

    x = resnet_base(x) # Pass the (potentially adapted) input
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.5)(x) # Optional dropout
    outputs = Dense(1, activation='linear', name='fc_regression')(x) # Linear activation for regression

    model = Model(inputs=inputs, outputs=outputs)
    return model

#######################################################################

#                回帰分析とROC曲線の描画する関数

#######################################################################

def save_combined_r2_auc(valuation, valuation_indicators_scores, error_list, model_name):
    """
    5分割交差検証のROC曲線の平均AUCと95%信頼区間を計算し、グラフを保存
    """
    mean = np.mean(valuation_indicators_scores)
    if len(valuation_indicators_scores) > 1:
        std_valuation = np.std(valuation_indicators_scores, ddof=1)
        se_valuation = std_valuation / np.sqrt(len(valuation_indicators_scores))
        confidence_interval = 1.96 * se_valuation
    else: # Cannot compute CI for a single value
        std_valuation = 0
        se_valuation = 0
        confidence_interval = 0
    
    # 信頼区間を計算
    upper_bound = mean + confidence_interval
    lower_bound = mean - confidence_interval
    error_list.append(confidence_interval) # Store the CI half-width as error
    print('Error list size : ', len(error_list))
    print(f'upper_bound : {upper_bound:.4f} | Lower bound : {lower_bound:.4f} | Confidence interval : {confidence_interval:.4f}')
    if valuation == "r2":
        print(f'Model : {model_name} | Combined Regression saved for Mean R^2 score = {mean:.4f} ± {confidence_interval:.4f}')
    else:
        print(f'Model : {model_name} | Combined ROC curve saved for Mean AUC = {mean:.4f} ± {confidence_interval:.4f}')

    return f'{mean:.4f} ± {confidence_interval:.4f}', mean, error_list


def plot_model_variables(valuation, str_valuation_list, valuation_list, error_list, epochs, save_path, snr_value):
    # matplotlibの設定
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    models = ['ResNet50'] # Only ResNet50

    if valuation == "r2":
        print("R^2 score List:", valuation_list)
        print("String R^2 score List:", str_valuation_list)
        
        plt.figure(figsize=(6, 6)) # Adjusted size for single model
        
        if len(valuation_list) != len(models) or len(str_valuation_list) != len(models):
            raise ValueError(f"Length of r2_list ({len(valuation_list)}) or str_r2_list ({len(str_valuation_list)}) does not match the number of models ({len(models)}).")

        bars = plt.bar(models, valuation_list, color=['cadetblue'], yerr=error_list, capsize=5, width=0.3)
        plt.ylim(0.0, 1.05)
        plt.ylabel('R² Score', fontsize=23)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=18)

        for i, (val, str_val) in enumerate(zip(valuation_list, str_valuation_list)):
            plt.text(i, 0.03, str_val, ha='center', va='bottom', fontsize=25, color='black', rotation=90)

        plt.savefig(os.path.join(save_path, f'Comparison_of_R^2_score_ep{epochs}_SNR={snr_value}_ResNet50.png'))
        plt.close()
    else: # AUC
        print("AUC List:", valuation_list)
        print("String AUC List:", str_valuation_list)

        plt.figure(figsize=(6, 6)) # Adjusted size for single model
        
        if len(valuation_list) != len(models) or len(str_valuation_list) != len(models):
             raise ValueError(f"Length of auc_list ({len(valuation_list)}) or str_auc_list ({len(str_valuation_list)}) does not match the number of models ({len(models)}).")

        bars = plt.bar(models, valuation_list, color=['cadetblue'], yerr=error_list, capsize=5, width=0.3)
        plt.ylim(0.0, 1.05)
        plt.ylabel('AUC', fontsize=20)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=18)

        for i, (val, str_val) in enumerate(zip(valuation_list, str_valuation_list)):
            plt.text(i, 0.03, str_val, ha='center', va='bottom', fontsize=25, color='black', rotation=90)

        plt.savefig(os.path.join(save_path, f'Comparison_of_AUC_ep{epochs}_SNR={snr_value}_ResNet50.png'))
        plt.close()


#######################################################################

#                               実行部

#######################################################################

def main():
    for data_path_item in DATA_PATH: # Renamed to avoid conflict with os.path
        # SNR値をパスから抽出
        if "no_noise" in data_path_item:
            snr_value = "no_noise"
        else:
            snr_value = data_path_item.split("SNR=")[-1]

        # データのロード
        start_time = time.time()
        data_loading = DataLoadingConversion()
        # The load_images_from_folder function is assumed to be part of DataLoadingConversion
        # If not, you might need to call it directly: x, y = load_images_from_folder(data_path_item)
        x, y = data_loading.load_images_from_folder(data_path_item)
        load_time = time.time() - start_time

        print("x shape:", x.shape)
        print("y shape:", y.shape)
        print(f"データの読み込み時間: {load_time:.2f} 秒")
        
        # Ensure x has the correct number of channels for the model input_shape
        # If COLOR_CHANNEL is 1 but model expects 3 (standard ResNet), data needs adaptation here
        # or inside create_resnet50_finetuned_model, which is what we did.
        # Example if COLOR_CHANNEL = 1 and you need to triplicate for a model expecting 3 channels:
        # if COLOR_CHANNEL == 1 and x.shape[-1] == 1:
        #    x = np.repeat(x, 3, axis=-1) # Make sure this is what you want.
        # Our `create_resnet50_finetuned_model` handles 1-channel to 3-channel conversion internally.


        # 分割交差検証の準備
        kf = KFold(n_splits=DIVISIONS, shuffle=True, random_state=42)

        # R^2 scoreとAUCの値を保存するリスト
        resnet50_r2_scores, resnet50_auc_scores = [], []

        # R^2 scoreとAUCエラーバー用のリスト
        r2_error_list = []
        auc_error_list = []

        plt.figure() # For ROC curves if calc.calc_roc_curve plots on current figure

        fold = 1
        for train_index, val_index in kf.split(x):
            x_train, x_val = x[train_index], x[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # モデルの作成 (ResNet50 Fine-tuned)
            input_img_shape = (x_train.shape[1], x_train.shape[2], COLOR_CHANNEL) # Should be (224,224,COLOR_CHANNEL)
            resnet50_model = create_resnet50_finetuned_model(input_shape=input_img_shape)
            
            # モデルのコンパイル
            resnet50_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE['ResNet50']), loss='mean_squared_error')
            resnet50_model.summary() # Print model summary

            if BOOTSTRAP_SAMPLING:
                x_train_resampled, y_train_resampled = resample(x_train, y_train, random_state=fold * 1)
            else:
                x_train_resampled, y_train_resampled = x_train, y_train


            # モデルの訓練
            # save_dir_1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(BASE_SAVE_PATH))))
            # The original BASE_SAVE_PATH was for ensemble, now it's for resnet50_finetuned.
            # Let's adjust path for weights if needed. Current SAVE_PATH is for results.
            weights_base_dir = os.path.join(os.path.dirname(os.path.dirname(BASE_SAVE_PATH)), "model_weights") # Example structure
            save_dir = os.path.join(weights_base_dir, f"all_weights_{SAVE_DATE}\\{snr_value}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            resnet50_weights_path = os.path.join(save_dir, f"resnet50_fold{fold}_{snr_value}.h5")

            if not PREVIOUS_MODEL:
                print(f"ResNet50 Model Training Start : Fold {fold} / {DIVISIONS}, SNR: {snr_value}")
                resnet50_model.fit(x_train_resampled, y_train_resampled, batch_size=BATCH_SIZES['ResNet50'], epochs=EPOCH_NUM, verbose=1, validation_data=(x_val, y_val))
                resnet50_model.save_weights(resnet50_weights_path)
                print(f"ResNet50 weights saved to {resnet50_weights_path}")
            else:
                if os.path.exists(resnet50_weights_path):
                    resnet50_model.load_weights(resnet50_weights_path)
                    print(f"ResNet50 weights loaded from {resnet50_weights_path}")
                else:
                    print(f"Warning: PREVIOUS_MODEL is True, but weights file not found at {resnet50_weights_path}. Training from scratch.")
                    print(f"ResNet50 Model Training Start (fallback): Fold {fold} / {DIVISIONS}, SNR: {snr_value}")
                    resnet50_model.fit(x_train_resampled, y_train_resampled, batch_size=BATCH_SIZES['ResNet50'], epochs=EPOCH_NUM, verbose=1, validation_data=(x_val, y_val))
                    resnet50_model.save_weights(resnet50_weights_path)
                    print(f"ResNet50 weights saved to {resnet50_weights_path}")
            
            # モデルの予測
            resnet50_pred = resnet50_model.predict(x_val)

            # 回帰分析の結果とR^2 scoreの計算
            calc = AUCorR2Calculation() # Instantiate your calculation class
            # calc_r2_score method is expected to append score to resnet50_r2_scores
            # and return error (or r2_score itself, depending on its implementation)
            _ = calc.calc_r2_score(y_val, resnet50_pred, resnet50_r2_scores, fold, "ResNet50") 

            # モデルの出力が閾値より高いか判定
            resnet50_pred_binary = (resnet50_pred > THRESHOLD).astype(int)
            
            # 正解ラベルを二値化
            y_val_binary = (y_val > THRESHOLD).astype(int)

            # ROC曲線のプロットとAUCの計算
            # calc_roc_curve method is expected to append score to resnet50_auc_scores
            # and potentially plot something or return fpr, tpr, auc_score
            _ = calc.calc_roc_curve(y_val_binary, resnet50_pred, resnet50_auc_scores, fold, "ResNet50") # Pass raw predictions for roc_curve typically

            fold += 1

        # 各モデルの回帰分析とR^2 scoreを保存
        str_r2_resnet, r2_resnet, r2_error_list = save_combined_r2_auc("r2", resnet50_r2_scores, r2_error_list, "ResNet50")
        
        # 各モデルのROC曲線とAUCを保存
        str_auc_resnet, auc_resnet, auc_error_list = save_combined_r2_auc("auc", resnet50_auc_scores, auc_error_list, "ResNet50")

        # 各モデルのR^2 scoreのリスト
        str_r2_list = [str_r2_resnet]
        r2_list = [r2_resnet]

        # 各モデルのAUCのリスト
        str_auc_list = [str_auc_resnet]
        auc_list = [auc_resnet]

        # 関数を呼び出して棒グラフを描画
        plot_model_variables("r2", str_r2_list, r2_list, r2_error_list, EPOCH_NUM, SAVE_PATH, snr_value)
        plot_model_variables("auc", str_auc_list, auc_list, auc_error_list, EPOCH_NUM, SAVE_PATH, snr_value)

    print("Processing complete.")

if __name__ == '__main__':
    main()
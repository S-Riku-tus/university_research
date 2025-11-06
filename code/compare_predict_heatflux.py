import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils.models.regression.base_regression import RegressionModelMaker
from utils.dataloading.dataloading_and_conversion import DataLoadingConversion
import matplotlib.pyplot as plt

# matplotlibの設定
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# =====================================================================
# ステップ1：【重要】推論の前提条件を設定するフェーズ
# =====================================================================

# 1. 読み込む学習済みモデルの情報を指定
SAVE_DATE = "20251020"
EPOCH_NUM = 100
CHUNK = 1
SNR_VALUE = "no_noise"

# 学習済みモデルの重みが保存されているディレクトリパス
WEIGHTS_BASE_DIR = rf"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\all_weights\{SAVE_DATE}_epoch{EPOCH_NUM}_chunk{CHUNK}\{SNR_VALUE}"

# 評価したいfoldの番号をリストで指定
FOLD_NUMBERS = [1, 2, 3, 4, 5] 

# 2. 推論したい3つの画像ファイル（.npy）のフルパスを指定
# # 205未沸騰
# INFERENCE_IMAGE_PATHS = [
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\2.05e+04_1.npy",
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\2.05e+04_2.npy",
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\2.05e+04_3.npy"
# ]
# # 261沸騰
# INFERENCE_IMAGE_PATHS = [
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\2.61e+05_9.npy",
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\2.61e+05_19.npy",
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\2.61e+05_34.npy"
# ]
# # 261未沸騰
# INFERENCE_IMAGE_PATHS = [
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\2.61e+05_1.npy",
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\2.61e+05_2.npy",
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\2.61e+05_3.npy"
# ]
# # 308沸騰
# INFERENCE_IMAGE_PATHS = [
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\3.08e+05_3.npy",
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\3.08e+05_7.npy",
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\3.08e+05_23.npy"
# ]
# # 360沸騰
# INFERENCE_IMAGE_PATHS = [
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\3.60e+05_7.npy",
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\3.60e+05_37.npy",
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\3.60e+05_40.npy"
# ]
# # 360_2沸騰
# INFERENCE_IMAGE_PATHS = [
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\3.60e+05_28.npy",
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\3.60e+05_36.npy",
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\3.60e+05_38.npy"
# ]
# # 416沸騰？
# INFERENCE_IMAGE_PATHS = [
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\4.16e+05_16.npy",
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\4.16e+05_17.npy",
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\4.16e+05_18.npy"
# ]
# # 416未沸騰
# INFERENCE_IMAGE_PATHS = [
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\4.16e+05_3.npy",
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\4.16e+05_4.npy",
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\4.16e+05_5.npy"
# ]
# # 540沸騰
# INFERENCE_IMAGE_PATHS = [
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\5.40e+05_56.npy",
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\5.40e+05_57.npy",
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\5.40e+05_58.npy"
# ]
# # 676沸騰
# INFERENCE_IMAGE_PATHS = [
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\6.76e+05_29.npy",
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\6.76e+05_30.npy",
#     r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\waterflow_20251015_1s\maxfreq=22kHz\heatflux_no_noise\6.76e+05_31.npy"
# ]
TMP = "360_2沸騰"

# 3. 学習時に使用したデータ全体のディレクトリパス（スケーラーの復元に必須）
DATA_DATE = "20251015"
noise_type = "waterflow"
highpass_info = f"_{DATA_DATE}_{CHUNK}s"
noise_folder_name = noise_type + highpass_info
BASE_DATA_PATH = rf"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\data\npy\{noise_folder_name}\maxfreq=22kHz"
DATA_PATH_FOR_SCALER = os.path.join(BASE_DATA_PATH, f"heatflux_{SNR_VALUE}")

BASE_RESULT_PATH = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\regression_result\npy\ensemble\heatflux_no_noise\pre_20251020_ep100_bs12_lr0.001"
# 今回の「3サンプル推論」の結果を保存するサブディレクトリ
INFERENCE_SUBDIR_NAME = "inference_on_ramdom_samples"
FINAL_SAVE_DIRECTORY = os.path.join(BASE_RESULT_PATH, INFERENCE_SUBDIR_NAME)

# 保存先ディレクトリを作成
os.makedirs(FINAL_SAVE_DIRECTORY, exist_ok=True)
print(f"結果は {FINAL_SAVE_DIRECTORY} に保存されます。")

print("ステップ1: 準備を開始します...")

# スケーラーの準備
try:
    data_loading = DataLoadingConversion()
    _, y_all = data_loading.load_npy_data(DATA_PATH_FOR_SCALER)
    scaler = MinMaxScaler()
    scaler.fit(y_all.reshape(-1, 1))
    print("スケーラーの準備が完了しました。")
except Exception as e:
    print(f"[エラー] スケーラーの準備に失敗しました: {e}")
    exit()

# 全foldの結果を保存するための辞書を準備
all_results = {path: [] for path in INFERENCE_IMAGE_PATHS}

# =====================================================================
# 指定されたfoldの数だけループ処理を実行
# =====================================================================
for fold_num in FOLD_NUMBERS:
    print(f"\n{'='*25} Fold {fold_num} の推論を開始 {'='*25}")

    # ステップ2：学習済みモデルをローカルから復元
    print(f"ステップ2 (Fold {fold_num}): 学習済みモデルの復元を開始...")

    regression_model_maker = RegressionModelMaker((224, 224, 1))
    alexnet_model = regression_model_maker.alexnet()
    resnet50_model = regression_model_maker.cnn_transformer_v1()
    vgg16_model = regression_model_maker.cnn_transformer_v2()

    try:
        alexnet_model.load_weights(os.path.join(WEIGHTS_BASE_DIR, f"AlexNet_fold{fold_num}_{SNR_VALUE}.weights.h5"))
        resnet50_model.load_weights(os.path.join(WEIGHTS_BASE_DIR, f"resnet50_fold{fold_num}_{SNR_VALUE}.weights.h5"))
        vgg16_model.load_weights(os.path.join(WEIGHTS_BASE_DIR, f"vgg16_fold{fold_num}_{SNR_VALUE}.weights.h5"))
        print(f"Fold {fold_num} のモデル重みを正常に読み込みました。")
    except Exception as e:
        print(f"[エラー] Fold {fold_num} の重みファイルの読み込みに失敗しました: {e}")
        continue

    # ステップ3：指定したデータで推論
    print(f"ステップ3 (Fold {fold_num}): 指定した3つのデータで推論を実行...")
    
    for image_path in INFERENCE_IMAGE_PATHS:
        try:
            image_data = np.load(image_path)
            image_data_batch = np.expand_dims(image_data, axis=0)
            
            # 各モデルで推論を実行 -> 出力は0-1のスケール
            alexnet_pred_scaled = alexnet_model.predict(image_data_batch, verbose=0)
            resnet50_pred_scaled = resnet50_model.predict(image_data_batch, verbose=0)
            vgg16_pred_scaled = vgg16_model.predict(image_data_batch, verbose=0)

            # --- ▼▼▼【今回の修正箇所】▼▼▼ ---
            # 1. 各モデルの予測を、個別に物理スケールに戻す
            alexnet_pred = scaler.inverse_transform(alexnet_pred_scaled)
            resnet50_pred = scaler.inverse_transform(resnet50_pred_scaled)
            vgg16_pred = scaler.inverse_transform(vgg16_pred_scaled)

            # 2. 物理スケールに戻した値でアンサンブル（単純平均）
            ensemble_prediction = (alexnet_pred + resnet50_pred + vgg16_pred) / 3
            predicted_heat_flux = ensemble_prediction[0][0]
            # --- ▲▲▲【今回の修正箇所】▲▲▲ ---

            # 結果を辞書に保存
            all_results[image_path].append(predicted_heat_flux)
            
        except Exception as e:
            file_name = os.path.basename(image_path)
            print(f"[エラー] ファイル '{file_name}' の処理中にエラー: {e}")

# =====================================================================
# ステップ4：全Foldの平均結果をまとめて表示・可視化
# =====================================================================
print(f"\n{'='*25} 全Foldの平均結果 {'='*25}")
print("-" * 70)
print(f"{'状態':<15} | {'真値 (W/m²)':>18} | {'平均予測値 (W/m²)':>18} | {'誤差率 (%)':>12}")
print("-" * 70)

final_labels = []
final_true_values = []
final_pred_values = []

for image_path, pred_list in all_results.items():
    if not pred_list:
        continue

    file_name = os.path.basename(image_path)
    # ファイル名から真値を取得 (例: '1.09e+05_5.npy' -> 1.09e+05)
    true_heat_flux = float(file_name.split('_')[0])
    
    average_prediction = np.mean(pred_list)
    error = average_prediction - true_heat_flux
    error_percentage = (error / true_heat_flux) * 100 if true_heat_flux != 0 else float('inf')
    
    # ファイル名に含まれる熱流束の値でおおよその状態を判定
    if true_heat_flux < 200000: state_label = "未沸騰"
    elif true_heat_flux < 300000: state_label = "沸騰開始点近傍"
    else: state_label = "核沸騰（活発）"
    
    print(f"{state_label:<15} | {true_heat_flux:>18,.2f} | {average_prediction:>18,.2f} | {error_percentage:>11.2f} %")

    final_labels.append(state_label)
    final_true_values.append(true_heat_flux)
    final_pred_values.append(average_prediction)

print("-" * 70)

# 結果を棒グラフで可視化
x = np.arange(len(final_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width/2, final_true_values, width, label='実際の熱流束 (真値)', color='skyblue')
rects2 = ax.bar(x + width/2, final_pred_values, width, label=f'AIによる平均予測値 ({len(FOLD_NUMBERS)} folds)', color='royalblue')

ax.set_ylabel('熱流束 (W/m²)', fontsize=14)
ax.set_title(f'指定データにおけるAIの平均予測結果 ({len(FOLD_NUMBERS)} Folds Average)', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(final_labels, fontsize=12)
ax.legend(fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val):,}'))

ax.bar_label(rects1, padding=3, fmt='{:,.0f}')
ax.bar_label(rects2, padding=3, fmt='{:,.0f}')

fig.tight_layout()

# ファイル名を定義 (SNR値やFold数を含めると分かりやすい)
save_filename = f"inference_result_{SNR_VALUE}_{len(FOLD_NUMBERS)}folds_{TMP}.png"
full_save_path = os.path.join(FINAL_SAVE_DIRECTORY, save_filename)

try:
    plt.savefig(full_save_path, dpi=300) # dpiを指定して解像度を上げる
    print(f"グラフを {full_save_path} に保存しました。")
except Exception as e:
    print(f"[エラー] グラフの保存に失敗しました: {e}")
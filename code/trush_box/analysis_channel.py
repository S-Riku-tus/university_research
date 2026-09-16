import os
import librosa
import numpy as np
from skimage.transform import resize

# 音声ファイルが格納されたフォルダパス
npy_file_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.9.18\data\npy\channel=1\heatflux_no_noise\1.040E+06_7.npy"


npy_y = np.load(npy_file_path)

print(npy_y.shape)
print(np.max(npy_y), np.min(npy_y))


folder_path = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.9.18\data\npy\channel=1\heatflux_no_noise"

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
print("y[0]:", y[61])
y_min, y_max = y.min(), y.max()
y = (y - y_min) / (y_max - y_min)
print("正規化後のデータ:")
print("y[0]:", y[61])
y = y * (y_max - y_min) + y_min
print("正規化前のデータ:")
print("y[0]:", y[61])

alexnet_pred_binary = [0, 1, 1, 0, 1]
resnet50_pred_binary = [1, 1, 0, 0, 0]
vgg16_pred_binary = [0, 1, 1, 1, 1]

# リストをNumPy配列に変換して、要素ごとに計算できるようにします
alexnet_pred_binary = np.array(alexnet_pred_binary)
resnet50_pred_binary = np.array(resnet50_pred_binary)
vgg16_pred_binary = np.array(vgg16_pred_binary)

ensemble_pred_binary = (alexnet_pred_binary + resnet50_pred_binary + vgg16_pred_binary) >= 2  # 2以上なら「沸騰」と判断
print(ensemble_pred_binary)
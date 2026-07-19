import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class DataLoadingConversion:
    def __init__(self):
        pass

    def load_npy_data(self, folder_path):
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
        # 熱流束の値を0から1にスケーリングするのは、ResNet50の学習に良くなかったのでやめた
        # global y_min, y_max
        # y_min, y_max = y.min(), y.max()
        # y = (y - y_min) / (y_max - y_min)
        x = x.astype('float32')
        if x.ndim == 3:
            x = x[..., None]

        return x, y
    
    # def inverse_scale_y(y_scaled):
    #     """スケーリングされたyを元のスケールに戻す関数"""
    #     return y_scaled * (y_max - y_min) + y_min

    def load_image_data(self, folder_path):
        x, y = [], []
        print("読み込みスタート")
        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                # ファイル名から熱流束の値を取得
                heat_flux = float(filename.split('_')[0])
                
                # 画像を読み込み、リサイズ
                img_path = os.path.join(folder_path, filename)
                # color_mode='rgb'で3チャンネルを保証
                image = load_img(img_path, target_size=(224, 224), color_mode='rgb') 
                
                # 画像をNumpy配列に変換してリストに追加
                x.append(img_to_array(image))
                y.append(heat_flux)

        # リストをNumpy配列に変換
        x = np.array(x)
        y = np.array(y)

        # ピクセル値を0-1に正規化
        x = x.astype('float32') / 255.0
        return x, y

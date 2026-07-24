import numpy as np
import os 
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

import keras
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils.image_utils import load_img, img_to_array
from keras.applications.resnet import ResNet50
#from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential

#https://tecsingularity.com/tensorflow/keras_alexnet/
#https://qiita.com/kiii142/items/eadfaf5cf305981353e8

# -------------------------------------------------------------------------------------
#                        初期設定部
# -------------------------------------------------------------------------------------

# GrayScaleのときに1、COLORのときに3にする
COLOR_CHANNEL = 3

# 入力画像サイズ(画像サイズは正方形とする)
INPUT_IMAGE_SIZE = 224

# 訓練時のバッチサイズとエポック数
BATCH_SIZE = 32
EPOCH_NUM = 1500

# 使用する訓練画像の入ったフォルダ(ルート)
TRAIN_PATH = "C:/Users/Ueki Lab/Desktop/計測データ/0.3/0.3_heat_flux_regression"
# 使用する訓練画像の各クラスのフォルダ名
folder = os.listdir(TRAIN_PATH)

# CLASS数を取得する
CLASS_NUM = len(folder)
print("クラス数 : " + str(CLASS_NUM))

# クラス名をフォルダ名から取得
class_names = folder

# -------------------------------------------------------------------------------------
#                        訓練画像入力部
# -------------------------------------------------------------------------------------

# 各フォルダの画像を読み込む
v_image = []
v_label = []
for index, name in enumerate(folder):
    dir = TRAIN_PATH + "/" + name
    files = glob.glob(dir + "/*.png")
    print(dir)
    for i, file in enumerate(files):
        if COLOR_CHANNEL == 1:
            img = load_img(file, color_mode = "grayscale", target_size=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
        elif COLOR_CHANNEL == 3:
            img = load_img(file, color_mode = "rgb", target_size=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
        array = img_to_array(img)
        v_image.append(array)
        v_label.append(name)  # フォルダ名をクラスラベルとして使用

v_image = np.array(v_image)
v_label = np.array(v_label)


# imageの画素値をint型からfloat型にする
v_image = v_image.astype('float32')
# 画素値を[0～255]⇒[0～1]とする
v_image = v_image / 255.0

# 正解ラベルの形式を変換
#v_label = np.array([class_names.index(label) for label in v_label])
#v_label = np_utils.to_categorical(v_label, CLASS_NUM)

# 学習用データと検証用データに分割する
train_images, valid_images, train_labels, valid_labels = train_test_split(v_image, v_label, test_size=0.20)

train_labels = train_labels.astype('float32')
valid_labels = valid_labels.astype('float32')



# -------------------------------------------------------------------------------------
#                      モデルアーキテクチャ定義部
# -------------------------------------------------------------------------------------

#ResNet50
# ResNet50の読み込み（重みはImageNetなら='imagenet'、初期化するなら=None）
base_model = ResNet50(weights=None,include_top=False, input_shape=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, COLOR_CHANNEL))

# 出力の形状を取得
output_shape = base_model.layers[-1].output_shape

#全結合層を新しく作る
top_model = Sequential()
top_model.add(Flatten(input_shape=output_shape[1:]))
top_model.add(Dense(1, activation='linear'))
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

# モデルのコンパイル
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 訓練
history = model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCH_NUM)

# -------------------------------------------------------------------------------------
#                              訓練実行&結果確認部
# -------------------------------------------------------------------------------------

# モデル構成の確認
model.summary()

score = model.evaluate(valid_images, valid_labels, verbose=0)
print(len(valid_images))
print('Loss:', score[0])
print('Accuracy:', score[1])

#回帰分析結果
pred  = model.predict(valid_images)
plt.scatter(valid_labels, pred)
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.title('regression analysis')
# 回帰直線を追加
plt.plot([min(valid_labels), max(valid_labels)], [min(valid_labels), max(valid_labels)], color='red', linestyle='--', linewidth=2)
# 目盛りを内向きにする
plt.tick_params(direction='in', which='both')
# 決定係数の計算
r2 = r2_score(valid_labels, pred)
# 決定係数を図に追加
plt.text(0.5, 0.1, f'R^2 Score: {r2:.2f}', ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
plt.show()

print(f'R^2 Score: {r2}')

# CSVに保存するためにデータをDataFrameに変換
result_df = pd.DataFrame({'True_Label': valid_labels.flatten(), 'Predicted_Label': pred.flatten()})
# CSVファイルとして保存
result_df.to_csv('ResNet50_regression_1500.csv', index=False)

#lossの推移を表示
plt.plot(range(EPOCH_NUM), history.history['loss'], label='Train loss')
plt.plot(range(EPOCH_NUM), history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
# 目盛りを内向きにする
plt.tick_params(direction='in', which='both')

# lossの推移を表示（対数表示）
plt.plot(range(EPOCH_NUM), history.history['loss'], label='Train loss')
plt.plot(range(EPOCH_NUM), history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.show()
# 目盛りを内向きにする
plt.tick_params(direction='in', which='both')

#学習した重みを保存
#model.save('my_model.h5')
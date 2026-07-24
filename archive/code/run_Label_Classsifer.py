import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, ResNet50
import matplotlib.pyplot as plt
import seaborn as sns
import random

#######################################################################

#                              変数の指定

#######################################################################

# 訓練時のバッチサイズとエポック数のリスト
BATCH_SIZES = {"AlexNet": 12, "VGG16": 12, "ResNet50": 12,}
EPOCH_NUMS = [300]
# 学習率
LEARNING_RATE = {"AlexNet": 0.0001, "VGG16": 0.00005, "ResNet50": 0.0005,}

# 特徴量データの読み込み
TRAIN_PATH = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\6月26日_サブクール度10度_0.3mm\スペクトログラム\train"
TEST_PATH = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\6月26日_サブクール度10度_0.3mm\スペクトログラム\test"
# val-lossグラフの保存先フォルダ
SAVE_PATH = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\6月26日_サブクール度10度_0.3mm\training_results\6クラス分類\Ensemble"

#######################################################################

#                       データの読み込みと形の変換

#######################################################################

def load_images_from_folder(folder_path):
    x, y = [], []
    class_names = ['boiling1', 'boiling2', 'boiling3', 'boiling4', 'boiling5', 'not_boiling']
    image_files = {class_name: [] for class_name in class_names}

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            for class_name in class_names:
                if filename.startswith(class_name):
                    image_files[class_name].append(filename)
                    break

    min_images = min(len(image_files[class_name]) for class_name in class_names[:-1])

    selected_images = {class_name: random.sample(image_files[class_name], min_images) for class_name in class_names[:-1]}
    selected_images['not_boiling'] = random.sample(image_files['not_boiling'], min_images)

    for class_idx, class_name in enumerate(class_names):
        for filename in selected_images[class_name]:
            image = load_img(os.path.join(folder_path, filename), target_size=(224, 224))
            x.append(img_to_array(image))
            y.append(class_idx)

    x = np.array(x)
    y = np.array(y)
    x = x.astype('float32')
    x = x / 255.0
    y = to_categorical(y, num_classes=len(class_names))
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    return x, y, class_names

#######################################################################

#                         AlexNetのモデルの作成

#######################################################################

def alexnet(input_shape, num_classes):
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
    model.add(Dense(num_classes, activation='softmax'))
    return model

#######################################################################

#                         VGG16のモデルの作成

#######################################################################

def vgg16(input_shape, num_classes):
    base_model = VGG16(weights=None, include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

#######################################################################

#                         ResNet50のモデルの作成

#######################################################################

def resnet50(input_shape, num_classes):
    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

#######################################################################

#                           混同行列の保存

#######################################################################

def save_confusion_matrix(y_true, y_pred, class_names, save_path, epochs):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 8))
    cmap = plt.get_cmap('Blues')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f'Confusion Matrix (Epochs: {epochs})')
    plt.colorbar()

    # 使用したクラス名をラベルとして設定
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # セル内の数字の表示
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=12)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'confusion_matrix_ep{epochs}.png'))
    plt.close()

#######################################################################

#                   Scikit-learn VotingClassifier のラッパー

#######################################################################

class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_fn, input_shape, num_classes, learning_rate=0.0005):
        self.model_fn = model_fn
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = None

    def fit(self, X, y, epochs, batch_size):
        self.model = self.model_fn(self.input_shape, self.num_classes)
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        return self

    def predict(self, X):
        if self.model is None:
            raise Exception("Model needs to be fitted before prediction.")
        y_prob = self.model.predict(X)
        print(y_prob)
        y_pred = np.argmax(y_prob, axis=1)
        print(y_prob)
        y_pred_prob = np.max(y_pred, axis=1)
        return [y_pred, y_pred_prob]  # わんちゃんy_predは予測ラベルのインデックスなのでいらないかも（予測ラベルの確率でアンサンブルした方が良き）

#######################################################################

#                                実行部

#######################################################################

def main():
    # 訓練データのロード
    x_train, y_train, class_names = load_images_from_folder(TRAIN_PATH)
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)

    # テストデータのロード
    x_test, y_test, _ = load_images_from_folder(TEST_PATH)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    for epochs in EPOCH_NUMS:
        # Kerasモデルのラッパーを作成
        alexnet_clf = KerasClassifier(model_fn=alexnet, input_shape=(224, 224, 3), num_classes=6, learning_rate=LEARNING_RATE["AlexNet"])
        vgg16_clf = KerasClassifier(model_fn=vgg16, input_shape=(224, 224, 3), num_classes=6, learning_rate=LEARNING_RATE["VGG16"])
        resnet50_clf = KerasClassifier(model_fn=resnet50, input_shape=(224, 224, 3), num_classes=6, learning_rate=LEARNING_RATE["ResNet50"])

        # 個々のモデルをトレーニング
        alexnet_clf.fit(x_train, y_train, epochs, BATCH_SIZES["AlexNet"])
        vgg16_clf.fit(x_train, y_train, epochs, BATCH_SIZES["VGG16"])
        resnet50_clf.fit(x_train, y_train, epochs, BATCH_SIZES["ResNet50"])

        # テストデータに対する予測
        y_test_pred_alexnet = alexnet_clf.predict(x_test)  # こいつらはすべてリストにした。(0が予測ラベル、1が予測ラベルの確率)
        y_test_pred_vgg16 = vgg16_clf.predict(x_test)
        y_test_pred_resnet50 = resnet50_clf.predict(x_test)

        # アンサンブルによる最終予測（多数決）
        y_test_pred = []
        for i in range(len(y_test_pred_alexnet)):
            preds = [y_test_pred_alexnet[i], y_test_pred_vgg16[i], y_test_pred_resnet50[i]]
            y_test_pred.append(np.bincount(preds).argmax())

        # 混同行列の保存
        save_confusion_matrix(np.argmax(y_test, axis=1), y_test_pred, class_names, SAVE_PATH, epochs=epochs)

        # 各モデルの精度を計算
        accuracy_alexnet = accuracy_score(np.argmax(y_test, axis=1), y_test_pred_alexnet)
        accuracy_vgg16 = accuracy_score(np.argmax(y_test, axis=1), y_test_pred_vgg16)
        accuracy_resnet50 = accuracy_score(np.argmax(y_test, axis=1), y_test_pred_resnet50)
        accuracy_ensemble = accuracy_score(np.argmax(y_test, axis=1), y_test_pred)

        # 各モデルとアンサンブルの精度を棒グラフで保存
        models = ['AlexNet', 'VGG16', 'ResNet50', 'Ensemble']
        accuracies = [accuracy_alexnet, accuracy_vgg16, accuracy_resnet50, accuracy_ensemble]

        plt.figure(figsize=(8, 6))
        bars = plt.bar(models, accuracies, color=['c', 'cadetblue', 'skyblue', 'dodgerblue'])
        plt.ylim(0.0, 1.0)
        plt.title(f'Test Accuracies (Epochs: {epochs})')
        plt.ylabel('Accuracy')

        # 各棒の上に精度の数値を表示
        for i, acc in enumerate(accuracies):
            plt.text(i, acc, f'{acc:.4f}', ha='center', va='bottom', fontsize=11, color='black')

        plt.savefig(os.path.join(SAVE_PATH, f'test_accuracies_ep{epochs}.png'))
        plt.close()

if __name__ == "__main__":
    main()

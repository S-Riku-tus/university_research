import os
import random
import shutil

def split_data(source_folder, train_folder, test_folder, train_ratio=0.8):
    # train_folder, test_folder を作成（存在しない場合）
    for folder in [train_folder, test_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # フォルダ内の.pngファイルをリスト化
    filenames = [filename for filename in os.listdir(source_folder) if filename.endswith(".png")]

    random.shuffle(filenames)  # ファイルをシャッフルしてランダムに分割

    # データセットのサイズを計算
    num_files = len(filenames)
    num_train = int(num_files * train_ratio)

    # データを分割して移動
    for i, filename in enumerate(filenames):
        source_path = os.path.join(source_folder, filename)
        if i < num_train:
            target_path = os.path.join(train_folder, filename)
        else:
            target_path = os.path.join(test_folder, filename)
        
        shutil.move(source_path, target_path)  # source_path で指定したファイルやディレクトリを target_path に移動

# 元のスペクトログラムフォルダのパス
source_folder = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\6月26日_サブクール度10度_0.3mm\スペクトログラム_熱流束"
# 分割後のフォルダのパス
train_folder = os.path.join(source_folder, "train")
test_folder = os.path.join(source_folder, "test")

# データを分割する
split_data(source_folder, train_folder, test_folder)

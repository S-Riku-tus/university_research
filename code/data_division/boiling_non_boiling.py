import os
import re
import shutil

#######################################################################

#                            数字でソートする関数

#######################################################################

def numerical_sort(value):
    # 文字列から数字の部分を抽出して整数型に変換する関数
    numbers = re.findall(r'\d+', value)
    return list(map(int, numbers)) if numbers else [0]

#######################################################################

#                           沸騰と非沸騰を分割する関数

#######################################################################

def split_train_data(source_folder, split_filename1, split_filename2):
    # フォルダ内のファイルを数値でソートする
    filenames = sorted(os.listdir(source_folder), key=numerical_sort)

    # 指定されたファイルが存在するか確認する
    if split_filename1 not in filenames or split_filename2 not in filenames:
        print(f"ファイル '{split_filename1}' または '{split_filename2}' がフォルダ '{source_folder}' 内に見つかりませんでした。")
        return

    # 指定されたファイルのインデックスを取得する
    split_index1 = filenames.index(split_filename1)
    split_index2 = filenames.index(split_filename2)

    # others フォルダを作成する
    others_folder = os.path.join(source_folder, "others")
    os.makedirs(others_folder, exist_ok=True)

    # 非沸騰の画像をリネーム
    for i in range(split_index1 + 1):
        file_path = os.path.join(source_folder, filenames[i])
        if os.path.isfile(file_path):
            new_filename = f"not_boiling_{filenames[i]}"
            new_file_path = os.path.join(source_folder, new_filename)
            os.rename(file_path, new_file_path)

    # 沸騰の画像をリネーム
    boiling_count = [0] * 5
    boiling_limit = (split_index2 - split_index1) // 5
    boiling_mod = (split_index2 - split_index1) % 5

    for i in range(split_index1 + 1, split_index2 + 1):
        file_path = os.path.join(source_folder, filenames[i])
        if os.path.isfile(file_path):
            for j in range(5):
                if boiling_count[j] < boiling_limit or (j < boiling_mod and boiling_count[j] == boiling_limit):
                    new_filename = f"boiling{j+1}_{filenames[i]}"
                    new_file_path = os.path.join(source_folder, new_filename)
                    os.rename(file_path, new_file_path)
                    boiling_count[j] += 1
                    break

    # それ以降のファイルをothersフォルダに移動
    for i in range(split_index2 + 1, len(filenames)):
        file_path = os.path.join(source_folder, filenames[i])
        if os.path.isfile(file_path):
            new_file_path = os.path.join(others_folder, filenames[i])
            shutil.move(file_path, new_file_path)

#######################################################################

#                                 実行

#######################################################################

# 訓練用データのフォルダパス
source_folder = r"C:\Users\Casper4\Python\Ueki\shibasaki\研究\6月26日_サブクール度10度_0.3mm\スペクトログラム"
# 指定する2つのファイル名
split_filename1 = "1.2V_51.png"
split_filename2 = "2.4V_56.png"

# 訓練データを分割する
split_train_data(source_folder, split_filename1, split_filename2)

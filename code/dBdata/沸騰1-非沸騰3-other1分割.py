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

def split_train_data(source_folder, split_filename):
    # フォルダ内のファイルを数値でソートする
    filenames = sorted(os.listdir(source_folder), key=numerical_sort)

    # 指定されたファイルが存在するか確認する
    if split_filename not in filenames:
        print(f"ファイル '{split_filename}' がフォルダ '{source_folder}' 内に見つかりませんでした。")
        return

    # 指定されたファイルのインデックスを取得する
    split_index = filenames.index(split_filename)

    # 非沸騰の画像を1つ、沸騰の画像を各3つに分けて名前を変更して移動する
    count_not_boiling = 0
    count_boiling1 = 0
    count_boiling2 = 0
    count_boiling3 = 0

    for i, filename in enumerate(filenames):
        file_path = os.path.join(source_folder, filename)
        if os.path.isfile(file_path):
            if i <= split_index and count_not_boiling < 1:
                # 非沸騰の画像として名前を変更
                new_filename = f"not_boiling_{filename}"
                new_file_path = os.path.join(source_folder, new_filename)
                os.rename(file_path, new_file_path)
                count_not_boiling += 1
            elif split_index < i <= split_index + 3:
                # 沸騰の画像として名前を変更
                if count_boiling1 < 1:
                    new_filename = f"boiling1_{filename}"
                elif count_boiling2 < 1:
                    new_filename = f"boiling2_{filename}"
                elif count_boiling3 < 1:
                    new_filename = f"boiling3_{filename}"
                else:
                    continue
                new_file_path = os.path.join(source_folder, new_filename)
                os.rename(file_path, new_file_path)
                count_boiling1 += 1
                count_boiling2 += 1
                count_boiling3 += 1

#######################################################################

#                                 実行

#######################################################################

# 訓練用データのフォルダパス
source_folder = r"C:\Users\shiba\研究\6月26日_サブクール度10度_0.3mm\スペクトログラム_等分割"
# 指定する1つのファイル名
split_filename = "1.2V_51.png"

# 訓練データを分割する
split_train_data(source_folder, split_filename)

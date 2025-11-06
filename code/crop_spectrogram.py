import numpy as np
from skimage import io, util
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.transform import resize # ★ 1. resize をインポート
import os

def analyze_spectrogram_skimage(image_path, output_dir):
    """
    scikit-image で画像を処理し、
    (224, 224) のグレースケールNPYとして保存する。
    """
    
    # --- 出力ディレクトリの準備 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"ディレクトリ '{output_dir}' を作成しました。")

    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # ★ 2. モデルのターゲット形状を定義
    TARGET_SHAPE = (224, 224)

    # ----------------------------------------------------
    # ステップ1 & 2: 画像の読み込みとグレースケール変換
    # ----------------------------------------------------
    
    try:
        image_color = io.imread(image_path)
    except FileNotFoundError:
        print(f"エラー: ファイル {image_path} が見つかりません。")
        return
    except Exception as e:
        print(f"エラー: 画像 {image_path} の読み込み中に問題が発生しました: {e}")
        return

    # RGBA (4ch) -> RGB (3ch) に変換
    if image_color.shape[2] == 4:
        image_color = np.delete(image_color, 3, axis=2) 

    # ★ 3. グレースケールデータ (0-255 の uint8) をNPYの元データとする
    gray_ubyte = util.img_as_ubyte(rgb2gray(image_color))

    # ----------------------------------------------------
    # ステップ3: 閾値処理
    # ----------------------------------------------------
    
    thresh_value = threshold_otsu(gray_ubyte)
    mask = gray_ubyte > thresh_value
    
    print(f"自動検出された閾値 (skimage): {thresh_value}")
    
    # --- マスク画像を .npy として保存 ---
    mask_path = os.path.join(output_dir, f"{base_filename}_mask.npy")
    # ★ 4. リサイズ処理を追加
    mask_data_uint8 = util.img_as_ubyte(mask) # (360, 360) の 0/255
    resized_mask = resize(mask_data_uint8, TARGET_SHAPE, anti_aliasing=True)
    # 推論スクリプトが /255 する前提のため、uint8 (0-255) に戻す
    np.save(mask_path, util.img_as_ubyte(resized_mask))

    # ----------------------------------------------------
    # 方法A: マスキング (グレースケール)
    # ----------------------------------------------------
    
    # ★ 5. カラー(3ch)ではなく、グレースケール(1ch)をマスクする
    masked_gray_image = gray_ubyte.copy()
    masked_gray_image[~mask] = 0 # (360, 360)
    
    # ★ 6. リサイズ処理を追加
    resized_masked_gray = resize(masked_gray_image, TARGET_SHAPE, anti_aliasing=True)
    
    masked_path = os.path.join(output_dir, f"{base_filename}_masked.npy")
    # uint8 (0-255) に戻して保存
    np.save(masked_path, util.img_as_ubyte(resized_masked_gray))
    print(f"マスキングしたNPYを '{masked_path}' として保存しました。")

    # ----------------------------------------------------
    # 方法B: クロッピング (グレースケール)
    # ----------------------------------------------------
    
    rows, cols = np.where(mask)
    
    if rows.size > 0 and cols.size > 0:
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()
        
        # ★ 7. カラー(3ch)ではなく、グレースケール(1ch)をクロップ
        cropped_gray_image = gray_ubyte[y_min:y_max+1, x_min:x_max+1] # (H, W)
        
        # ★ 8. リサイズ処理を追加
        resized_cropped_gray = resize(cropped_gray_image, TARGET_SHAPE, anti_aliasing=True)
        
        cropped_path = os.path.join(output_dir, f"{base_filename}_cropped.npy")
        # uint8 (0-255) に戻して保存
        np.save(cropped_path, util.img_as_ubyte(resized_cropped_gray))
        print(f"クロッピングしたNPYを '{cropped_path}' として保存しました。")
    else:
        print("クロッピング: 強度の高い部分が検出されませんでした。")

# --- メイン処理 ---
if __name__ == "__main__":
    
    # -----------------------------------------
    # ★ 設定項目 ★
    # -----------------------------------------
    
    # 1. 入力する画像ファイル
    input_filename = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\regression_result\npy\ensemble\heatflux_no_noise\pre_20251020_ep100_bs12_lr0.001\4.77e+05_45.png"

    # 2. 出力先のフォルダ名
    output_folder = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2024.11.12_1_2.13_1\regression_result\npy\ensemble\heatflux_no_noise\pre_20251020_ep100_bs12_lr0.001"
    
    # -----------------------------------------
    
    if not os.path.exists(input_filename):
        print(f"エラー: 入力ファイル {input_filename} が見つかりません。")
    else:
        analyze_spectrogram_skimage(input_filename, output_folder)
# 修士論文草稿（2026年10月19日）

[PDFを読む](build/main.pdf) ／ [本文の入口](main.tex) ／ [不足・判断事項の説明](../../docs/thesis/2026-10-19_latex_manuscript_and_remaining_work.md)

要旨・6章・謝辞欄・参考文献・付録まで文章化した執筆用草稿です。未確認事項は本文の赤字「要確認：ID」と付録Bで追跡できます。題目・提出様式は仮です。

目的はONB近傍の検知に置き、統合の全域回帰への利得と、ONB発生段階の未達を併記しました。第2章は既存データの出典と解析条件です。実験取得者を推測していません。

## 編集とPDF作成

PowerShellでこのディレクトリへ移動し、次を実行します。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1
```

TectonicがPATHにない場合、公式配布のWindows版0.17.0をこのディレクトリ内の `.tools/` にダウンロードし、SHA-256を照合して使用します。初回はパッケージ・日本語フォントの取得にネット接続が必要です。キャッシュは `.cache/tectonic/` に置きます。管理者権限やグローバルPATHの変更は不要です。生成先は `build/main.pdf` です。

別のTectonic実行ファイルを指定する場合：

```powershell
.\build.ps1 -EnginePath 'C:\path\to\tectonic.exe'
```

XeLaTeXのあるTeX Live環境でも組版できます。和文クラスは `bxjsreport`、フォントは原ノ味、欧文はTeX Gyreです。本文はUTF-8、参考文献は `thebibliography` のためBibTeXを必要としません。通常のTeX環境では `xelatex -output-directory=build main.tex` を参照が安定するまで実行します。

VS Codeではこの `master_thesis` ディレクトリをフォルダーとして開き、LaTeX Workshopがある場合に同梱のローカル設定を利用できます。拡張機能を使わずターミナルだけでも組版できます。既存のユーザー設定やワークスペース全体の設定は変更していません。

## ファイル構成

- `chapters/`：緒言、使用データ、手法、性能評価、考察、結言。
- `frontmatter.tex`：仮表紙、草稿の位置づけ、和英要旨。
- `references.tex`：確認した論文・学位論文・保存結果の出典。要旨までの確認は明記。
- `appendices/`：再現条件・補足、不足ID一覧と条件付き検証。
- `tables/`、`figures/`：保存値から作成した本文用の図表。
- `data/`：小さい出典表のコピーとSHA-256 manifest。
- `scripts/prepare_materials.py`：保存CSV・予測から組版素材を作る。モデルfit・推論なし。
- `scripts/check_manuscript.py`：参照、出典、数値表、PDFの確認。

通常の本文修正には素材再生成は不要です。素材を更新する場合だけ、研究用Python環境（NumPy・Matplotlibあり）で次を実行します。元結果の保存先が必要です。

```powershell
python .\scripts\prepare_materials.py
python .\scripts\check_manuscript.py
```

PDFの文字とページの追加検査は `python .\scripts\check_manuscript.py --pdf` で行います（PyMuPDFが必要）。原結果や先輩のPDFは変更しません。生データや学習済みモデルの再生成を行うコマンドは含めていません。

## 元のLaTeX資料との関係

`../sample/` は三上氏の別研究のサンプル、`../setting/LaTeX_setting.pptx` は2023年時点の導入資料でした。二段組の発表用原稿をそのまま修論本体にはせず、独立した和文レポート構成にしています。古い設定にあるユーザー全体のセキュリティ・クリーン設定はコピーしていません。もとの `sample`、`setting`、`Elsevier template` は保持しています。

組版方式の根拠：[Tectonic公式導入資料](https://tectonic-typesetting.github.io/book/latest/installation/)、[キャッシュの公式説明](https://tectonic-typesetting.github.io/book/latest/getting-started/first-document.html)。TeX本体が見つからなかったことは、過去の環境が削除された証明ではありません。今回確認したPATH・一般的な設置先では見つからなかったため、局所的に用意しました。

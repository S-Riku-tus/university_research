"""Build a read-only-source, page-by-page assembly guide and layout previews."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import re
import zipfile
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
ASSETS = ROOT / "docs/progress/assets/2026-10-16"
PREVIEW = ASSETS / "layout_preview"
GUIDE = ROOT / "docs/progress/2026-10-16_slide_build_guide.md"
OLD = ROOT / "experiments/2026-10-14_error_feature_linkage"
PPTX = next(ROOT.glob("*/2026/1016*/*1016.pptx"))
FONT = "C:/Windows/Fonts/YuGothM.ttc"
fontManager.addfont(FONT)
PROP = FontProperties(fname=FONT)
plt.rcParams.update({"font.family": PROP.get_name(), "pdf.fonttype": 42})
W, H = 20.0, 11.25
BLUE, GRAY, PALE, AMBER = "#1685c4", "#526779", "#eaf4fb", "#a76c24"
SLIDES = []


def slide(number, title, claim, role, oral, transition, caution, footer=""):
    item = dict(number=number, title=title, claim=claim, role=role, oral=oral,
                transition=transition, caution=caution, footer=footer, objects=[])
    SLIDES.append(item)
    return item


def text(s, name, value, x, y, width, height, size=26, color="black", bold=False):
    s["objects"].append(dict(kind="text", name=name, text=value,
        rect=[x, y, width, height], font_pt=size, color=color, bold=bold))


def image(s, name, source, x, y, width):
    with Image.open(source) as im:
        height = width * im.height / im.width
    s["objects"].append(dict(kind="image", name=name, source=str(source.relative_to(ROOT)).replace("\\", "/"),
                             rect=[x, y, width, height]))


def box(s, name, value, x, y, width, height, size=26, face=PALE):
    s["objects"].append(dict(kind="box", name=name, text=value, rect=[x, y, width, height],
                             font_pt=size, color=GRAY, face=face))


def table(s, name, cells, x, y, widths, row_height=.65, size=24):
    s["objects"].append(dict(kind="table", name=name, cells=cells,
        rect=[x, y, sum(widths), row_height*len(cells)], widths=widths,
        row_height=row_height, font_pt=size))


s = slide(1, "表紙", "", "研究題目と今回の日付を示す。既存の表紙を保持する。",
          "今回は、雑音問題の整理、モデル追加の結果、改善理由の分析、修論への配置を報告する。",
          "まず、今回の結果が研究目的のどこに位置するかを説明する。",
          "下記は配置見本の再現用。実作業では本人の表紙をそのまま使い、作り直さない。")
text(s, "資料種別", "研究進捗", 1.8, .85, 16.5, .8, 30)
text(s, "題目1", "アンサンブル学習に基づく音響識別による", 2.0, 3.7, 16.8, .95, 44)
text(s, "題目2", "プール沸騰の発生検知", 2.0, 5.0, 16.8, 1.25, 66)
text(s, "所属氏名", "植木研究室　柴崎陸", 2.0, 7.3, 16.8, .7, 30)
text(s, "日付", "2026年10月16日", 2.0, 8.25, 16.8, .7, 28)

s = slide(2, "研究の概要と目的",
          "音響から熱流束・ONBを推定し，雑音下での統合の働きを説明する",
          "目指す結論を示し、後続の性能評価と要因分析の目的を共有する。",
          "熱流束を回帰し、その予測からONBを判定する。異なるモデルを組み合わせて何が改善するか、特徴利用と誤り方から説明する。全方式に勝つことは達成条件に置かない。",
          "その評価へ入る前に、過去に生じた雑音増加時の精度逆転を整理する。",
          "図は研究の処理系。音響だけから真の気泡発生秒を同定できたという図ではない。",
          "今回の主系列：2025/6/11＋6/18の音響データ")
image(s, "研究全体の処理系", ASSETS/"overview_pipeline.png", 1.65, 3.15, 17.6)
text(s, "評価の三観点", "確認する点：推定誤差／ONB判定／特徴利用と誤りの対応", 1.8, 9.3, 17.2, .7, 28)

s = slide(3, "1. 雑音増加時の精度逆転",
          "旧処理には，付加雑音だけでも熱流束を予測できる経路があった",
          "2025年データで起きた逆転を、処理側の対照から説明する。",
          "棒を左から読む。相対雑音では元音響を除いて雑音だけを与えてもR²が高かった。元信号の強度に応じて付加雑音も変わる経路があり、固定基準の付加へ修正した。診断はPCAモデルの結果で、今日の5モデルの性能ではない。",
          "生成方法の修正後にも回復が残るため、次に学習方針を区別する。",
          "この対照は旧処理の情報経路を示す。すべての旧runの非単調性の原因を一つに確定したものではない。",
          "診断条件：2025/6/18・22 kHz・PCA診断モデル・通常KFold（現在の5モデルと別条件）")
image(s, "旧処理の4条件対照", OLD/"slide_old_noise_shortcut.png", 1.6, 3.35, 10.9)
text(s, "棒の日本語対応", "左から：無雑音／相対混合／雑音のみ／固定混合", 1.8, 9.2, 11.0, .65, 22)
box(s, "経路1", "元信号の強度", 13.1, 3.45, 5.6, .95)
text(s, "矢印1", "↓", 15.55, 4.45, 1.0, .65, 32)
box(s, "経路2", "付加雑音の強度へ反映", 13.1, 5.15, 5.6, 1.0, 24)
text(s, "矢印2", "↓", 15.55, 6.2, 1.0, .65, 32)
box(s, "経路3", "雑音のみでも予測可能", 13.1, 6.9, 5.6, 1.0, 24)
text(s, "診断結果", "雑音のみ：R² = 0.9085", 13.1, 8.45, 5.8, .7, 25, BLUE, True)

s = slide(4, "1. 修正後の挙動と評価方針",
          "同じ学習済みモデルへ雑音を加え，雑音の影響を評価する",
          "生成修正と、同じモデルへの雑音評価を分けて説明する。",
          "左の旧matchedでは、生成を直しても一部で回復した。matchedは雑音ごとに学習する。一方、今回のclean_onlyは無雑音で学習したモデル・前処理・重みを固定する。これにより同じモデルの雑音への応答を読む。",
          "この方針の下で、実際に使用した分割とモデル構成を示す。",
          "左の旧22 kHz・別日matchedと、今回の3 kHz・日内clean_onlyは一要因比較ではない。日内なら非単調性が起きない、という保証もない。",
          "左：9/18実行・6/11→6/18・22 kHzの旧matched／右：今回のclean_onlyの評価方針")
text(s, "左見出し", "修正後の旧matched：回復は一部に残る", 1.7, 2.85, 10.5, .65, 25)
image(s, "修正後matchedの曲線", OLD/"slide_after_generation_fix_matched.png", 1.6, 3.75, 9.7)
text(s, "右見出し", "今回：clean_only", 11.7, 2.85, 7.3, .65, 27, BLUE, True)
box(s, "今回の学習", "無雑音のみで学習", 11.8, 3.75, 7.25, 1.0, 25)
text(s, "固定評価の矢印1", "↓", 15.1, 4.8, .9, .5, 26)
box(s, "固定する対象", "モデル・前処理・統合重みを固定", 11.8, 5.3, 7.25, 1.0, 24)
text(s, "固定評価の矢印2", "↓", 15.1, 6.35, .9, .5, 26)
box(s, "今回の評価", "同じ評価区間へ雑音を付加", 11.8, 6.85, 7.25, 1.0, 24)
text(s, "評価する雑音", "clean／0／−4／−8／−12／−16／−20 dB", 11.8, 8.2, 7.3, .65, 21)
text(s, "このページの結び", "生成の修正と，雑音条件ごとに学習し直す影響を区別する", 1.8, 9.5, 17.0, .7, 26)

s = slide(5, "2. 現在の評価条件と統合構成",
          "2025年6月の未使用区間で，元3・4・5モデルを評価する",
          "結果を読むために必要な条件と、追加したモデルの位置を示す。",
          "6/11と6/18の36録音を使い、各録音を1秒ごとの60chunkへ分け、45を学習、15を評価に使う。計1620学習・540評価。各録音内のシャッフル分割で、連続した15秒を丸ごと保持する図ではない。元3へHGBまたはETを加えた4、両方を加えた5を比較する。重みは学習側の全域MSEに基づき求める。",
          "まず、この構成変更で全域の回帰誤差がどう変わったかを示す。",
          "既知録音の未使用chunkの評価で、未知日・未知録音の性能ではない。RFは表示名で実装はPCA＋XGBRF、従来Conformer表記は現行CNN＋Transformerに対応。",
          "各録音内のシャッフル分割：学習45／評価15chunk。既知録音の未使用区間を評価（未知日は対象外）")
text(s, "重みの決定", "統合重みは学習側の全域MSEに基づき決定し，評価中は固定", 1.8, 2.65, 17.2, .7, 24)
image(s, "分割と3・4・5モデル構成", ASSETS/"evaluation_and_models.png", 1.65, 3.45, 17.6)

s = slide(6, "2. 熱流束回帰の結果",
          "5モデル統合は元3より全7条件で全域RMSEが低下した",
          "今回の主成果を全雑音の曲線と具体値で示す。",
          "青の元3 MSEと黒の5を全7条件で比較する。無雑音34.07→27.80、−20で81.14→75.23。4や追加単体を含めた最良方式は条件で変わる。雑音下の絶対誤差の改善と、無雑音からの悪化量は別で、後者は元3 MSE比で47.07→47.43だった。",
          "全域の回帰改善が、ONB判定でも同じ形で現れるかを見る。",
          "元3はMSE統合。従来performance統合と混ぜない。無雑音/−20だけの棒図から全7条件を主張しない。",
          "全域RMSE・同じ540chunk・無雑音学習固定。元3＝MSE統合。2025/6/11＋6/18・1秒・3 kHz")
image(s, "全7雑音の回帰曲線", ASSETS/"current_regression_noise_trend.png", 1.6, 3.15, 11.2)
table(s, "端点のRMSE", [["条件", "元3", "5モデル"], ["無雑音", "34.07", "27.80"], ["−20 dB", "81.14", "75.23"]],
      13.25, 3.3, [1.9, 1.8, 2.1], .78, 24)
text(s, "表の単位", "単位：kW/m²（小さいほど良い）", 13.25, 5.9, 5.9, .8, 21)
text(s, "条件差", "最良方式は条件で異なる\n無雑音：ET4 = 27.35\n−20 dB：HGB単体 = 74.19", 13.25, 7.0, 6.0, 1.9, 23)
text(s, "改善率", "元3 MSE比：無雑音18.4%，−20 dB7.3%のRMSE低下", 1.8, 9.35, 17.0, .75, 27, BLUE, True)

s = slide(7, "2. ONB判定の結果",
          "元3より見逃しは減ったが，検知指標の改善は一律ではない",
          "全域回帰・見逃し・q100を分け、検知上の成果と限界を示す。",
          "左の見逃しは無雑音42→27、−20で65→52。元3と5のFPは無雑音1→1、−20で0→0。右の6/18無雑音q100は元3434.02、5は376.32で、ET4の322.11の方が早い。6/11は4構成とも368.98。さらに最初のONB段階をほぼ全検知した状態ではない。",
          "改善が一律でない理由を、領域別の誤差と入力依存から説明する。",
          "q100は従来の高q側100%到達点と同じ理解でよい。FP=0を条件に含める新定義にはしない。ONB段階30chunkは2録音由来で、30独立実験ではない。",
          "日別ONB閾値。ONB以上315／ONB前225chunk。最初のONB段階30chunkは2録音由来")
image(s, "無雑音・強雑音の見逃し", ASSETS/"current_onb_false_negatives.png", 1.6, 3.2, 10.75)
text(s, "q100表見出し", "6/18・無雑音のq100［kW/m²］", 12.9, 2.95, 6.2, .75, 23)
table(s, "日別q100", [["構成", "q100"], ["元3 MSE", "434.02"], ["元3＋HGB", "376.32"], ["元3＋ET", "322.11"], ["5モデル", "376.32"]],
      12.9, 3.9, [3.65, 2.5], .62, 23)
text(s, "6/11の結果", "6/11は4構成とも368.98", 12.9, 7.25, 6.1, .65, 23)
text(s, "誤報の併記", "FP：無雑音1→1\n　　 −20 dB0→0（元3→5）", 12.9, 8.05, 6.1, 1.2, 23)
text(s, "ONB段階での限界", "最初のONB段階の検知：5モデルで無雑音8/30，−20 dB0/30 chunk", 1.8, 9.6, 17.2, .65, 23, AMBER)

s = slide(8, "3. 統合の改善が生じた理由",
          "統合の効果は領域で異なり，逆向きの誤差相殺だけでは説明できない",
          "性能の数値を、誤りの調整と入力依存の違いへ接続する。",
          "左は正が元3→5のMSE改善、負が悪化。無雑音では近傍と高q、−20ではONB前と高qが改善する。右は無雑音で時間順序を並べ替えた既存の対照。CNN系は変化し、時間集計のHGB/ETは不変。さらに−20のMSE減少の87.15%は3ブロックが同方向に外れた標本群から生じ、逆向き相殺以外も寄与する。",
          "このような接続を、前回指摘された説明性の活用への回答にする。",
          "87.15%は元3/HGB/ETの3ブロックによる損失の加算分解で、5単体の符号分類や因果割合ではない。時間並べ替えと水流雑音は別の外乱。物理的音源の確定はしていない。",
          "左：元3→5のMSE減少（現行結果でも確認）／右：10/7同条件・別fitの時間並べ替え対照")
text(s, "左図見出し", "どの領域が改善したか", 2.5, 2.75, 7.8, .6, 24)
text(s, "右図見出し", "どの入力変化に依存するか", 11.45, 2.75, 7.6, .6, 24)
image(s, "領域別利得と時間配置依存", OLD/"slide_error_and_feature_linkage.png", 1.6, 3.35, 17.6)
text(s, "同方向誤差の寄与", "−20 dBのMSE減少の87.15%は，同方向の誤差群から生じた", 1.8, 9.75, 17.1, .65, 25, BLUE, True)

s = slide(9, "3. 説明性の意味と研究への活用",
          "説明図を，成功・失敗と統合の理由を説明する根拠へつなぐ",
          "教授の前回の指摘を1枚で再掲し、今回どう回答が進んだかを示す。",
          "指摘は、何を意味し、修士研究へどう使うか。左のマスクは帯域変更への感度、右のIGは基準入力からの出力差への寄与で意味が異なる。今回、誤差の領域差と共通入力加工の対照へ接続した。旧IGを再掲しただけで今日の5モデルの補完を直接証明したとは扱わない。",
          "以上の性能と理由の分析を、修論の章へ配置する。",
          "旧画像は3点とも縮尺固定で配置する。IGは個別正規化された色で、モデル間の絶対寄与量を色の明るさだけで比べない。細かい数値は原図で確認する。",
          "9/18旧図の再掲：マスクとIGは条件が異なる。IGは22 kHz・0 dB・ONBの同一入力。最新fitの図ではない")
text(s, "教授の指摘", "前回の指摘：何を意味するか／修士研究にどう活用するか", 1.8, 2.75, 17.1, .7, 26)
text(s, "マスク見出し", "周波数帯マスク", 1.8, 3.7, 7.3, .6, 25)
text(s, "IG見出し", "IG：AlexNet", 9.65, 3.7, 4.4, .6, 24)
text(s, "IG見出し2", "IG：CNN系", 14.45, 3.7, 4.4, .6, 24)
image(s, "旧マスク図", ASSETS/"mask_recap_0918.png", 1.65, 4.5, 7.4)
image(s, "旧AlexNet IG", ASSETS/"ig_alexnet_recap_0918.png", 9.65, 4.5, 4.4)
image(s, "旧CNN系 IG", ASSETS/"ig_transformer_recap_0918.png", 14.45, 4.5, 4.4)
text(s, "マスクの意味", "帯域の変更に対する感度\n→ 利用する特徴の違い", 1.8, 8.05, 7.5, 1.2, 23)
text(s, "IGの意味", "基準入力からの出力差への寄与\n→ 同じ入力でも寄与分布が異なる", 9.65, 8.65, 9.2, 1.2, 23)
text(s, "今回の活用", "今回：入力加工と領域別誤差を対応づけ，統合の効果を説明する", 1.8, 9.9, 17.1, .55, 24, BLUE, True)

s = slide(10, "4. 修論の構成と現在の成果",
          "性能結果と理由の分析を中心に，既存成果を修論へ配置する",
          "研究全体の筋と、現在ある成果の配置を明示する。",
          "第1〜4章で背景・データ・方法・雑音評価の判断を整える。第5章に性能、第6章に誤りと特徴の接続、第7章に主張と限界を置く。表は成果の状態であり、章の執筆完了率ではない。",
          "その構成へまとめるため、後期計画の残りを更新する。",
          "目次は案。既存成果の原資料を保持し、過去条件と現行条件をそれぞれの条件として記述する。",
          "表は配置可能な成果の状態。本文執筆済みを示すものではなく，章番号・目次は案")
table(s, "修論の章と成果", [
    ["章", "内容", "配置する成果の状態"],
    ["1", "緒論・研究目的", "目的と問いを整理済み"],
    ["2", "実験・音響データ", "原資料・データ条件あり"],
    ["3", "回帰・ONB評価方法", "現行方法・指標を整備済み"],
    ["4", "雑音評価条件の検討", "逆転診断・学習方針の対照済み"],
    ["5", "単体・統合の性能評価", "元3/4/5・回帰・ONB結果あり"],
    ["6", "誤りと特徴の要因分析", "接続分析済み・物理的解釈を整理"],
    ["7", "総合考察・結言", "全体の主張と限界をまとめる"],
], 1.65, 3.15, [1.2, 6.1, 10.2], .7, 24)
text(s, "中核となる章", "第5章で「何が改善したか」，第6章で「なぜ改善したか」を示す", 1.8, 9.35, 17.1, .75, 27, BLUE, True)

s = slide(11, "4. 後期の研究計画",
          "主系列の検証を区切り，説明・考察と修論への整理を進める",
          "元の4区分を保ち、完了成果・現在以降の作業・候補を分けて示す。",
          "評価と3→4→5の実行・解析、説明性の接続分析は完了側へ移す。今後は採用理由、特徴と誤りの解釈、章別整理と本文への展開を進める。10〜11月は整理と本文展開、12月は結果・考察・結論、翌1〜2月は完成と発表準備という既存計画を土台とする。自動重み切替は時間と費用に見合う場合だけの候補。",
          "相談するなら、現在の成果と説明の到達点で修論の軸が通るかを確認する。",
          "正式締切は未確認。本文執筆済み・教授了承済みとはしない。修論完成後の最終確認、新2026年評価、停止matchedは今回の予定へ追加しない。")
image(s, "進捗を反映した後期計画", ASSETS/"semester_plan_progress.png", 1.45, 3.0, 17.9)


def rect_cm(rect):
    return [round(v*2.54, 2) for v in rect]


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_link(path, base=GUIDE.parent):
    import os
    return Path(os.path.relpath(path, base)).as_posix()


def draw(slide, source_records):
    fig = plt.figure(figsize=(W, H), dpi=100, facecolor="white")
    # Approximate only the common frame; the source PPTX remains unchanged.
    fig.patches.append(Rectangle((.015, .025), .045, .95, transform=fig.transFigure,
                                  facecolor="#f1f5f8", edgecolor="none"))
    fig.text(.038, .52, "Tokyo University of Science", rotation=90, fontsize=12,
             ha="center", va="center", color=GRAY)
    fig.text(.037, .025, str(slide["number"]), fontsize=22, ha="center", color=GRAY)
    fig.text(.965, .972, "配置見本", fontsize=15, ha="right", va="top", color="#7c8993")
    bounds = []
    def add_text(name, value, rect, size, color="black", bold=False, center=False):
        x,y,w,h = rect
        artist=fig.text((x+w/2 if center else x)/W, 1-y/H, value,
                        fontsize=size, fontproperties=PROP, color=color,
                        va="top", ha="center" if center else "left",
                        linespacing=1.35, fontweight="bold" if bold else "normal")
        bounds.append((name, artist, rect))
    if slide["number"] != 1:
        add_text("共通タイトル", slide["title"], [1.98,.33,16.3,.85],30)
        add_text("共通結論", slide["claim"], [1.65,1.56,17.6,1.05],32)
        if slide["footer"]:
            add_text("条件注記", slide["footer"], [1.65,10.57,17.65,.48],18,GRAY)
    for item in slide["objects"]:
        x,y,w,h=item["rect"]
        if item["kind"] == "image":
            path=ROOT/item["source"]
            ax=fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])
            ax.imshow(plt.imread(path)); ax.axis("off")
            source_records[str(path.relative_to(ROOT)).replace("\\","/")]=hash_file(path)
        elif item["kind"] == "box":
            fig.patches.append(Rectangle((x/W,1-(y+h)/H),w/W,h/H,
                transform=fig.transFigure,facecolor=item["face"],edgecolor=BLUE,lw=1.5))
            add_text(item["name"],item["text"],[x+.16,y+.24,w-.32,h-.3],item["font_pt"],GRAY,center=True)
        elif item["kind"] == "table":
            height=item["row_height"]
            for row,values in enumerate(item["cells"]):
                cursor=x
                for col,value in enumerate(values):
                    width=item["widths"][col]
                    fig.patches.append(Rectangle((cursor/W,1-(y+(row+1)*height)/H),width/W,height/H,
                        transform=fig.transFigure,facecolor=PALE if row==0 else "white",edgecolor="#c1cfda",lw=.8))
                    add_text(f"{item['name']}_{row}_{col}",value,
                             [cursor+.13,y+row*height+.18,width-.26,height-.24],
                             item["font_pt"],GRAY if row==0 else "black",row==0)
                    cursor+=width
        else:
            add_text(item["name"],item["text"],item["rect"],item["font_pt"],item["color"],item["bold"])
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    overflow=[]
    for name,artist,rect in bounds:
        actual=artist.get_window_extent(renderer)
        x,y,w,h=rect
        if actual.width > w*fig.dpi+2 or actual.height > h*fig.dpi+2:
            overflow.append(dict(slide=slide["number"],name=name,
                                 actual_inches=[round(actual.width/fig.dpi,3),round(actual.height/fig.dpi,3)],
                                 available_inches=[w,h]))
    return fig,overflow


def write_guide():
    preview=relative_link(PREVIEW)
    lines=["# 10/16研究進捗発表：貼り付け文章と配置の作成ガイド", "",
        "更新日：2026-10-15。本人の『文章・図・全体配置を、見てすぐ作れる程度まで具体化する』依頼に対応。",
        "",
        f"[全11枚の配置見本PDF]({preview}/slide_layout_preview.pdf)／[一覧画像]({preview}/contact_sheet.png)／[基本の内容案](2026-10-16_slide_content_plan.md)／[結果と後期計画の根拠](2026-10-16_slide_readiness_and_semester_plan.md)。",
        "",
        "配置見本は実データの図と以下の文章を組み合わせた作成例。元PPTXの完全な描画ではなく、大学の装飾は簡略化している。PPTX原本・研究コード・元図・元結果は変更していない。",
        "",
        "## 共通の設定と作成順序", "",
        "元PPTXは16:9、幅50.8 cm×高さ28.575 cm（20×11.25 inch）。既存のサイズを保持する。以下の座標はスライド左上を原点とするcmで、PowerPointの『サイズと位置』に入力できる。画面幅が33.87 cmの別テンプレートへ移す場合は、全座標・幅・高さ・フォントを2/3に縮小する。",
        "",
        "| 共通要素 | 左 | 上 | 幅 | 高さ | フォント |", "|---|---:|---:|---:|---:|---|",
        "| 章・ページタイトル | 5.03 | 0.84 | 41.40 | 2.16 | 30 pt |",
        "| 上部の結論1文 | 4.19 | 3.96 | 44.70 | 2.67 | 32 pt |",
        "| 条件の注記 | 4.19 | 26.85 | 44.83 | 1.22 | 18 pt |",
        "| 左の大学装飾・ページ番号 | 元の配置を保持 | — | — | — | 元の書式 |", "",
        "本文は24〜28 pt、表は23〜24 ptを基本とする。注記を除き本文を小さくして詰め込まない。結論は黒、重要な今回の数値だけ青、結果の限界は落ち着いた茶色。画像は縦横比を固定し、図中の軸・単位・凡例を残す。グラフの色は元3青／HGB追加桃／ET追加緑／5黒を維持する。",
        "",
        "1. 作業用コピーで、表紙を保持し本文を11枚へ構成する。旧目次ページは今回の研究目的へ置き換える。",
        "2. 元5枚目など、結論＋図があるページを複製し、下記のタイトル・結論・本文を貼る。",
        "3. 指定画像を挿入し、縦横比固定で幅を設定、左・上の位置を設定する。表はPowerPointの表として作れば文字を編集しやすい。",
        "4. 古い日付・別日学習／評価・選別ありの記述を、現在の条件へ置き換える。旧診断や旧説明図には当時の条件を残す。",
        "5. 原本と見比べて大学装飾を保持し、最後にページ番号を1〜11へ揃える。", "",
        "主結果は10/15の143609。再実行194258は全7条件で丸め程度の差に収まり、見逃し・誤報・日別q100は完全一致。追加bar4点も使用可能。照合は完了済みで、新しい課題へ戻さない。", ""]
    for s in SLIDES:
        n=s["number"]
        lines += [f"## {n}枚目　{s['title']}", "", f"**このページの役割**：{s['role']}", "",
                  f"[配置見本]({preview}/slide_{n:02d}.png)", ""]
        if n != 1:
            lines += ["**タイトル欄に貼る文章**", "", "```text", s["title"], "```", "",
                      "**上部の結論欄に貼る文章**", "", "```text", s["claim"], "```", ""]
        lines += ["**本文・図・表の配置**（左・上・幅・高さ、単位cm）", ""]
        for item in s["objects"]:
            rect=rect_cm(item["rect"])
            lines += [f"- **{item['name']}**：左{rect[0]:.2f}／上{rect[1]:.2f}／幅{rect[2]:.2f}／高さ{rect[3]:.2f}。" +
                      (f"文字{item['font_pt']} pt。" if "font_pt" in item else "縦横比固定。"), ""]
            if item["kind"]=="image":
                lines += [f"  画像：[挿入する元画像]({relative_link(ROOT/item['source'])})。ファイル名 `{Path(item['source']).name}`。", ""]
            elif item["kind"]=="table":
                lines += ["  列幅："+"／".join(f"{v*2.54:.2f} cm" for v in item["widths"])+f"。1行の高さ{item['row_height']*2.54:.2f} cm。", "",
                          "| "+" | ".join(item["cells"][0])+" |", "| "+" | ".join(["---"]*len(item["cells"][0]))+" |"]
                lines += ["| "+" | ".join(row)+" |" for row in item["cells"][1:]]
                lines += [""]
            else:
                lines += ["```text",item["text"],"```",""]
        if s["footer"]:
            lines += ["**下端の条件注記に貼る文章**", "", "```text",s["footer"],"```",""]
        lines += [f"**口頭で説明する順序**：{s['oral']}","",
                  f"**次のページへのつなぎ**：{s['transition']}","",
                  f"**説明上の注意**：{s['caution']}",""]
    lines += ["## 作成時の判断と残る対応", "",
        "- 図の中に既にある凡例・数値を本文で全て繰り返さない。6の表と7の表は、聴衆に読んでほしい値だけを抜き出す。",
        "- 9は教授指摘への回答を主とする。旧画像の細かい数値を本編で読むのではなく、マスクとIGの意味の違い、現在の分析との接続を説明する。原寸図を拡大したい場合は元画像を予備資料へ置く。",
        "- 新しいbarを使う場合は6の補助・予備へ。全7条件の主張には全7条件の曲線を残す。Zoomed y-axisの表示を隠さず、±0.0000を実験の不確かさゼロと説明しない。",
        "- 日別ONBのFN/q100と、統合平均ONBを使う通常ROC/PR barを同一基準として混ぜない。",
        "- 11の『確立』は『検証』へ。現在を示す線は10/16の1本。完了作業を未来のタスクへ戻さず、自動重み切替は時期未定の条件付き候補へ置く。",
        "- 本ガイドは本編11枚を作る手順。持ち時間が未指定なので所要時間は確定していない。長い場合は基本案の短縮方針を使い、文字を小さくする前にページを統合する。",
        "",
        "今回完了：文章の具体化、図の実寸・縦横比調査、配置座標、図表の出典、11枚の見本、原本保持の検算。残る資料作業：PPTX本文への反映、実機での可読性と持ち時間の確認。追加学習・モデル推論・停止matchedの調査・新2026年評価は実行していない。", ""]
    GUIDE.write_text("\n".join(lines), encoding="utf-8")


def main():
    PREVIEW.mkdir(parents=True, exist_ok=True)
    original_hash=hash_file(PPTX)
    original_csv=ROOT/"experiments/2026-10-15_clean_only_result_analysis/pooled_metrics.csv"
    csv_hash=hash_file(original_csv)
    # Recheck actual source size and original text anchors, without editing Office.
    ns={"p":"http://schemas.openxmlformats.org/presentationml/2006/main"}
    with zipfile.ZipFile(PPTX) as archive:
        size=ET.fromstring(archive.read("ppt/presentation.xml")).find("p:sldSz",ns)
        assert int(size.attrib["cx"])/914400==W and int(size.attrib["cy"])/914400==H
    sources={}; overflows=[]
    geometry_failures=[]
    with PdfPages(PREVIEW/"slide_layout_preview.pdf") as pdf:
        for s in SLIDES:
            for item in s["objects"]:
                x,y,w,h=item["rect"]
                if min(x,y,w,h)<0 or x+w>W or y+h>H:
                    geometry_failures.append(dict(slide=s["number"],name=item["name"],rect=item["rect"]))
            fig,overflow=draw(s,sources)
            overflows+=overflow
            fig.savefig(PREVIEW/f"slide_{s['number']:02d}.png",dpi=100)
            pdf.savefig(fig)
            plt.close(fig)
    # A montage is a layout visualization; source image files are preserved.
    fig,axes=plt.subplots(4,3,figsize=(18,14),facecolor="#edf1f4")
    for ax,s in zip(axes.flat,SLIDES):
        ax.imshow(plt.imread(PREVIEW/f"slide_{s['number']:02d}.png"));ax.axis("off")
        ax.set_title(f"{s['number']:02d}  {s['title']}",fontproperties=PROP,fontsize=12)
    axes.flat[-1].axis("off")
    fig.tight_layout(pad=1.0)
    fig.savefig(PREVIEW/"contact_sheet.png",dpi=120)
    plt.close(fig)
    write_guide()
    spec=dict(canvas_inches=[W,H],canvas_cm=[W*2.54,H*2.54],slides=SLIDES)
    (AUDIT/"layout_spec.json").write_text(json.dumps(spec,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (AUDIT/"source_hashes.json").write_text(json.dumps(sources,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    record=dict(checked_at_jst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
                slides=len(SLIDES),canvas_inches=[W,H],source_figures=len(sources),
                original_pptx_sha256=original_hash,original_pptx_unchanged=hash_file(PPTX)==original_hash,
                main_result_csv_unchanged=hash_file(original_csv)==csv_hash,
                source_image_hashes_unchanged=all(hash_file(ROOT/p)==v for p,v in sources.items()),
                geometry_failures=geometry_failures,text_overflows=overflows,
                new_training=0,new_inference=0,original_pptx_edited=False,
                preview_is_original_office_render=False)
    (AUDIT/"verification.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(record,ensure_ascii=False,indent=2))
    assert not geometry_failures, geometry_failures
    assert not overflows, overflows


if __name__=="__main__":
    main()

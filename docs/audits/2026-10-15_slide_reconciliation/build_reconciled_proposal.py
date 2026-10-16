"""Reconcile another AI's proposal with the user's recorded selections.

Only existing evidence and layout outputs are read. Original decks, source
figures, results, and the preceding layout guide remain intact.
"""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
import os
import re

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
OUT = ROOT / "docs/progress/assets/2026-10-16/reconciled_preview"
GUIDE = ROOT / "docs/progress/2026-10-16_slide_reconciled_proposal.md"
module_path = ROOT / "docs/audits/2026-10-15_slide_layout_guide/build_guide.py"
spec = importlib.util.spec_from_file_location("preceding_slide_layout", module_path)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
OLD = deepcopy(b.SLIDES)
b.SLIDES = []
S = b.slide
T = b.text
B = b.box
I = b.image
TABLE = b.table
FONT = FontProperties(fname=b.FONT)
CANONICAL = ROOT / "experiments/2026-10-15_clean_only_result_analysis"
FEATURES = ROOT / "experiments/2026-10-14_error_feature_linkage/existing_feature_dependence.csv"
ATTACHMENT = Path("C:/Users/Casper4/.codex/attachments/030ae9c8-efd2-4f9b-bf1d-41f7117c562b/貼り付けたテキスト.txt")

SELECTIONS = [
    (1,"入れる","3：研究目的。2でも今回の進展を先に示す"),
    (2,"入れる","2・5〜7：何を確認したかと比較対象を明示。最終的な有効条件確定はしない"),
    (3,"入れない","新2026年データを使わない理由の説明は本編へ追加しない"),
    (4,"入れない","一般的な内部修正・新旧コード照合の作業報告はしない。採用指定された5の診断は区別"),
    (5,"入れる","4：2025年の逆転、旧生成の診断と修正を1枚に圧縮して保持"),
    (6,"後続の流れに賛成","4：修正後matchedにも回復が残った事実を1文。旧曲線は予備資料"),
    (7,"入れない","雑音逆転の局所biasを独立説明しない。8の統合の誤り補完（11）とは区別"),
    (8,"入れる","4・5：無雑音学習済みモデルを固定して雑音を加える理由"),
    (9,"入れる","6・7：元3から4・5への変化"),
    (10,"9と同テーマとして入れる","6・7：4や単体との順位は条件依存。全方式での最終優位を要求しない"),
    (11,"9・10と同テーマとして入れる","8：誤差方向と領域別の改善を、統合の働きとして説明"),
    (12,"新方式は後段候補","5で全域MSE目的を1文。内部配分制約・別目的方式の実装は追加しない"),
    (13,"詳細改善を保留","近傍の交換は8の既存結果。個別FN救済・最初のONB段階の件数を主題化しない"),
    (14,"入れない","q100の定義訂正・内部照合は本編なし。結果としてのq100は7に最小限残す"),
    (15,"入れない","秒単位の発生時刻や新しい検知定義の独立説明を加えない"),
    (16,"独立論点は入れない","帯域選択・34特徴の網羅説明はしない。9は17への回答として既存の共通加工対照1点だけ"),
    (17,"約1ページ入れる","9：教授の『何を意味するか／修論へどう使うか』を明示して回答"),
    (18,"6と同様に統合","5の評価条件に既知録音内の未使用chunkと明記。独立した問題ページなし"),
    (19,"入れない","seed・epoch・パラメータ・再現確認の独立報告なし"),
    (20,"入れる","6〜11：結果からの判断、採用理由、説明の到達点を示す"),
    (21,"必ず入れる","10：修論目次と成果の配置を専用1ページで保持"),
    (22,"入れない","追加候補を列挙しない。後から本人が希望した自動切替だけ11に条件付きで残す"),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def link(path):
    return Path(os.path.relpath(path, GUIDE.parent)).as_posix()


def clone(index, number):
    s=deepcopy(OLD[index]);s["number"]=number;b.SLIDES.append(s)
    return s


def save_plot(fig, name):
    fig.tight_layout(pad=.6)
    fig.savefig(OUT/(name+".png"),dpi=160,bbox_inches="tight",facecolor="white")
    fig.savefig(OUT/(name+".svg"),bbox_inches="tight",facecolor="white")
    plt.close(fig)


def figures():
    regional=pd.read_csv(CANONICAL/"regional_loss_changes.csv",dtype={"noise":str})
    fig,ax=plt.subplots(figsize=(8.6,5.1))
    bars=[]
    for offset,noise,color,label in [(-.18,"clean","#9bcddb","無雑音"),(.18,"-20","#1685c4","−20 dB")]:
        rows=regional[regional.noise==noise].set_index("region")
        # The high-q region excludes the near-ONB subset from ONB-and-above.
        above=rows.loc["onb_and_above"];near=rows.loc["near_onb"]
        n_high=above.n-near.n
        high_delta=(above.n*above.mse_change_kW2_m4-near.n*near.mse_change_kW2_m4)/n_high
        vals=[-rows.loc["pre_onb","mse_change_kW2_m4"],-near.mse_change_kW2_m4,-high_delta]
        ax.bar([v+offset for v in range(3)],vals,width=.35,color=color,label=label)
        bars.append(dict(noise=noise,region_n=[225,30,int(n_high)],mse_reduction=vals))
    ax.axhline(0,color="#777",lw=1)
    ax.set_xticks(range(3),["ONB前","ONB近傍","高熱流束側"],fontsize=20)
    ax.set_ylabel("元3→5のMSE減少\n[(kW/m²)²]",fontsize=20)
    ax.tick_params(axis="y",labelsize=18);ax.legend(fontsize=18)
    ax.spines[["top","right"]].set_visible(False)
    save_plot(fig,"regional_mse_reduction")
    groups=pd.read_csv(CANONICAL/"error_direction_contributions.csv",dtype={"noise":str})
    fig,ax=plt.subplots(figsize=(8.2,5.1))
    shares=[]
    for y,noise,label in [(1,"clean","無雑音"),(0,"-20","−20 dB")]:
        rows=groups[groups.noise==noise].set_index("group")
        same=float(rows.loc["same_direction_or_zero","share_of_total_mse_improvement_percent"])
        opposite=float(rows.loc["opposite_signs","share_of_total_mse_improvement_percent"])
        assert abs(same+opposite-100)<1e-8
        ax.barh(y,opposite,height=.48,color="#9bcddb",label="逆符号を含む" if y==1 else None)
        ax.barh(y,same,left=opposite,height=.48,color="#1685c4",label="同方向" if y==1 else None)
        for center,value,color in [(opposite/2,opposite,"#263b49"),(opposite+same/2,same,"white")]:
            ax.text(center,y,f"{value:.2f}%",ha="center",va="center",fontsize=16 if value<20 else 20,color=color)
        shares.append(dict(noise=noise,same_direction_percent=same,opposite_signs_percent=opposite))
    ax.set_yticks([1,0],["無雑音","−20 dB"],fontsize=20)
    ax.set_xlim(0,100);ax.set_ylim(-.6,1.8)
    ax.set_xlabel("全体MSE減少への寄与 [%]",fontsize=20)
    ax.tick_params(axis="x",labelsize=18)
    ax.legend(loc="upper center",ncol=2,fontsize=17,frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    save_plot(fig,"error_group_contribution")
    feature=pd.read_csv(FEATURES,dtype={"noise":str})
    feature=feature[(feature.noise=="clean")&(feature.intervention=="time_permutation")].set_index("model_key")
    names=["conformer","alexnet","hgb","extra_trees"]
    fig,ax=plt.subplots(figsize=(10.2,5.2))
    ax.bar([i-.18 for i in range(4)],feature.loc[names,"rmse_all_kW_m2__original"],width=.35,color="#9bcddb",label="元入力")
    ax.bar([i+.18 for i in range(4)],feature.loc[names,"rmse_all_kW_m2"],width=.35,color="#1685c4",label="時間並べ替え")
    ax.set_xticks(range(4),["CNN＋Tf","AlexNet","HGB","ET"],fontsize=20)
    ax.set_ylabel("無雑音RMSE [kW/m²]",fontsize=20)
    ax.tick_params(axis="y",labelsize=18);ax.legend(fontsize=18)
    ax.spines[["top","right"]].set_visible(False)
    save_plot(fig,"time_permutation_models")
    result=dict(regional_bars=bars,error_group_shares=shares,
                integration_time_permutation={key:{"before":float(feature.loc[key,"rmse_all_kW_m2__original"]),
                     "after":float(feature.loc[key,"rmse_all_kW_m2"])} for key in
                     ["ensemble__original3_mse","ensemble__original3_hgb_extra_trees"]},
                high_q_definition="ONB-and-above 315 minus near-ONB 30 = disjoint high-q 285",
                new_model_fits=0,new_inference=0)
    (AUDIT/"figure_values.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def assemble():
    clone(0,1)
    s=S(2,"前回からの進展","元3からの性能改善を確認し，誤り方と特徴利用の分析へ進んだ",
        "冒頭で今回の進展を提示する。内部修正や検算の完了報告を主役にはしない。",
        "固定モデルへの雑音影響、元3からの性能変化、その変化の説明という順で報告する。性能改善と物理原因の確定は区別する。",
        "この進展を、研究目的へ対応づける。",
        "『改善理由を全て解明した』とはしない。新しい研究課題を追加するページでもない。")
    B(s,"評価の問い","① 何を調べたか\n\n無雑音で学習した\n同じモデルへの\n雑音の影響",1.7,3.35,5.35,4.6,25)
    B(s,"性能の進展","② 何が改善したか\n\n元3 MSE比で全7条件の\n全域RMSEと\n見逃しを改善",7.8,3.35,5.35,4.6,24)
    B(s,"説明の進展","③ 説明はどこまでか\n\n領域別の改善・悪化と\n同方向誤差の縮小\n入力依存の差を確認",13.9,3.35,5.35,4.6,24)
    T(s,"到達点","改善は全指標で一律ではなく，成立条件と限界を整理する",1.8,9.25,17.1,.75,27,b.BLUE)
    s=clone(1,3)
    s["title"]="研究の概要と今回の問い"
    s["objects"]=[o for o in s["objects"] if o["name"]!="評価の三観点"]
    T(s,"入力表現の補足","元3：スペクトログラムを利用／追加2モデル：34要約特徴を利用",1.8,8.95,17.2,.6,24)
    T(s,"今回の問い","問い：何が改善し，どの誤り方・特徴利用と対応するか",1.8,9.7,17.2,.65,25)
    s=S(4,"1. 雑音評価方法の見直し","旧生成の情報経路を修正し，同じ学習済みモデルへの雑音影響を調べる",
        "本人が入れたい5・6・8を1枚へまとめ、旧matched曲線は予備へ回す。",
        "左の旧診断で雑音だけでも予測可能な経路を確認し生成を修正した。ただし修正後matchedにも回復が残った。matchedは雑音ごとに学習するため、今回は無雑音学習を固定し同じモデルへの応答を見る。",
        "次に今回の評価対象と統合構成を示す。",
        "雑音生成を直したからclean_onlyにした、という一段の因果にしない。別条件の旧診断と最新性能の一要因比較もしない。",
        "左：6/18・22 kHz・PCAの旧診断。matchedは雑音ごとに再学習。今回とは別条件")
    T(s,"診断見出し","旧処理の診断（別条件）",1.8,2.8,9.9,.65,25)
    I(s,"旧雑音生成の診断",b.OLD/"slide_old_noise_shortcut.png",1.6,3.65,9.9)
    T(s,"旧処理の経路","元信号の強度が付加雑音にも反映",1.8,9.0,10.0,.65,23)
    T(s,"今回の方針","今回：無雑音学習を固定",11.85,2.8,7.2,.65,25,b.BLUE)
    B(s,"今回の学習","無雑音のみで学習",11.8,3.75,7.25,1.0,25)
    T(s,"矢印1","↓",15.1,4.8,.9,.5,26)
    B(s,"固定する対象","モデル・前処理・統合重みを固定",11.8,5.3,7.25,1.0,24)
    T(s,"矢印2","↓",15.1,6.35,.9,.5,26)
    B(s,"今回の評価","同じ評価区間へ雑音を付加",11.8,6.85,7.25,1.0,24)
    T(s,"残った回復","生成修正後のmatchedにも回復が残ったため，固定モデルで評価",1.8,9.75,17.1,.65,24)
    s=clone(4,5)
    s["claim"]="既知録音内の未使用区間で，元3・2種類の4・5モデルを比較する"
    s["objects"][0]["text"]="重みは無雑音の学習側OOF予測の全域MSEで決め，評価中は固定"
    s=clone(5,6)
    s["claim"]="5モデルは元3 MSEより，全7条件で全域RMSEが低下した"
    s["objects"]=[o for o in s["objects"] if o["name"]!="条件差"]
    T(s,"無雑音からの悪化量","clean→−20 dBの増分\n元3 MSE：＋47.07\n5モデル：＋47.43",13.25,7.0,6.0,1.9,23)
    s["caution"]+=" 4・単体に対する順位は条件依存と口頭で添え、詳細表は予備へ置く。"
    s=clone(6,7)
    s["claim"]="見逃しは減少したが，q100の改善は実験日によって異なる"
    s["objects"]=[o for o in s["objects"] if o["kind"]=="image"]
    T(s,"q100表見出し","無雑音のq100［kW/m²］",12.9,3.0,6.2,.75,24)
    TABLE(s,"元3と5の日別q100",[["実験日","元3 MSE","5モデル"],["6/11","368.98","368.98"],["6/18","434.02","376.32"]],
          12.9,4.0,[1.7,2.25,2.2],.78,23)
    T(s,"4との交換","6/18ではET追加4が322.11",12.9,6.85,6.15,.65,23)
    T(s,"誤報","FP：無雑音1→1\n　　 −20 dB0→0（元3→5）",12.9,7.8,6.1,1.3,23)
    T(s,"指標の違い","見逃し件数とq100は，それぞれの改善を確認する",1.8,9.6,17.1,.65,26,b.BLUE)
    s["footer"]="日別ONB閾値。ONB以上315／ONB前225chunk。q100表は無雑音の元3 MSEと5モデルの比較"
    s["oral"]="左でFN42→27、65→52を読む。右でq100の6/11不変、6/18改善を読む。FPも併記する。境界個別事例と最初のONB段階の詳細は予備/口頭に保持し、本編の新テーマにしない。"
    s=S(8,"3. 誤り補完から見た統合の働き","改善には，逆向き相殺に加えて同方向誤差の縮小が含まれる",
        "11の誤り補完に絞る。雑音逆転の局所bias（除外7）や境界救済（保留13）の独立ページにはしない。",
        "左は正がMSE改善、負が悪化。無雑音は近傍と高q、−20はONB前と高qが改善。右は全体MSE減少の寄与を標本群で分けた結果で、−20では同方向群が87.15%。同じ方向に外れていても誤差の小さい予測へ寄せる効果がある。",
        "この誤りの違いを特徴利用と接続し、教授の説明性への指摘へ答える。",
        "3ブロックは元3・HGB・ETであり、5単体全ての誤差符号ではない。同方向群にはゼロ誤差の境界を含む。高q領域はONB以上315から近傍30を除いた285。",
        "元3・HGB・ETの3ブロック。割合は標本群別の数値寄与であり，物理原因の割合ではない")
    T(s,"領域見出し","改善・悪化した領域",2.3,2.8,8.0,.65,25)
    T(s,"方向群見出し","MSE減少を生んだ誤差群",11.3,2.8,7.8,.65,25)
    I(s,"領域別のMSE減少",OUT/"regional_mse_reduction.png",1.6,3.65,8.6)
    I(s,"誤差方向群の寄与",OUT/"error_group_contribution.png",11.15,3.65,8.0)
    T(s,"同方向群の寄与","−20 dBのMSE減少の87.15%は，同方向の誤差群から生じた",1.8,9.75,17.1,.65,25,b.BLUE)
    s=S(9,"3. 教授指摘への対応：説明性と研究の接続","説明性を，特徴利用の違いと予測・誤差の変化へつないだ",
        "17を約1ページで保持。入力依存一般（16）や帯域選択の独立ページへ変更しない。",
        "前回指摘を最初に再掲する。旧マスク/IGは依存の可視化、今回は共通入力加工で予測変化を確認した。左の時間並べ替え対照でCNN系は変わりHGB/ETは不変、右の統合結果で元3と5の応答差を示す。これを性能と特徴利用の対応へ使う。旧IGを起点にモデル選定を行ったという未確認の経緯は作らない。",
        "性能とその説明を修論の章へ配置する。",
        "水流雑音の利得の因果的説明は未確定。旧IG全体を未収束とみなさない。9/18再掲の0 dBの1試料は両モデル収束済み。",
        "10/7同条件・別fitの無雑音共通加工。時間順序の対照は水流雑音の因果効果や音源同定を示さない")
    T(s,"教授の指摘","前回の指摘：何を意味するか／修士研究にどう活用するか",1.8,2.7,17.1,.7,25)
    I(s,"既存の時間並べ替え対照",OUT/"time_permutation_models.png",1.6,3.8,10.7)
    T(s,"統合表見出し","統合のRMSE［kW/m²］",12.9,3.75,6.2,.65,23)
    TABLE(s,"統合の加工前後",[["統合","元入力","加工後"],["元3 MSE","34.07","47.52"],["5モデル","27.80","31.10"]],
          12.9,4.6,[2.2,1.95,2.0],.7,23)
    T(s,"活用の流れ","前回：依存を可視化\n今回：共通加工で予測変化を確認\n修論：特徴と誤り補完を対応づける",12.9,7.3,6.2,1.8,22)
    T(s,"説明の限界","時間並べ替えは人工的対照であり，水流雑音での改善原因を直接証明しない",1.8,9.7,17.1,.65,23,b.AMBER)
    clone(9,10)
    s=clone(10,11)
    s["title"]="4. まとめと後期の研究計画"
    s["claim"]="元3からの改善と限界を軸に，説明・考察と修論への整理を進める"
    s["role"]="本人が今回検討を求めた後期計画を大きく保持し、結論から今後へつなぐ。"
    s["caution"]+=" まとめ中心の左右2分割で計画図を小さくしない。候補の列挙はしない。"


def write_guide():
    preview=link(OUT)
    lines=["# 10/16発表：別AI案と本人の1〜22採否を踏まえた再検討案","",
        "記録日：2026-10-15。これは添付提案と本人の過去の採否を照合した更新提案。本人の最終採用や教授の了承を記録したものではない。","",
        f"[今回の11枚の配置見本PDF]({preview}/reconciled_slide_preview.pdf)／[一覧]({preview}/contact_sheet.png)／[旧案と配置ガイド](2026-10-16_slide_build_guide.md)。旧案・原PPTX・原図を保持した。","",
        "## 結論と採否の根拠","",
        "別AI案の『進展を先に示す』『旧雑音問題を1枚へ圧縮』『旧可視化より定量的対照を主役にする』を採用する。一方、q100全面予備化、境界問題の主題化、教授指摘の再掲を消す変更、後期計画の縮小はそのまま採用しない。内容の役割を本人の採否へ合わせる。","",
        "| 添付の提案 | 判断 | 今回の反映 |","|---|---|---|",
        "| 冒頭に前回からの進展 | 採用 | 2。設定変更・検算ではなく、問い・性能・説明の進展を示す |",
        "| 旧雑音問題を2枚→1枚 | 採用 | 4。旧診断＋修正後にもmatched回復が残った1文＋固定評価の理由 |",
        "| 既知録音内・元3 MSEを明記 | 採用 | 5〜7。比較基盤を新しい研究課題にしない |",
        "| 絶対誤差と悪化量を区別 | 採用 | 6。増分の確認は完了済みとして結果を示す |",
        "| q100詳細を全て予備へ | 一部変更 | 7に元3/5×2日の最小表を残す。4構成全表は予備 |",
        "| ONB近傍・最初の段階を強く前面へ | 範囲を限定 | 近傍交換は8の既存領域結果、最初の段階件数と個別FNは予備/口頭。13・15を主題へ戻さない |",
        "| 旧IG3枚→共通加工中心 | 採用条件を明示 | 9は教授指摘17への回答の1ページ。独立した入力依存16の紹介にしない |",
        "| 説明用の架空残差で相殺を図示 | 今回は使わない | 8は既存の領域表と誤差方向寄与の実数を使用 |",
        "| まとめ中心へ変更 | 一部変更 | 11の結論文を更新し、本人が求めた後期計画は大きく保持 |",
        "| 自動重み切替を条件付き候補 | 採用・既存方針維持 | 本人の後続希望を優先。新方式実装や最終確認を今回の必須課題にしない |","",
        "## 添付でそのまま使わない二つの説明","",
        "**旧IGの品質**：『20例中7例のみ収束』は今回確認した保存記録では対象集合を特定できなかった。9/18本runはConformer91/175、AlexNet89/175、22 kHz無雑音は1/5・0/5。一方、実際に再掲予定だった22 kHz・0 dBの同一1例は両モデルとも収束済み。したがって旧画像全てが未収束とは扱わない。主役を定量対照へ移す理由は、現在の進捗と研究への接続を伝えやすくするため。収束と物理的正しさも別である。[保存記録](../../experiments/2026-09-18_onb_full_sweep_151715/README.md)。","",
        "**入力依存と雑音利得の因果**：同方向誤差の寄与は保存予測から分解できる。時間並べ替えへの応答差も確認済み。ただし後者が水流雑音の利得を生んだ原因とまでは確定していない。『誤差の調整を説明し、入力依存の違いも確認した』と分けて述べる。旧IGからモデル採用を決めたという未確認の経緯も作らない。","",
        "## 本人が指定した1〜22と今回の対応","",
        "番号は元の論点番号で、ページ番号ではない。本人の最新指示を添付AIや過去AIの提案より優先する。『入れない』項目を独立ページ・必須作業へ戻さず、結果の解釈に必要な短い条件表示と区別する。","",
        "| 元番号 | 本人の指定 | 今回の扱い |","|---|---|---|"]
    lines += [f"| {n} | {choice} | {use} |" for n,choice,use in SELECTIONS]
    lines += ["","[本人選択の根拠](../research_plan/2026-10-14_friday_selection_and_question_answers.md)／[後続の本人方針](../research_plan/2026-10-14_research_synthesis_and_presentation_policy.md)。","",
        "## 全体の順序と共通の作成設定","",
        "表紙→進展→目的→旧問題と評価方針→現在の条件→回帰→ONB→誤り補完→教授指摘への回答→修論目次→後期計画。最初に今回の進展を示し、元の11枚を保ちながら旧問題の比重を下げる。","",
        "元PPTXは50.8×28.575 cm。タイトルは左5.03／上0.84 cm・30 pt、結論は左4.19／上3.96 cm・32 pt、条件注記は左4.19／上26.85 cm・18 pt。下記の座標は左上原点・cm。図の縦横比を固定し、左の大学装飾とページ番号は原本の書式を保持する。配置見本は大学装飾を簡略化した作成例で、Officeの完全な再描画や完成PPTXではない。","",
        "### 対象と完成状態","",
        "主結果143609と再実行194258は全7条件で数値同等という完了済み照合を引き継ぐ。今回は既存CSVから3図を描画し配置を変更しただけで、新学習・モデル推論・新しい入力加工は0。新2026年評価、停止matchedの再調査、修論完成後の最終確認は対象外。","",
        "## ページ別の貼り付け文章・図・配置",""]
    for s in b.SLIDES:
        n=s["number"]
        lines += [f"### {n}枚目　{s['title']}","",f"[配置見本]({preview}/slide_{n:02d}.png)","",
            f"**役割**：{s['role']}","", "**タイトル**", "", "```text",s["title"],"```",""]
        if s["claim"]:
            lines += ["**結論欄**","","```text",s["claim"],"```",""]
        for item in s["objects"]:
            x,y,w,h=b.rect_cm(item["rect"])
            lines += [f"**{item['name']}**：左{x:.2f}／上{y:.2f}／幅{w:.2f}／高さ{h:.2f} cm"+
                (f"、{item['font_pt']} pt。" if "font_pt" in item else "、縦横比固定。"),""]
            if item["kind"]=="image":
                lines += [f"[挿入する画像]({link(ROOT/item['source'])})。`{Path(item['source']).name}`。",""]
            elif item["kind"]=="table":
                lines += ["| "+" | ".join(item["cells"][0])+" |","| "+" | ".join(["---"]*len(item["cells"][0]))+" |"]
                lines += ["| "+" | ".join(row)+" |" for row in item["cells"][1:]]
                lines += [""]
            else: lines += ["```text",item["text"],"```",""]
        if s["footer"]:
            lines += ["**下端の注記**","","```text",s["footer"],"```",""]
        lines += [f"**口頭説明**：{s['oral']}","",f"**次へのつなぎ**：{s['transition']}","",f"**注意**：{s['caution']}",""]
    lines += ["## 本編から移す内容と、今回残る作業","",
        "予備へ移す：旧matched曲線、4構成全体の日別q100、最初のONB段階の詳細/個別FN、旧マスク・IGの原寸、全方式の数値表、数値分解の全式。保管することと、本編で説明すること、今後必ず実行することは別。","",
        "後期計画は、評価と3→4→5解析を完了側、説明・考察・章別整理を現在以降へ。10〜11月整理と本文展開、12月結果/考察/結論、翌1〜2月完成と発表準備という既存計画を土台にする。正式締切や本文執筆済みとはしない。自動切替は条件付き候補1点だけで、追加候補を列挙しない。","",
        "今回完了：添付の全11枚・5変更点と1〜22採否の照合、根拠確認、再検討11枚の文章/配置、3図の再描画と配置見本。残る資料作業は本人による採否とPPTXへの反映・実機可読性/持ち時間の確認。追加研究実験は今回の前提にしない。",""]
    GUIDE.write_text("\n".join(lines),encoding="utf-8")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protected=[b.PPTX,ATTACHMENT,CANONICAL/"pooled_metrics.csv",CANONICAL/"regional_loss_changes.csv",
        CANONICAL/"error_direction_contributions.csv",FEATURES,
        ROOT/"docs/progress/2026-10-16_slide_build_guide.md"]
    before={str(p):digest(p) for p in protected}
    figures();assemble()
    source_images={};overflows=[];geometry=[]
    with PdfPages(OUT/"reconciled_slide_preview.pdf") as pdf:
        for s in b.SLIDES:
            for item in s["objects"]:
                x,y,w,h=item["rect"]
                if min(x,y,w,h)<0 or x+w>b.W or y+h>b.H: geometry.append([s["number"],item["name"]])
            fig,overflow=b.draw(s,source_images);overflows+=overflow
            fig.savefig(OUT/f"slide_{s['number']:02d}.png",dpi=100)
            pdf.savefig(fig);plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(18,14),facecolor="#edf1f4")
    for ax,s in zip(axes.flat,b.SLIDES):
        ax.imshow(plt.imread(OUT/f"slide_{s['number']:02d}.png"));ax.axis("off")
        ax.set_title(f"{s['number']:02d} {s['title']}",fontproperties=FONT,fontsize=12)
    axes.flat[-1].axis("off");fig.tight_layout(pad=1.0)
    fig.savefig(OUT/"contact_sheet.png",dpi=120);plt.close(fig)
    write_guide()
    source_record=dict(protected_sources=before,inserted_images=source_images,
        attachment_role="User supplied alternative proposal, not an instruction authority",
        authoritative_user_selection="docs/research_plan/2026-10-14_friday_selection_and_question_answers.md")
    (AUDIT/"sources.json").write_text(json.dumps(source_record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (AUDIT/"layout_spec.json").write_text(json.dumps(dict(slides=b.SLIDES),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    local_links=0
    for target in re.findall(r'\]\(([^)]+)\)',GUIDE.read_text(encoding="utf-8")):
        if re.match(r"^[a-z]+://",target):continue
        assert (GUIDE.parent/target).exists(),target
        local_links+=1
    verification=dict(checked_at_jst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
        all_22_user_choices_mapped=len(SELECTIONS)==22,slides=len(b.SLIDES),
        source_hashes_unchanged=all(digest(Path(p))==value for p,value in before.items()),
        original_pptx_edited=False,previous_preview_preserved=True,
        local_links_checked=local_links,geometry_failures=geometry,text_overflows=overflows,
        new_plots_from_saved_tables=3,new_training=0,new_inference=0,
        new_input_interventions=0,stopped_matched_read=False,
        not_yet_adopted_by_user=True)
    (AUDIT/"verification.json").write_text(json.dumps(verification,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(verification,ensure_ascii=False,indent=2))
    assert not geometry and not overflows


if __name__=="__main__":main()

# -*- coding: utf-8 -*-
"""Apply reviewed documentation updates; preserve pre-review text in the audit.

Only documentation is changed. Original Office/PDF files and result CSVs are kept.
"""
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
AUDIT=Path(__file__).parent
BEFORE={r['path']:r for r in csv.DictReader((AUDIT/'before/document_catalog.csv').open(encoding='utf-8-sig'))}
CHANGES=[]


def update(name,reason,transform):
 p=ROOT/name
 old=p.read_text(encoding='utf-8-sig')
 if name in BEFORE:
  assert hashlib.sha256(p.read_bytes()).hexdigest()==BEFORE[name]['sha256'],f'Changed since review: {name}'
 new=transform(old)
 if old==new:return
 p.write_text(new.rstrip()+'\n',encoding='utf-8')
 CHANGES.append(dict(path=name,reason=reason,before_sha256=hashlib.sha256(old.encode()).hexdigest(),after_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))


def replace(name,reason,pairs):
 def apply(s):
  for old,new in pairs:
   assert old in s,(name,old)
   s=s.replace(old,new)
  return s
 update(name,reason,apply)


def historical(name,note):
 def apply(s):
  rel=lambda target:os.path.relpath(ROOT/target,(ROOT/name).parent).replace('\\','/')
  addition=f'> **9/15追加比較後の更新先**: {note} 最新の[比較解析]({rel("experiments/2026-09-15_onb_frequency_comparison/analysis.md")})、[本人の研究仮説]({rel("docs/research_plan/2026-09-15_master_thesis_hypothesis.md")})、[解析手順]({rel("docs/analysis_workflow.md")})を参照する。以下の本文は当時の記録として保持する。\n\n'
  pos=s.find('\n')+1 if s.startswith('#') else 0
  return s[:pos]+'\n'+addition+s[pos:].lstrip('\n')
 update(name,'Historical scope and specific successor: '+note,apply)


def main():
 update('docs/progress/2026-09-18_weekly_progress_draft.md','Integrate completed two-band analysis and user hypothesis into the active SOAP draft',lambda _:
'''# 植木研・研究進捗報告（発表前の下書き）

日付：2026年9月18日（9月15日の完了済み2帯域比較に基づく）  
名前：柴崎陸

## 【S】Subjective

学部研究で観察した「ノイズ増加に伴って性能が基本的に低下し、アンサンブルによってその低下を抑えられる」という効果を、修士研究では成立する条件と理由まで含めて説明したい。今回、3 kHzと22 kHzでノイズに対する性能変化が異なったため、周波数帯域・モデルの特徴利用・説明性の信頼性を整理した。学部の観察を基礎とし、現行の評価条件で何が再現し、何が異なるかを確認する。

## 【O】Objective

2025年6月11日の18録音を用いた、3 kHzと22 kHzの各7ノイズ条件を比較した。元WAV分離3-fold、各ノイズで学習・評価するmatched、深層300 epochs・batch 16、seed 42である。比較した2帯域では同じ元WAV/chunk集合・外側分割を確認した。進行中の別runは今回の対象に含めていない。

主評価は元WAV内の予測中央値を全foldでpoolしたR²とした。

| 項目 | 3 kHz | 22 kHz |
|---|---|---|
| CNN＋Transformerの無雑音→−20 R² | 0.9558→0.9194 | 0.9623→0.9300 |
| −20の単純平均 / inner holdout R² | 0.9227 / 0.9234 | 0.9306 / 0.9283 |
| 最良単体を上回った条件数：単純平均 / inner | 4/7 / 5/7 | 4/7 / 2/7 |
| 最初のONB点を見逃したモデル×条件 | 31/35 | 32/35 |

RFは両帯域でほぼ横ばいだった。WAV連続ROC-AUCは全70組で1.0、誤警報は0だった。一方、3 kHzの両アンサンブルは全7条件で最初のONB点を見逃した。閾値±10%の録音は1本である。

卒論PDFの比較図と数表を読み、二値化後AUC・chunkのfold平均に揃えた比較図も作成した。3 kHzのCNNはこの集計でもR²が低下する一方、二値AUCは無雑音0.9537→−20で0.9699へ上がった。

各録音の0/30/59秒から54区間を取り、元信号と付加水流音を別々に調べた。reference −20での帯域内SNR中央値は2–3 kHzで+4.05 dB、10–15 kHzで−35.18 dBだった。対応する216保存配列の再構成を照合した。

共通マスクの最大R²低下帯域は、RFでは3 kHz入力の2–3 kHz、22 kHz入力の2–5 kHzが各21/21組で一致した。深層モデルは別の帯域依存を示した。TreeSHAPの再構成絶対誤差中央値は3 kHzで0.249、22 kHzで0.262 W/m²だった。IGは両帯域420/420例でcompleteness相対誤差が説明用の目安0.05を超え、配列総和とCSVの対応は確認した。

## 【A】Assessment

3 kHzでは、強ノイズ時の回帰における統合効果が研究仮説を部分的に支持した。ただし統合は無雑音ではCNNより低精度であり、劣化量が小さいことと強ノイズで高精度なことを併記する必要がある。22 kHzにも劣化と補完はあり、今回の結果を一括して失敗とは位置づけない。1実験日・1 seedでの記述比較で、未知ノイズや別日での再現性は今後の課題である。

帯域内SNRが高い2–3 kHzにRFが依存することは、精度維持の説明候補と整合する。一方、両上限とも224列へ変換するため、周波数範囲と画像上の模様の細かさが同時に変わる。22 kHzの−12→−16の回復は最低熱流束WAVの誤差減少に大きく左右されていたが、そのモデル内部の原因は未確定である。

IGはbaselineとの差を説明する手法だが、生powerのゼロbaseline・64分割積分・先頭のlog変換の組合せが数値不一致の原因候補となっている。安定したmapが得られることと正しく予測差を分解できることを分けて扱う。共通マスクもゼロ置換に伴う入力分布の変化を含むため、物理的重要度そのものとは断定しない。

回帰誤差の減少は、固定閾値でのONB検知改善と一致しない。現在の測定点順の評価に加え、近傍の独立反復と同期した正解観測が必要であり、秒単位の早期検知は未評価である。

## 【P】Plan

1. 金曜は研究仮説→学部との比較→2帯域の性能→帯域SNRとマスク→説明性の信頼性→次の検証の順で報告する。
2. 発表後、まず既存データの代表条件について、回復と統合効果を複数seedで確かめる条件案を作る。clean_onlyで同じモデル・重みを固定する転送評価との違いも明確にする。
3. 少数例のIG収束確認と、必要な共通周波数grid・マスク置換の比較を設計する。学習済みモデル本体が未保存のため、必要な限定再学習・保存範囲を先に決める。
4. 並行してONB前後の測定点、同条件反復、同期する観測源を相談し、新実験・同時計測の準備を進める。

## 相談したいこと

- 学部の劣化抑制を発展させ、修士では「成立条件と補完の理由」を示す構成について。
- 次の限定比較として、代表条件の複数seed・clean転送・共通周波数gridの着手順について。
- 共通マスクと数値診断を今回の説明性の主材料にし、IGの収束を少数例から確認する方針について。
- ONB近傍の独立反復、熱流束刻み、同期正解の観測源について。

根拠と図表: [2帯域比較解析](../../experiments/2026-09-15_onb_frequency_comparison/analysis.md)、[発表準備メモ](../research_plan/2026-09-18_xai_progress_brief.md)。本稿は提出・発表済みの記録ではない。今回の文書更新では追加学習や新実験を実施していない。
''')

 def brief(s):
  s=s.split('---\n\n## 以下は先行する22 kHz単独版の原稿')[0]
  s=s.replace('既存の[SOAP下書き](../progress/2026-09-18_weekly_progress_draft.md)は22 kHz単独時点の案で、比較の本文統合は未実施。',
              '[SOAP下書き](../progress/2026-09-18_weekly_progress_draft.md)も完了済み2帯域の比較・本人の仮説へ更新した。')
  s+='''
## 話すときの確認点

- 希望する結論は仮説として示し、今回支持された部分と未確認の部分を続けて述べる。
- 劣化量・強ノイズ時の絶対性能・ONB誤りを分ける。3 kHzの統合は回帰で改善したが最初のONB点は捉えていない。
- マスク結果と帯域内SNRは原因の手がかり。22 kHz回復の原因や気泡由来の同定が確定したとは話さない。
- 手法を選んだ理由は[説明性の役割表](../notes/model_notes.md)、追加検証の選び方は[解析手順](../analysis_workflow.md)へ戻る。
- 今回の対象外runや他実験日を、準備資料へ自動的に追加しない。

## 残る仕上げ

発表時間に合わせて図を選び、PPTXへ配置して話す時間を確認する。現時点で作成済みなのは解析・図・構成案・SOAP下書きで、完成PPTXや発表実施記録ではない。

先行する22 kHz単独の草稿は[文書監査の改訂前記録](../audits/2026-09-15_document_consistency/before/extracted_text.jsonl)に保持し、この準備メモでは現行案を一つにした。[22 kHz単独解析](../../experiments/2026-09-15_onb_22khz_noise_sweep/analysis.md)の当時の数値は保持している。
'''
  return s
 update('docs/research_plan/2026-09-18_xai_progress_brief.md','Keep one active presentation draft; retain prior text in audit',brief)

 replace('docs/research_status.md','Align research focus, completed materials and parallel post-presentation work',[
 ('回帰・評価・説明性出力の基盤は整った。現在の中心課題は、説明結果を検証して研究上の解釈へつなげることにある。',
  '回帰・評価・説明性出力の基盤は整った。現在は、**アンサンブルによるノイズ下の劣化抑制が成立する条件と、周波数・モデルによる差の理由を、性能と説明性の両方から確かめる段階**にある。ONB近傍の検知性能と物理的な根拠へつなげる。'),
 ('**9月18日（金）の進捗発表**で、説明性指標の出力、その解釈、結果のまとめを話す。',
  '**9月18日（金）の進捗発表**で、本人の仮説、卒論との対応、3/22 kHzのノイズ傾向・統合効果、説明性の役割・解釈・原因の確かさを話す。'),
 ('今回の比較は本人の回答により完了済み06.11の2帯域に限定し、進行中runは対象外。',
  '今回の比較は本人の回答により完了済み06.11の2帯域に限定し、進行中runは対象外。この文書整理でも新しい完了状況を監視・採用していない。'),
 ('| 現在のコード設定 |','| 最後に確認したコード設定（9/15・比較解析時） |'),
 ('IG210例の配列総和照合。金曜の主材料 |','IG210例の配列総和照合。2帯域比較の片側 |'),
 ('最新7条件と数値を混ぜない','9/15の各7条件と数値を混ぜない'),
 ('**完了**: 9/15・22 kHzの7条件を採用し、数値・図表を固定','**完了**: 9/15・3/22 kHz各7条件の数値・図表を固定'),
 ('[既存SOAP下書き](progress/2026-09-18_weekly_progress_draft.md)は22 kHz単独解析時点の案で、2帯域比較の本文への統合はまだ行っていない。',
  '[SOAP下書き](progress/2026-09-18_weekly_progress_draft.md)にも本人の仮説・2帯域の性能・原因候補・説明性を統合済み。'),
 ('## 発表後の順序（提案）\n\n1. **ONB近傍の評価・追加実験設計**: 近傍幅、最初のONB点、誤警報、未検出、測定点差を整理。同時計測の正解源・同期方法を決める。\n2. **新しい実験の準備と予約**: ONB前後の細かい刻み、同条件の独立反復、新しい実験日を優先候補にする。設備や可能日程は本人・先生と決める。\n3. **現行実装で一般化を検証**: 小さな代表条件でclean_only、次いでleave_one_day_out。全ノイズを増やす前に分割・同一モデル再利用・完了を確認する。\n4. **XAIの信頼性と帯域の役割を詰める**: IG整合性を確認し、共通周波数gridの帯域保持/除去・再学習を必要な範囲で行う。\n5. **単体と統合の最終比較**: 代表条件の複数seed、残差相関、ONB誤り訂正を評価。新しいstackingや残差補正は、補完余地が確認できた場合の候補とする。\n6. **結果を固定して修論へ**: 改善する場合・しない場合の両方を、条件・根拠・限界とともにまとめる。',
 '''## 発表後の進め方（作業案）

**既存データの原因検証と、新実験の準備を並行して進める。** アンサンブルの検証は性能・帯域・一般化の各比較に含め、最後まで後回しにしない。

| 順序 | 既存データで進めること | 終わりの基準 |
|---|---|---|
| 1 | 3/22 kHz・代表ノイズの複数seed比較を設計。モデル・分割・比較単体を固定する | 回復と統合効果の再現性を判定できる小規模な条件書がある |
| 2 | clean_onlyで同じモデル・PCA・scaler・重みをノイズ間で固定する | matchedでの適合と固定モデルの転送を区別して示せる |
| 3 | 必要な共通周波数gridの帯域保持/除去、少数例のIG収束とマスク置換を調べる | 帯域・resize・説明手法の数値誤差のどれが影響したかを絞れる |
| 4 | 別日・独立ノイズで、単体と統合の回帰/ONB性能を確認する | 検証した条件範囲と一般化の限界を示せる |
| 5 | 成果と限界を固定し修論へ反映する | 主張・数値・図・出典の対応が取れている |

並行する新実験では、ONB前後の細かい刻み、同条件反復、観測源・同期方法を先に設計する。設備・実験可能日・正解の判定方法は先生と相談して決める。既存データの全検証終了を準備の前提にしない。

次回の具体的な一手は**代表条件の再現性比較の条件書を作ること**。学習本体は未起動。IGで必要なモデル保存・限定再学習も別作業として条件を残す。検証の目的と判定方法は[解析手順](analysis_workflow.md)、月別の配分は[年間計画](research_plan/2026_annual_plan.md)を参照する。''')])

 replace('docs/research_plan/2026_annual_plan.md','Reflect the integrated hypothesis and parallel validation/new-experiment work',[
 ('| 2026-09-15の本人の指示 | 9/18は説明性の出力・解釈・まとめ。ONB検知、新実験、同時計測は発表後 |',
  '| 2026-09-15の本人の指示 | 学部の劣化抑制を発展させ、帯域差と説明性の原因を整理。9/18は2帯域比較と説明性を発表。ONB専用設計、新実験、同時計測は発表後 |'),
 ('アンサンブルが単体より安定することも検証対象に残す。ただし、改善を必ず得るという結論を先に置かず、精度・ONB性能・ノイズ下の安定性のどこに効果があるかを調べる。',
  'アンサンブルによるノイズ下の劣化抑制を中心的な仮説として検証する。各条件の絶対性能・無雑音からの低下量・ONB誤りを分け、どこに改善があり、その理由をどこまで説明できるかを調べる。'),
 ('**説明性の出力・解釈・結果整理と進捗発表**。9/15に22 kHz全7ノイズを解析し図表・SOAP下書きを作成済み',
  '**本人の仮説・卒論比較・3/22 kHzの性能と説明性を整理して発表**。9/15に各7ノイズの比較、帯域SNR、原因分析、図表・SOAP下書きを作成済み'),
 ('ONB評価定義、新実験・同時計測の設計。設備/実験日程の確認。一般化の代表条件検証',
  '既存データの代表条件の再現性比較・clean転送を設計。並行してONB評価定義、新実験・同時計測、設備/実験日程を相談'),
 ('評価仕様・取得項目・同期方法・必要な反復・最初の検証条件が決まる',
  '劣化抑制と回復を確かめる条件書、取得項目・同期方法・独立反復の設計が決まる'),
 ('ONB近傍の追加測定、独立日/ノイズ評価、XAI信頼性の確認、必要な帯域比較',
  'ONB近傍の追加測定と、代表条件の単体/統合比較、独立日/ノイズ評価、XAI信頼性・共通gridの帯域比較'),
 ('4. アンサンブルの評価目標。全域RMSE、ONB見逃し、ノイズ下安定性のどれを改善対象にするか。',
  '4. アンサンブルの劣化抑制を、絶対性能・低下量・ONB誤りのどの指標で判定するか。各比較に単体と統合を含める。')])

 replace('docs/document_index.md','Update active draft pointers, successor evidence and full catalog',[
 ('| 22 kHz単独解析時点を確認する | [9/15・22 kHz解析](../experiments/2026-09-15_onb_22khz_noise_sweep/analysis.md)、[SOAP下書き](progress/2026-09-18_weekly_progress_draft.md) | 先行解析と報告案。SOAP本文には追加の2帯域比較を未統合 |',
  '| 今週の報告を作る | [2帯域のSOAP下書き](progress/2026-09-18_weekly_progress_draft.md) | 本人の仮説・最新比較・原因候補を統合した未提出の草稿 |\n| 追加結果を解析する | [解析手順](analysis_workflow.md) | 対象固定→性能→原因候補→XAI→次の比較の判定基準 |\n| 原資料の位置づけを確認する | [原資料と現在方針の対応](original_document_guide.md) | 学部からの連続性、旧計画・説明手法・原本の扱い |\n| 22 kHz単独解析時点を確認する | [9/15・22 kHz解析](../experiments/2026-09-15_onb_22khz_noise_sweep/analysis.md) | 先行結果は保持。発表草稿は現行2帯域へ更新 |'),
 ('log-power採用後の[9/8結果](../experiments/2026-09-14_status_audit/README.md)と[9/14結果](../experiments/2026-09-15_research_status_snapshot/README.md)が現在の判断材料',
  '[9/15・2帯域比較](../experiments/2026-09-15_onb_frequency_comparison/analysis.md)が最新の判断材料。モデル順位・利用帯域は当時と異なる'),
 ('当日の試行記録。現行は7条件・2方式。','当日の試行記録。最新解析は2帯域×7条件・2方式。'),
 ('[最新完了結果](../experiments/2026-09-15_research_status_snapshot/README.md)を参照',
  '[9/15・2帯域比較](../experiments/2026-09-15_onb_frequency_comparison/analysis.md)を参照'),
 ('既存91 Markdownの全件索引は',
  '最新の全件索引・抽出範囲・修正理由は[文書整合性監査](audits/2026-09-15_document_consistency/README.md)。過去の棚卸し時点の91 Markdown索引は'),
 ('これらは今回の整理前の棚卸しで、新規文書の自動追随一覧ではない。',
  'これらは同日先行監査の固定記録で、新規文書の自動追随一覧ではない。'),
 ('5. 新しい判断と衝突する旧ページに後継リンクを付ける。本文の過去数値は置き換えない。',
  '5. 新しい判断と衝突する旧ページに、更新された判断と後継を記す。本文の過去数値は置き換えない。\n6. 発表準備中ならSOAPと発表メモも更新する。旧草稿は監査記録へ保持し、現行草稿に二つの異なる次工程を併記しない。')])

 replace('README.md','Expose user hypothesis and reusable analysis entry points',[
 ('| 年間計画・目指す成果 |','| 本人が目指す修論の結論と背景 | [研究仮説と確かめ方](docs/research_plan/2026-09-15_master_thesis_hypothesis.md) |\n| 年間計画・目指す成果 |'),
 ('| 最新の確認済み数値 | [結果スナップショット一覧](experiments/README.md) |',
  '| 最新の確認済み数値 | [結果スナップショット一覧](experiments/README.md) |\n| 追加結果を段階的に分析する | [解析手順](docs/analysis_workflow.md) |\n| 原資料と現在の方針の関係 | [原資料ガイド](docs/original_document_guide.md) |'),
 ('9月15日に、文書・コード・保存結果の更新差を整理しました。今週は説明性の出力と解釈を優先し、',
  '9月15日に、3/22 kHzの比較と本人の研究仮説を文書全体へ反映しました。今週は劣化抑制・帯域差・説明性の出力と解釈を整理し、'),
 ('[今回の確認範囲と変更理由](docs/audits/2026-09-15_workspace_review/README.md)',
  '[文書全体の確認範囲と変更理由](docs/audits/2026-09-15_document_consistency/README.md)')])

 replace('AGENTS.md','Ensure future sessions retain user hypothesis and analysis workflow',[
 ('- 年間計画は `docs/research_plan/2026_annual_plan.md`、文書の更新先と履歴の扱いは `docs/document_index.md`。',
  '- 年間計画は `docs/research_plan/2026_annual_plan.md`、文書の更新先と履歴の扱いは `docs/document_index.md`。\n- 本人の修論の理想と背景は `docs/research_plan/2026-09-15_master_thesis_hypothesis.md`。学部で観察したアンサンブルのノイズ劣化抑制を、成立条件と原因まで含めて発展させる仮説を中心に置く。\n- 追加結果の解析は `docs/analysis_workflow.md` に沿い、確認事実・原因仮説・次の識別比較を分ける。現在の対象外runを自動で追加しない。'),
 ('6. 最後に「先生に相談したいこと」「次週やること」を明確にする。',
  '6. 最後に「先生に相談したいこと」「次週やること」を明確にする。\n7. 発表準備メモとSOAP下書きの対象run・結論・次工程を揃える。過去の提出原本は保持する。')])

 replace('docs/notes/experiment_log.md','Point to latest completed evidence instead of September 14 only',[
 ('## 直近の固定記録\n',
  '## 直近の固定記録\n\n- [9/15：3/22 kHz各7ノイズ、卒論比較・帯域SNR・説明性](../../experiments/2026-09-15_onb_frequency_comparison/analysis.md)：最新の比較根拠。\n- [9/15：22 kHz単独解析](../../experiments/2026-09-15_onb_22khz_noise_sweep/analysis.md)：先行結果。\n')])
 replace('docs/code_map.md','Link current analysis scripts and explain model-dependent diagnostics',[
 ('- [今回の数値採取スクリプト]', '- [9/14結果の数値採取スクリプト]'),
 ('## 過去コード・資料処理',
  '- [9/15・2帯域解析スクリプト](../experiments/2026-09-15_onb_frequency_comparison/analysis.md): 完了runの抽出、WAV/chunk指標の検算、卒論比較、54区間の帯域SNR。学習なしの後処理。\n- [解析の進め方](analysis_workflow.md): 代表条件を選ぶ問いと、性能・ノイズ・XAIの因果検証を分ける手順。\n\n## 過去コード・資料処理')])
 replace('docs/notes/data_notes.md','Correct display spacing versus physical resolution and connect band-SNR evidence',[
 ('最大周波数を変えて毎回224画素にする比較では、周波数分解能や補間も変わる。純粋な帯域除去実験とは区別する。',
  '最大周波数を変えて毎回224列にすると、1列のHz幅と補間・縮小が変わる。元STFTの周波数刻み32.8125 Hzは共通であり、3 kHz画像の約13.4 Hz/列は物理分解能の向上ではない。純粋な帯域除去実験とは区別する。'),
 ('旧データの扱いと生成不具合の経緯は',
  '帯域別の実現SNRと同一ノイズ対応は[9/15の54区間診断](../../experiments/2026-09-15_onb_frequency_comparison/analysis.md)で確認した範囲の根拠。付加前の元録音にも背景音が含まれ、純粋な気泡音とは限らない。\n\n旧データの扱いと生成不具合の経緯は')])
 update('docs/notes/model_notes.md','Make explanation targets and acceptance limits explicit',lambda s:s+'''
## 説明性で答える問いと、現在の使い方

| 手法 | 説明対象 | 数値・解釈で確認すること |
|---|---|---|
| TreeSHAP | RFのPCA成分が予測に与える正負の寄与 | 期待値＋寄与和の再構成。PCA成分を元画素・物理周波数へ直結しない |
| IG | 深層モデルのbaselineから入力への回帰出力差 | 寄与総和と予測差の一致、積分経路・点数・baseline。現行2帯域では大きな不一致が残る |
| Grad-CAM | AlexNet最終畳み込み特徴の粗い局在 | 正負の寄与分解ではない。細かい周波数・気泡時刻を同定しない |
| 局所occlusion | 代表chunkの帯域/時間を置換したときの予測変化 | 代表例の選び方、予測の符号、マスクの重なりと非加法性 |
| 集合マスク | 検証WAV群の帯域/時間置換による性能変化 | fold内WAV評価。pooled主R²とは別。ゼロ置換の分布変化も含む |
| deletion/insertion | 重要画素を消す/戻す順による回帰出力曲線 | 出力単位の面積で、分類ROC-AUCではない。符号・baseline・ランダム順位との比較が必要 |
| 安定性・ランダム化 | 小入力摂動へのmap安定性、学習パラメータへの依存 | 絶対値mapの類似だけでIG整合性・物理妥当性を保証しない。現行は最終層だけの部分診断 |

現在の固有手法は単体モデルのchunk回帰出力を説明する。WAV中央値・アンサンブル全体・ONBの物理的発生原因を直接分解しているわけではない。Attention Rollout、RISE、Guided Grad-CAM、CA-LIGは旧候補・論文紹介と現行採用を分ける。

最新の数値と帯域依存は[2帯域解析の§6](../../experiments/2026-09-15_onb_frequency_comparison/analysis.md)、検証を進める方法は[解析手順](../analysis_workflow.md)。手法を「採用済み」であることと、「説明として数値的・物理的に検証済み」であることを区別する。
''')
 update('docs/progress/weekly_report_guide.md','Carry user hypothesis, evidence units and updates into recurring reporting',lambda s:s+'''
## 9/15以降の引継ぎ事項

本人の[研究仮説](../research_plan/2026-09-15_master_thesis_hypothesis.md)をSの基礎にし、Oには今回固定した結果、Aには支持された範囲と原因仮説、Pにはその仮説を区別できる比較を書く。劣化抑制の希望を確定結果へ読み替えず、改善しない条件も根拠とともに示す。

最新の解析が増えたら、[発表準備メモ](../research_plan/2026-09-18_xai_progress_brief.md)とSOAPの対象・数値・次工程を同時に揃える。参照すべき手順は[解析手順](../analysis_workflow.md)。古い下書きは監査や日付付き記録へ保持し、提出済み原本を最新値で置換しない。
''')
 update('templates/weekly_progress_SOAP.md','Add lightweight evidence prompts to reusable SOAP template',lambda s:s+'''
<!-- 作成時の確認（提出時には削除）
S: 本人の仮説・希望と今回答えたい問い。
O: 収録日/解析日・run・評価単位・実際に完了した範囲・根拠へのリンク。
A: 支持された点、未確認の点、原因候補。絶対性能/劣化量/ONBを分ける。
P: 次の比較で何を区別するか、終わりの基準。実行中と未着手を混同しない。
現在地・発表準備メモと一致させる。手順は docs/analysis_workflow.md。
-->
''')
 replace('experiments/_template/run_summary.md','Record controlled comparisons, competing causes and stopping evidence',[
 ('- 研究上の問い:', '- 研究上の問い:\n- 本人の仮説と、支持/弱化のどちらでも今回記録する結果:\n- 今回の対象外run・対象外条件:'),
 ('- 周波数上限 / chunk / reference SNR / 実現SNR:', '- 周波数上限 / 元STFTのHz刻み / resize後の列間隔 / chunk / reference SNR / 帯域内実現SNR:'),
 ('- 最良単体との比較方法・事前選択/事後選択の区別:',
  '- 最良単体との比較方法・事前選択/事後選択の区別:\n- 無雑音の基準性能・ノイズによる低下量・強ノイズの絶対性能:\n- 統合で訂正された/追加されたONB誤り:'),
 ('- 今回確認できたこと:', '- 今回確認できたこと:\n- 原因候補ごとの根拠・反証・区別する追加比較:'),
 ('- 現在地/進捗索引の更新先:', '- 現在地/進捗索引/発表草稿の更新先:\n- 再利用する解析手順: docs/analysis_workflow.md')])
 update('docs/repository_management.md','Explain current versus historical revisions and report synchronization',lambda s:s+'''
## 更新判断の最小ルール

現行の目的・現在地・運用メモ・未提出草稿は、本人の希望と最新の固定根拠に沿って本文を更新する。日付付き分析・提出済み原本は保存し、変わった判断と後継を注記する。汎用テンプレートには、今回限りの数値ではなく再発を防ぐ確認項目を加える。

新しい解析の終了時は、現在地→進捗索引→発表準備中のSOAP/構成案を揃える。改訂前後の内容・対象範囲は[文書監査](audits/2026-09-15_document_consistency/README.md)で追える。[原資料ガイド](original_document_guide.md)と[解析手順](analysis_workflow.md)も参照する。
''')

 history={
 'docs/research_plan/2026-08-17_noise_accuracy_reversal_investigation.md':'本文の「現行逆転の主因」は旧相対RMSデータの診断を指す。9/15の固定global RMSデータの非単調性の原因を確定したものではない。matchedも条件に合わせた学習を含むノイズ下性能として有用で、固定モデルの転送と区別する。',
 'docs/research_plan/2026-08-28_research_landing_xai_ensemble_assessment.md':'当時のRF優位・全モデル共通帯域・ONBを副評価へ回す提案は現行方針へ引き継がない。log-power後の順位と帯域は異なり、本人はアンサンブルの劣化抑制とその原因を中心仮説として指定している。',
 'docs/research_plan/2026-08-30_treeshap_guided_gradcam_ensemble_recovery_assessment.md':'RFを固定の基準モデルとする提案や追加モデル/stackingは当時の候補。現行innerは元WAV非重複へ修正済み。今回はIGが両帯域で不整合であり、物理的補完の確定や新方式の採用を意味しない。',
 'docs/research_plan/2026-06-26_xai_method_selection.md':'手法の候補選定の記録。PCAを変更し終えるまで共通マスクを実施できないという必須の順序にはしない。現行ではPCA空間SHAPと元スペクトログラムのマスクを分担して出力済み。',
 'docs/research_plan/2026-06-26_xai_method_and_metric_selection.md':'本文の「確定」は当時の手法選定。現在はTreeSHAP・IG・Grad-CAM・共通マスクを出力済みだが、IGの数値整合は未解決。手法選定と説明としての検証完了を分ける。',
 'docs/research_plan/2026-07-16_model_specific_xai_selection_and_implementation.md':'モデル別の手法分担は現行と接続するが、IGの寄与量を研究上の根拠に使える状態とは限らない。9/15の両帯域で420/420例の不整合を確認した。',
 'docs/research_plan/2026-07-24_explainability_output_guide.md':'出力の読み方は実装時点の記録。最新は単体chunkの説明、fold内WAVマスク、IG積分不整合、回帰のdeletion/insertion面積を区別する。',
 'docs/research_plan/2026-07-24_current_progress_and_priorities.md':'旧ノイズ生成の診断、現行データ生成、log-power改善、元WAV分離、一般化4方針の実装は進展済み。本文の未着手と優先順位を現在へそのまま戻さない。',
 'docs/research_plan/2026-09-14_current_state_and_next_steps.md':'現在の解析根拠は06.11の3/22 kHz各7条件。旧6条件/105条件の成績・ONB見逃しを最新runへ代入せず、本人の劣化抑制の仮説と追加比較に沿って次工程を選ぶ。',
 'docs/research_plan/2026-09-14_execution_policy_and_generalization_design.md':'70条件完了は金曜の準備の前提ではない。今回の対象は本人の指定で完了済み06.11の2帯域に限定。一般化は実装済みで本性能の確認とは別。',
 'docs/research_plan/2026-09-14_result_layout_and_generalization.md':'保存階層・一般化4方針の実装記録として有効。本文の6条件/3方式は当時の試行で、最新比較は2帯域各7条件/2方式。追加実行の採用範囲は現在地で指定する。',
 'experiments/2026-09-15_research_status_snapshot/README.md':'本書は9/14の6条件と9/15午前の状態の固定記録。最新2帯域比較の数値・IG例数・ONB結果へ読み替えない。',
 'experiments/2026-09-15_onb_22khz_noise_sweep/README.md':'22 kHz単独7条件の入口。3 kHzとの比較・本人の研究仮説・帯域差の原因分析は後続に追加済み。',
 }
 for name,note in history.items():historical(name,note)
 # Original Markdown research plans can be encountered without passing through docs/.
 for name in BEFORE:
  if name.startswith('研究進捗報告/') and name.endswith('.md'):
   note=('論文輪講の理論紹介と当時の自研究への接続案。現在のIG実装の信頼性検証やCA-LIG採用を示すものではない。'
         if '/605' in name else '当時の発表・計画・指導後整理の履歴。本文の実行状況・次の作業・モデル順位はその時点に限る。')
   historical(name,note)
 update('docs/audits/2026-09-15_workspace_review/README.md','Point prior inventory to the later comprehensive text audit',lambda s:s.replace('\n\n実施日:',
  '\n\n> 同日後続の[文書整合性監査](../2026-09-15_document_consistency/README.md)で、追加2帯域解析と本人の仮説を反映した。本書の91 Markdown・92 PDF等は先行時点の棚卸しとして保持する。\n\n実施日:',1))
 update('experiments/README.md','Add reusable analysis workflow and original-document interpretation',lambda s:s+'''
文書の更新判断・原資料との対応は[文書整合性監査](../docs/audits/2026-09-15_document_consistency/README.md)、追加結果を読む順序は[解析手順](../docs/analysis_workflow.md)を参照する。文書監査は新しい学習結果ではない。
''')
 (AUDIT/'changes.json').write_text(json.dumps(CHANGES,ensure_ascii=False,indent=2),encoding='utf-8')
 print(f'Updated {len(CHANGES)} documents')


if __name__=='__main__':main()

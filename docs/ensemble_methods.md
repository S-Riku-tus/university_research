# ONB回帰のアンサンブル手法と現行の重み決定

更新日: 2026-09-24。

**現行定義**：9/24本比較の`inner_holdout`は、学習日の元WAV非共有holdoutに対する**1秒chunk R²**から `normalize(1 / max(1-R², 1e-6))` で重みを決める。旧版にあった「WAV中央値R²」は9/17以前の定義であり、現行コードには該当しない。深層モデルのepochを選ぶ元WAV非共有3-fold `rmse_all` validationとは別処理である。詳細は[研究方針と次の検証](research_plan/2026-09-24_ensemble_research_position_and_next_steps.md)を参照する。

重みは全実験共通の定数ではない。`matched`ではnoiseごとの学習familyから毎回求め、`clean_only`だけはcleanで求めた重みを複数の評価noiseへ共有する。別の学習実験日を指定すれば、そのデータから重みを再計算する。

下記5方式は実装済みだが、9/24本比較で実行したのは`simple_equal`と`inner_holdout`である。このページは全方式を直近に実行する指示ではない。別に`performance_kfold`と再現用`val_fold_legacy`も実装されている。

本書は、熱流束回帰に用いる3モデル、Random Forest系モデル（RF）、CNN＋Transformer、AlexNetを、5つの方法でどのように統合するかを数式とともに整理する。対象は次の5方式である。

1. `simple_equal`: 単純等重み平均
2. `inner_holdout`: inner holdoutの単体R²に基づく逆誤差重み
3. `subset_equal_cv`: crossfit予測による部分集合選択＋等重み平均
4. `crossfit_wav_stack`: crossfit予測によるWAV単位制約付きstacking
5. `crossfit_shrinkage_stack`: 等重みへの縮小を加えたWAV単位stacking

## 1. 共通する記号と最終的な統合式

モデル数を $M=3$ とし、モデルを $m=1,2,3$ で表す。

- $m=1$: RF
- $m=2$: CNN＋Transformer
- $m=3$: AlexNet
- $i$: 元WAVの番号
- $c\in C_i$: WAV $i$ に含まれるchunk
- $y_i$: WAV $i$ の熱流束の真値
- $\hat y_{imc}$: モデル $m$ がWAV $i$ のchunk $c$ に出した予測
- $w_m$: モデル $m$ の統合重み

5方式とも、各chunkの統合予測は次式で求める。

$$
\hat y^{\mathrm{ens}}_{ic}(\mathbf w)
=
\sum_{m=1}^{M}w_m\hat y_{imc}.
$$

重みには原則として次の凸結合制約を課す。

$$
w_m\geq0,
\qquad
\sum_{m=1}^{M}w_m=1.
$$

したがって統合値は、そのchunkに対する3モデルの予測値の最小値と最大値の間に入る。負の重みを使った外挿は行わない。

現行runの通常性能評価は1秒chunk単位であり、式のchunk統合値をそのまま評価する。一方、後述する3つのcrossfit方式は、**重みをfitする内部目的だけ**でchunk統合後のWAV内中央値を使う。

$$
\hat y^{\mathrm{ens}}_{i}(\mathbf w)
=
\operatorname{median}_{c\in C_i}
\left[
\sum_{m=1}^{M}w_m\hat y_{imc}
\right].
\tag{1}
$$

ここで重要なのは、一般に次の2つが一致しないことである。

$$
\operatorname{median}_{c}
\left[
\sum_m w_m\hat y_{imc}
\right]
\neq
\sum_m w_m
\operatorname{median}_{c}
\left[
\hat y_{imc}
\right].
\tag{2}
$$

新しい3方式は式(1)を内部の重みfitに用いる。現行`inner_holdout`はWAV中央値を使わず、holdoutの全chunkに対するモデル別R²から重みを決める。

## 2. `simple_equal`: 単純等重み平均

### 基本的な考え方

3モデルを同じ重要度として扱い、学習による重み推定を行わずに平均する。

一般の $M$ モデルでは、

$$
w_m=\frac{1}{M}
$$

とする。現在の3モデルでは、

$$
\mathbf w_{\mathrm{equal}}
=
\left(
\frac13,\frac13,\frac13
\right).
$$

したがってchunk予測は、

$$
\hat y^{\mathrm{equal}}_{ic}
=
\frac{
\hat y_{i,\mathrm{RF},c}
+\hat y_{i,\mathrm{CNN},c}
+\hat y_{i,\mathrm{AlexNet},c}
}{3}
$$

となる。crossfit方式の内部目的でWAV値が必要な場合は、このchunk統合値のWAV内中央値を用いる。

$$
\hat y^{\mathrm{equal}}_{i}
=
\operatorname{median}_{c\in C_i}
\left[
\hat y^{\mathrm{equal}}_{ic}
\right].
$$

### この方式が利用する情報

単体モデルの精度、誤差分散、モデル間の誤差相関は重み決定に使わない。すべてのモデルが同程度に信頼できるという単純な仮定に相当する。

### 長所と弱点

重み推定誤差がなく、WAV数が少なくても重みが変動しない。一方で、特定条件で大きく悪化したモデルにも必ず1/3の重みを与える。そのモデルの誤差が他モデルで相殺されなければ、最良単体より統合精度が低くなる。

## 3. `inner_holdout`: 単体R²の逆誤差重み

### 基本的な考え方

外側学習集合の一部を重み決定用holdoutとし、そのholdoutで単体性能が高かったモデルへ大きな重みを与える。

現行条件ではholdout割合を20%とする。例えば外側学習集合が12 WAVなら、9 WAVで重み算出用の一時モデルを学習し、残り3 WAVで各モデルのR²を求める。この3 WAVは元WAV単位で分離され、9 WAV側とchunkを共有しない。

重み決定用holdoutに含まれる全chunkの集合を $H$ とする。モデル $m$ のholdout上の決定係数を、

$$
R_m^2
=
1-
\frac{
\sum_{(i,c)\in H}(y_i-\hat y_{imc})^2
}{
\sum_{(i,c)\in H}(y_i-\bar y_H)^2
},
$$

$$
\bar y_H
=
\frac{1}{|H|}
\sum_{(i,c)\in H}y_i
$$

として求める。

次に、単体誤差を、

$$
e_m=1-R_m^2
$$

と定義し、その逆数を未正規化重みとする。

$$
a_m=\frac{1}{\max(e_m,\varepsilon)}.
$$

$\varepsilon$ はゼロ除算や数値誤差を防ぐ小さな正数であり、現行実装では $10^{-6}$ である。最終的な重みは、

$$
w_m
=
\frac{a_m}{\sum_{j=1}^{M}a_j}
$$

となる。同じholdout標本では $1-R_m^2=\mathrm{SSE}_m/\mathrm{SST}$ なので、この方式は正規化後には逆MSE重みと同じである。

例えば $R^2=(0.90,0.80,0.50)$ なら、

$$
e=(0.10,0.20,0.50),
\qquad
a=(10,5,2),
$$

したがって、

$$
\mathbf w
=
\left(
\frac{10}{17},
\frac{5}{17},
\frac{2}{17}
\right)
\approx
(0.588,0.294,0.118)
$$

となる。

### この方式が利用する情報

各モデル単体のholdout R²だけを利用する。モデル $m$ とモデル $j$ の残差共分散、

$$
\operatorname{Cov}
\left(
y-\hat y_m,
y-\hat y_j
\right)
$$

は直接扱わない。このため、2モデルが同じ方向に間違える場合と、逆方向に間違えて相殺できる場合を、単体R²が同じなら区別できない。

### 長所と弱点

等重みよりも学習日clean holdoutでの単体性能を反映でき、仕組みも比較的単純である。一方、一度の少数holdout WAVから求めたR²は変動しやすい。また、単体性能を個別に重みへ変換する方式であり、統合後のMSE、モデル間残差相関、noise下の性能順位変化を直接扱わない。

## 4. 新しい3方式で共通するcrossfit予測

`subset_equal_cv`、`crossfit_wav_stack`、`crossfit_shrinkage_stack`は、重み決定用の予測としてcrossfit、すなわちinner OOF予測を使用する。

現行方式は元WAV単位のinner 4-foldである。外側学習集合が12 WAVなら、「9 WAVで学習し、学習に使わなかった3 WAVを予測する」処理を4回行う。これにより12 WAVすべてについて、自分自身を学習に含めていない予測を得る。3つの新方式はこの同じOOF予測を使い、以降の重みの選び方だけが異なる。

重み学習対象の各WAV $i$ について、そのWAVを学習に含めずにモデル $m$ が出したchunk予測を、

$$
\hat y^{\mathrm{OOF}}_{imc}
$$

と書く。この予測を使った任意の重み $\mathbf w$ に対するWAV予測を、

$$
g_i(\mathbf w)
=
\operatorname{median}_{c\in C_i}
\left[
\sum_{m=1}^{M}
w_m\hat y^{\mathrm{OOF}}_{imc}
\right]
\tag{3}
$$

と定義する。

WAV数を $N$ とすると、重み評価に用いるWAV単位平均二乗誤差は、

$$
L(\mathbf w)
=
\frac{1}{N}
\sum_{i=1}^{N}
\left[
y_i-g_i(\mathbf w)
\right]^2
\tag{4}
$$

である。各WAVを1標本として平均するため、1本のWAVから多数のchunkが得られても、そのWAVが他のWAVより大きな重みを持つことはない。

## 5. `subset_equal_cv`: 部分集合選択＋集合内等重み

### 基本的な考え方

連続的な重みを細かく推定する代わりに、使用するモデルの集合だけを選び、選ばれたモデルは等重みにする。

全モデル集合を、

$$
\mathcal M=\{1,2,\ldots,M\}
$$

とする。空集合を除く任意の部分集合 $S\subseteq\mathcal M$ に対し、重みを、

$$
w_m^{(S)}
=
\begin{cases}
\dfrac{1}{|S|}, & m\in S,\\[6pt]
0, & m\notin S
\end{cases}
$$

と定める。

3モデルの場合、候補は $2^3-1=7$ 個である。

$$
\begin{aligned}
&(1,0,0),\quad(0,1,0),\quad(0,0,1),\\
&\left(\frac12,\frac12,0\right),\quad
\left(\frac12,0,\frac12\right),\quad
\left(0,\frac12,\frac12\right),\\
&\left(\frac13,\frac13,\frac13\right).
\end{aligned}
$$

式(4)のWAV MSEが最小となる部分集合を選ぶ。

$$
S^*
=
\underset{\varnothing\neq S\subseteq\mathcal M}{\arg\min}
\;L\left(\mathbf w^{(S)}\right).
\tag{5}
$$

最終的な重みは、

$$
\mathbf w^*=\mathbf w^{(S^*)}
$$

である。

### この方式が表す選択

例えば $S^*=\{\mathrm{RF},\mathrm{CNN}\}$ なら、

$$
\hat y^{\mathrm{ens}}_{ic}
=
\frac12\hat y_{i,\mathrm{RF},c}
+
\frac12\hat y_{i,\mathrm{CNN},c}
$$

とし、AlexNetは使用しない。

単体だけの部分集合も候補なので、アンサンブルによる改善余地がないと判断された場合は、1モデルだけを選択できる。つまりこの方式は、必ず複数モデルを混ぜる方式ではなく、「単体を含む候補群から、安定した離散的な重みを選ぶ方式」である。

### 長所と弱点

推定するのは集合の選択だけなので、小標本でも重みの自由度を抑えやすい。大きく悪化したモデルの重みを完全にゼロにできる。一方、選んだ集合内は必ず等重みであり、RF 0.2、CNN 0.8のような細かな性能差や補完関係は表現できない。

## 6. `crossfit_wav_stack`: WAV損失を直接最小化するstacking

### 基本的な考え方

単体R²を別々に重みへ変換するのではなく、3モデルを統合した結果のWAV MSEを直接最小化する。

許される重みの集合、すなわち単体を含む3モデルの凸結合を、

$$
\Delta_M
=
\left\{
\mathbf w\in\mathbb R^M
\mid
w_m\geq0,\;
\sum_{m=1}^{M}w_m=1
\right\}
$$

とする。求める重みは、

$$
\mathbf w^*_{\mathrm{stack}}
=
\underset{\mathbf w\in\Delta_M}{\arg\min}
\;L(\mathbf w)
\tag{6}
$$

である。

式(4)を代入すると、

$$
\mathbf w^*_{\mathrm{stack}}
=
\underset{\mathbf w\in\Delta_M}{\arg\min}
\frac1N
\sum_{i=1}^{N}
\left[
y_i-
\operatorname{median}_{c\in C_i}
\left(
\sum_{m=1}^{M}
w_m\hat y^{\mathrm{OOF}}_{imc}
\right)
\right]^2.
\tag{7}
$$

### 誤差の補完を扱える理由

モデル $m$ の残差を、

$$
e_{im}=g_i(\mathbf e_m)-y_i
$$

とする。平均集約だけを使う通常の線形な場合、統合残差は概念的に、

$$
e_i^{\mathrm{ens}}
=
\sum_mw_me_{im}
$$

となり、統合MSEには各モデルの誤差だけでなく、モデル間の誤差積、

$$
w_mw_j
\operatorname{Cov}(e_m,e_j)
$$

も影響する。したがって、片方が過大予測、もう片方が過小予測する場合には、その相殺を利用できる。

現行方式は中央値を含むため厳密には上の線形式にならないが、式(7)で統合後の予測を直接評価することにより、実際の中央値集約後に残る補完効果を重み選択へ反映する。

例えば真値100に対して、ある範囲でRFが110、CNNが90を予測するなら、

$$
0.5\times110+0.5\times90=100
$$

となる。`inner_holdout`は両モデルの単体誤差を個別に見るが、`crossfit_wav_stack`はこの組合せによる誤差減少を直接評価する。

### 長所と弱点

部分集合方式より自由度が高く、連続的な重みで補完関係を利用できる。反面、独立WAV数が少ないと、inner OOFに偶然よく適合する極端な重みを推定する可能性がある。また、式(7)は中央値を含む区分的で非凸な目的関数であるため、数値的に得られた解を数学的な大域最適解とはみなさない。

## 7. `crossfit_shrinkage_stack`: 等重みへ縮小するstacking

### 基本的な考え方

`crossfit_wav_stack`のWAV MSE最小化に、重みが等重みから離れすぎることへの罰則を加える。少数WAVに偶然適合した極端な重みを抑えることが目的である。

等重みベクトルを、

$$
\mathbf u
=
\left(
\frac1M,\ldots,\frac1M
\right)
$$

とする。現在の3モデルでは、

$$
\mathbf u
=
\left(
\frac13,\frac13,\frac13
\right).
$$

まず、MSEと罰則の数値スケールを揃えるため、各単体モデルのWAV MSEの平均を、

$$
s
=
\frac1M
\sum_{m=1}^{M}
L(\mathbf e_m)
$$

とする。ここで $\mathbf e_m$ はモデル $m$ だけに重み1を置く単位ベクトルである。

縮小付き目的関数は、

$$
J_\lambda(\mathbf w)
=
\frac{L(\mathbf w)}{s}
+
\lambda
\left\|
\mathbf w-\mathbf u
\right\|_2^2
\tag{8}
$$

である。求める重みは、

$$
\mathbf w^*_{\mathrm{shrink}}
=
\underset{\mathbf w\in\Delta_M}{\arg\min}
J_\lambda(\mathbf w).
\tag{9}
$$

現在の方式では、

$$
\lambda=0.1
$$

を使用する。3モデルの場合の罰則部分は、

$$
\lambda
\left[
\left(w_1-\frac13\right)^2
+\left(w_2-\frac13\right)^2
+\left(w_3-\frac13\right)^2
\right]
$$

となる。

### $\lambda$ の意味

- $\lambda=0$: 罰則がなく、`crossfit_wav_stack`と同じ目的になる。
- $\lambda>0$: 同程度のWAV MSEなら、等重みに近い重みを選びやすくなる。
- $\lambda$ が非常に大きい: 重みは等重みへ近づき、`simple_equal`に近い性質になる。

式(8)でMSEを $s$ により正規化するのは、熱流束の数値単位を変えたときに罰則の相対的な強さが変わるのを防ぐためである。例えば全ての真値と予測を一律に $10^3$ 倍するとMSEは $10^6$ 倍になるが、$L(\mathbf w)/s$ は変わらない。

### 通常stackingとの違い

通常stackingが、

$$
\mathbf w=(0,1,0)
$$

のような1モデルへの集中を選ぶとき、縮小付き方式は、WAV MSEの増加が小さければ、

$$
\mathbf w=(0.08,0.84,0.08)
$$

のような等重みに近い解を選び得る。これにより重み推定の分散を抑えることを狙う。

ただし、これは最適化後の重みと等重みを固定比率 $\gamma$ で混ぜる、

$$
(1-\gamma)\mathbf u+\gamma\mathbf w^*
$$

という方式ではない。現行方式は式(8)の罰則を含む目的関数を直接最小化する。

### 長所と弱点

連続重みの柔軟性を残しながら、少数WAVによる極端な重みを抑えられる可能性がある。一方、実際には除外すべきモデルにも重みを残すことがあり、$\lambda=0.1$ があらゆる周波数・ノイズ条件で最適であるとは限らない。

## 8. 5方式の数理的な違い

| 方式 | 重みを決める情報 | 許される重み | 最小化・選択する量 | 誤差の補完を直接評価 |
|---|---|---|---|---|
| `simple_equal` | なし | $(1/3,1/3,1/3)$ 固定 | なし | しない |
| `inner_holdout` | 元WAV非共有holdoutの各単体chunk R² | 非負・合計1 | 各単体の $1-R^2$ を逆数化（逆MSE相当） | しない |
| `subset_equal_cv` | inner OOFの統合予測 | 単体または部分集合内の等重み | 7候補のWAV MSE | 候補の範囲で評価 |
| `crossfit_wav_stack` | inner OOFの統合予測 | simplex上の連続重み | 統合後WAV MSE | する |
| `crossfit_shrinkage_stack` | inner OOFの統合予測 | simplex上の連続重み | 正規化WAV MSE＋等重み距離 | する |

重みの自由度は、おおむね次の順に大きくなる。

$$
\text{simple equal}
\;<\;
\text{subset equal}
\;<\;
\text{continuous stacking}.
$$

一方、小標本での重みの安定性は一般に逆方向の関係を持ちやすい。`crossfit_shrinkage_stack`は、連続stackingの自由度を保ちながら等重み側へ制約することで、この両者の中間を狙う。

## 9. 「単体より悪化する」現象に対する各方式の位置づけ

`simple_equal`は、悪いモデルにも必ず重みを与えるため、単体より悪化し得る。

`inner_holdout`は、単体性能が悪いモデルの重みを下げられる。ただし、少数holdoutでのR²推定誤差と、モデル間誤差相関を扱わないことが弱点となる。

`subset_equal_cv`は、悪化原因となるモデルを完全に除外できる。また単体候補を含むため、inner OOFのWAV MSE上では、候補に含まれる最良単体より悪い重みを選ばない。ただし、この性質は重み学習集合内のものであり、未知の外側WAVに対する保証ではない。

`crossfit_wav_stack`は、各モデルが異なる方向へ間違える場合に、その補完を最も柔軟に利用できる。ただし、WAV数が少ないと推定された連続重み自体が過学習する可能性がある。

`crossfit_shrinkage_stack`は、その過学習を抑えるために等重みを基準として使う。予測誤差だけを最小化する解よりも少し保守的な重みを選び、未知WAVでの安定性向上を狙う。

どの方式も、3モデルが同じWAV・chunkを同じ方向に誤予測する場合には、統合だけで大きく改善できない。また、全モデルの予測が同一chunkでONB閾値未満なら、非負・合計1の凸結合も閾値未満になる。したがって、回帰誤差の改善と最初のONB点の検知改善は分けて評価する必要がある。

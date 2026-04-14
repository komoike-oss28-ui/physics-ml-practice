import pymc as pm
import pytensor.tensor as pt
import numpy as np

# ダミーデータの情報（本来は pandas 等から読み込みます）
n_subjects = 1000 # 保健師の人数
n_items = 10      # 質問項目の数
# subj_idx, item_idx, X_obs, C_obs は長さが等しい1次元配列と想定

def build_weighted_2pl_model(subj_idx, item_idx, X_obs, C_obs, n_subjects, n_items):
    """
    パターン1：確信度を重みとして扱う2PLモデル
    """
    with pm.Model() as weighted_model:
        
        # ---------------------------------------------------------
        # 1. パラメータの事前分布（Prior）の定義
        # ---------------------------------------------------------
        # 被験者能力値 theta (標準正規分布を仮定)
        theta = pm.Normal("theta", mu=0.0, sigma=1.0, shape=n_subjects)
        
        # 項目難易度 b (正規分布)
        b = pm.Normal("b", mu=0.0, sigma=1.0, shape=n_items)
        
        # 項目識別力 a (対数正規分布: 識別力は必ずプラスになるため)
        a = pm.LogNormal("a", mu=0.0, sigma=0.5, shape=n_items)
        
        # ---------------------------------------------------------
        # 2. 2PLモデルの基本式の計算
        # ---------------------------------------------------------
        # theta と b, a を組み合わせて、真に正答する確率(P)のロジットを計算
        # eta = a * (theta - b)
        eta = a[item_idx] * (theta[subj_idx] - b[item_idx])
        
        # ロジットを確率(0.0〜1.0)に変換（シグモイド関数 / 逆ロジット関数）
        p = pm.math.invlogit(eta)
        
        # ---------------------------------------------------------
        # 3. 尤度（Likelihood）の計算と「重み(確信度)」の適用
        # ---------------------------------------------------------
        # 通常のIRTなら pm.Bernoulli("obs", p=p, observed=X_obs) で終わりますが、
        # 今回は確信度(C_obs)を重みとして掛け算したいため、対数尤度を自分で計算します。
        
        # (1) もし X=1 なら log(p)、X=0 なら log(1-p) を取得する
        logp_standard = pt.switch(X_obs, pt.log(p), pt.log(1 - p))
        
        # (2) 計算された対数尤度に、確信度（C_obs）を掛け算する
        # 確信度が低い（例:0.5）と、このデータがパラメータ推定に与える影響力が半減します。
        weighted_logp = C_obs * logp_standard
        
        # (3) カスタムの尤度としてモデルに組み込む（pm.Potentialを使用）
        pm.Potential("weighted_obs", weighted_logp)
        
    return weighted_model

# 実行・推定のイメージ（※データセットがある前提）
# model1 = build_weighted_2pl_model(subj_idx, item_idx, X_obs, C_obs, n_subjects, n_items)
# with model1:
#     trace1 = pm.sample(draws=2000, tune=1000, chains=4) # ここでMCMCが一斉に回ります


import pymc as pm
import numpy as np

def build_misclassification_2pl_model(subj_idx, item_idx, X_obs, C_obs, n_subjects, n_items):
    """
    パターン2：真の達成状態と評価者の誤分類を分離した階層モデル（動的4PL）
    """
    with pm.Model() as misclass_model:
        
        # ---------------------------------------------------------
        # 1. 従来IRTパラメータの定義（真実の力を測るための変数）
        # ---------------------------------------------------------
        theta = pm.Normal("theta", mu=0.0, sigma=1.0, shape=n_subjects)
        b = pm.Normal("b", mu=0.0, sigma=1.0, shape=n_items)
        a = pm.LogNormal("a", mu=0.0, sigma=0.5, shape=n_items)
        
        # ---------------------------------------------------------
        # 2. 【新規性】確信度をエラー率に変換するパラメータの定義
        # ---------------------------------------------------------
        # エラー率を予測する回帰モデルの「切片(gamma0)」と「傾き(gamma1)」を同時に推定します。
        
        # gamma0: 確信度がゼロのときの基本エラー率（ロジットスケール）
        gamma0 = pm.Normal("gamma0", mu=0.0, sigma=1.0)
        
        # gamma1: 確信度(C)が上がった時に、どれくらいエラー率を下げるかの係数
        # 通常、確信度が高いほどエラーは減るはずなので、マイナスになることを期待して事前分布を設定
        gamma1 = pm.Normal("gamma1", mu=-1.0, sigma=1.0)
        
        # 確信度(C_obs)から、データ1件ごとの「エラー発生確率(epsilon)」を計算
        # epsilon = invlogit(gamma0 + gamma1 * C)
        epsilon = pm.math.invlogit(gamma0 + gamma1 * C_obs)
        
        # ---------------------------------------------------------
        # 3. 真の達成確率(P*)の計算 (ステップ1: 神の視点)
        # ---------------------------------------------------------
        # 保健師の純粋な実力から導かれる、真実の正答確率
        eta = a[item_idx] * (theta[subj_idx] - b[item_idx])
        p_star = pm.math.invlogit(eta)
        
        # ---------------------------------------------------------
        # 4. 観測される確率(P_obs)の計算 (ステップ2: 誤分類の混入)
        # ---------------------------------------------------------
        # ここでは数式をシンプルにするため、偽陽性と偽陰性の確率が同じ(epsilon)と仮定します。
        # P(X=1) = 実力で達成して正しく評価される確率 + 未達成だけどオマケで1に誤分類される確率
        # P_obs = p_star * (1 - epsilon) + (1 - p_star) * epsilon
        
        p_obs = p_star * (1.0 - epsilon) + (1.0 - p_star) * epsilon
        
        # ---------------------------------------------------------
        # 5. 尤度の計算 (データのフィッティング)
        # ---------------------------------------------------------
        # 最終的に計算された p_obs を用いて、実際のラベル X_obs を予測する
        pm.Bernoulli("obs", p=p_obs, observed=X_obs)
        
    return misclass_model

# 実行・推定のイメージ（MCMCによる同時推定）
# model2 = build_misclassification_2pl_model(subj_idx, item_idx, X_obs, C_obs, n_subjects, n_items)
# with model2:
#     trace2 = pm.sample(draws=2000, tune=1000, chains=4) # MCMC実行
#     # 推定後、az.summary(trace2, var_names=["gamma0", "gamma1"]) などで
#     # 「AIの自己認知（確信度）がどれくらい正確だったか」を確認できます！
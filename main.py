import numpy as np
import matplotlib.pyplot as plt   
from scipy.optimize import minimize 


def calculate_mean(data):
    """データの平均を計算する（物理実験データの処理などを想定）"""
    return np.mean(data)

if __name__ == "__main__":
    sample_data = np.array([1.2, 2.5, 3.8, 4.2, 5.1])
    print(f"Mean: {calculate_mean(sample_data)}")

def calculate_stats(data):
    mean = np.mean(data)
    std = np.std(data)
    max_val = np.max(data)
    min_val = np.min(data)
    return mean, std, max_val, min_val

def irt_2pl_demo(theta, a, b):
    """
    Parameters
    ----------
    theta : float or array-like
        受験者の能力値
    a : float
        識別力パラメータ（discrimination）
    b : float
        困難度パラメータ（difficulty）

    Returns
    -------
    float or np.ndarray
        正答確率 P(theta)
    """
    return 1 / (1 + (1.7 * np.exp(-a * (theta - b))))


# 単一の能力値
print(irt_2pl_demo(theta=-0.14, a=1.0, b=0.2))  # → 0.3775

# 複数の能力値（配列）
thetas = np.linspace(-4, 4, 200)
p = irt_2pl_demo(thetas, a=1.0, b=0.0)

# ICCを描画
plt.plot(thetas, p)
plt.axhline(0.5, color='gray', linestyle='--')
plt.xlabel('θ')
plt.ylabel('P(θ)')
plt.title('2PL-ICC')
plt.grid(True)
plt.show()




"""
[パラメータリカバリ初期実験]
1: 真のパラメータを設定してデータ生成
"""

np.random.seed(42)

# 設定
N = 500   # 受験者数
J = 20    # 項目数

# 真のパラメータ（これが「答え」）
true_a = np.random.uniform(0.5, 2.5, J)   # 識別力
true_b = np.random.uniform(-2.0, 2.0, J)  # 困難度
true_theta = np.random.normal(0, 1, N)    # 能力値

# 正答確率を計算してデータ生成
def irt_2pl(theta, a, b):
    return 1 / (1 + np.exp(-a * (theta - b)))

# 反応行列 (N x J)
P = irt_2pl(true_theta[:, None], true_a[None, :], true_b[None, :])
X = (np.random.uniform(size=(N, J)) < P).astype(int)

"""
2: パラメータを推定(周辺最尤法)
"""

def marginal_log_likelihood(params, X, n_quad=21):
    J = X.shape[1]
    a = np.exp(params[:J])   # 正値制約
    b = params[J:]

    # Gauss-Hermite 求積
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    theta_q = nodes * np.sqrt(2)
    w_q = weights / np.sqrt(np.pi)

    log_L = 0
    for i in range(X.shape[0]):
        p = irt_2pl(theta_q[:, None], a[None, :], b[None, :])
        lik = np.prod(p ** X[i] * (1 - p) ** (1 - X[i]), axis=1)
        log_L += np.log(np.dot(w_q, lik) + 1e-300)
    return -log_L

# 初期値
x0 = np.zeros(2 * J)
result = minimize(marginal_log_likelihood, x0, args=(X,),
                  method='L-BFGS-B', options={'maxiter': 500})

est_a = np.exp(result.x[:J])
est_b = result.x[J:]

"""
3: 真値と推定値を比較(RMSE, Bias, 相関)
"""
def plot_recovery(true_vals, est_vals, label):
    r = np.corrcoef(true_vals, est_vals)[0, 1]
    rmse = np.sqrt(np.mean((true_vals - est_vals) ** 2))
    bias = np.mean(est_vals - true_vals)

    plt.figure(figsize=(5, 5))
    plt.scatter(true_vals, est_vals, alpha=0.7)
    lims = [min(true_vals.min(), est_vals.min()),
            max(true_vals.max(), est_vals.max())]
    plt.plot(lims, lims, 'r--', label='y=x')
    plt.xlabel(f'true_{label}')
    plt.ylabel(f'estimated_{label}')
    plt.title(f'{label}  r={r:.3f}, RMSE={rmse:.3f}, Bias={bias:.3f}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return {'r': r, 'rmse': rmse, 'bias': bias}

res_a = plot_recovery(true_a, est_a, 'a(discrimination)')
res_b = plot_recovery(true_b, est_b, 'b(difficulty)')



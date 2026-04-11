import numpy as np

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

def irt_2pl(theta, a, b):
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

import matplotlib.pyplot as plt

# 単一の能力値
print(irt_2pl(theta=-0.14, a=1.0, b=0.2))  # → 0.3775

# 複数の能力値（配列）
thetas = np.linspace(-4, 4, 200)
p = irt_2pl(thetas, a=1.0, b=0.0)

# ICCを描画
plt.plot(thetas, p)
plt.axhline(0.5, color='gray', linestyle='--')
plt.xlabel('θ')
plt.ylabel('P(θ)')
plt.title('2PL：ICC')
plt.grid(True)
plt.show()
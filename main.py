import numpy as np

def calculate_mean(data):
    """データの平均を計算する（物理実験データの処理などを想定）"""
    return np.mean(data)

if __name__ == "__main__":
    sample_data = np.array([1.2, 2.5, 3.8, 4.2, 5.1])
    print(f"Mean: {calculate_mean(sample_data)}")
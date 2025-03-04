import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# 1차원 데이터 예제 (여러 개의 가우시안 분포가 합쳐진 형태)
np.random.seed(42)
data1 = np.random.normal(loc=10, scale=2, size=100)  # 평균 10
data2 = np.random.normal(loc=30, scale=3, size=100)  # 평균 30
data3 = np.random.normal(loc=50, scale=2, size=100)  # 평균 50
data4 = np.random.normal(loc=70, scale=4, size=100)  # 평균 70

# 데이터 합치기
X = np.concatenate([data1, data2, data3, data4]).reshape(-1, 1)

# GMM 모델 학습 (4개의 가우시안 분포라고 가정)
gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(X)

# 추정된 평균 출력
means = np.sort(gmm.means_.flatten())  # 정렬하여 보기 쉽게 출력
print("각 가우시안 분포의 평균 (centroids):", means)

# 데이터 분포와 GMM 평균 시각화
plt.hist(X, bins=50, density=True, alpha=0.5, label="Histogram of Data")
plt.vlines(
    means, ymin=0, ymax=0.05, colors="red", linestyles="dashed", label="GMM Centers"
)
plt.legend()
plt.show()

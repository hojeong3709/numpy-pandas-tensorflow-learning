import numpy as np

# 통계함수
data = np.array([1, 2, 3, 4, 5, 6])

print(data.min(), data.mean(), data.max(), data.std())
print(np.sum(data), np.mean(data), np.min(data), np.max(data), np.std(data))
print(np.median(data))
print(np.quantile(data, [0.25, 0.5, 0.75]))
print(np.percentile(data, [25, 50, 75]))

# 주식 - 피보나치값 황금비율(38.2%, 50%, 61.8%)
data = np.array([1450, 2050])
print(np.quantile(data, [0.382, 0.5, 0.612]))

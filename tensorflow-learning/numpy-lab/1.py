import numpy as np

arr = np.array([1, 4, 1])
print(arr)

arr1 = np.arange(1, 4, 1)
print(arr1)

# 복원추출
r = np.random.choice(np.arange(1, 11), 3)
print(r)

# 비복원추출
r = np.random.choice(np.arange(1, 11), 3, replace=False)
print(r)

r = np.ones(shape=(3, 3), dtype=np.float32)
print(r)

# 희소행렬, One Hot Encoding
# 정방행렬
r = np.eye(3)
print(r)

# 중복데이터 제거 ( feature or label 확인 )
r = np.unique([11, 11, 2, 2, 34, 34, 34])
print(r)

# 균등분할
r = np.linspace(10, 20, 3)
print(r)


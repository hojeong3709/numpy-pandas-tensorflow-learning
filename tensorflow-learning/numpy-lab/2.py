import numpy as np

# list to ndarray
data1 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
print(data1)
print(data1.dtype)

data2 = np.int32([1, 2, 3, 4, 5])
print(data2)

data3 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
print(data3)

data4 = np.float32([1, 2, 3, 4, 5])
print(data4)

data5 = np.array([1, 2, 3, 4, 5], dtype=np.str)
print(data5)
print(type(data5))

# 속성
# 1차원 ( 행, ) -> 열 생략
data = np.array([1, 2, 3, 4, 5])
print(data.shape)
print(data.dtype)
print(data.size)

# 2차원 ( 행, 열 )
data = np.array([[1, 2], [3, 4]])
print(data)
print(data.shape)
data = data.astype(np.float32)
data = np.float32(data)
print(data)

# 함수
data6 = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
print(data6)
data6 = data6.reshape(3, 2)
print(data6)
data6 = data6.reshape(-1, 2)
print(data6)

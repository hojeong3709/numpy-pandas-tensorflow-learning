import numpy as np

# ndarray 객체 생성
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
data = np.array([1, 2, 3, 4, 5])
# 1차원 ( 행, ) -> 열 생략
print(data.shape)
print(data.dtype)
print(data.size)

data = np.array([[1, 2], [3, 4]])
# 2차원 ( 행, 열 )
print(data.shape)

print(data)
data = data.astype(np.float32)
print(data)

# 함수
data6 = np.array([1, 2, 3, 4, 5, 6])
print(data6)
data6 = data6.reshape(3, 2)
print(data6)
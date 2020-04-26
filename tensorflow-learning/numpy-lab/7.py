import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

ax = np.array([[1, 2], [3, 4]])
ay = np.array([[5, 6], [7, 8]])

print(ax)
print(ay)
print("-" * 50)

# 수평병합
print(np.hstack((ax, ay)))

# 수직병합
print(np.vstack((ax, ay)))
print("-" * 50)

# CRUD
# axis를 지정하지 않고 append 할 경우 2차원을 1차원으로 풀어서 추가
# 기본값이 axis=None
print(np.append(ax, ay, axis=None))

# 행추가 ( vstack과 동일 )
print(np.append(ax, ay, axis=0))

# 열추가 ( hstack과 동일 )
print(np.append(ax, ay, axis=1))
print("-" * 50)

# 삭제
print(np.delete(arr, 0))
print(np.delete(arr, 0, axis=0))
print(np.delete(arr, 0, axis=1))
print("-" * 50)

# 합
print(arr)
print(np.max(arr, axis=0))
print(np.max(arr, axis=1))

# 평균
print(np.mean(arr, axis=0))
print(np.mean(arr, axis=1))
print("-" * 50)

# 검색
print(arr[:, 0] > 3)
print(arr[arr[:, 0] > 3])
print("-" * 50)

# 값 수정
arr[0] = [3, 20, 23]

# 정렬수행
# 정렬은 기본값이 axis=-1
# axis=-1이므로 정렬이 수행되지 않음
print(np.sort(arr))

# axis=None 은 2차원을 1차원으로 다풀어서 정렬
print(np.sort(arr, axis=None))
print(np.sort(arr, axis=0))
print(np.sort(arr, axis=1))

# 파일 불러오기
b = np.loadtxt("../data/births.txt", delimiter=",", dtype=np.int32)
print(b)

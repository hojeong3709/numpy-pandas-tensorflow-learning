import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(arr)
print(arr.shape)
# print(arr.size)
# print(arr.dtype)
print("-"*50)

print(arr[0, :])
print(arr[-1, :])
print(arr[-1, :].shape)
print("-"*50)

print(arr[:, 0])
print(arr[:, -1])
print(arr[:, -1].shape)
print("-"*50)

arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(arr)
print(arr.shape)
print("-"*50)

print(arr[-1, :])
print(arr[-1, :].shape)
print("-"*50)

print(arr[:, -1])
print(arr[:, -1].shape)



#
# # [행, 열] # 3차원 [명, 행, 열]
# # print(arr[0, 0])
# print(arr[1:])
# print("-"*50)
#
# # 복수개 행선택
# print(arr[[0, 2]])
# print("-"*50)
#
# # 복수개 열선택
# print(arr[1:, 1:])
# print(arr[[0, 2], 1:])
# print(arr[:, :-1])
# print("-"*50)
#
# # 결과 스칼라
# print(arr[2][2])
# print(arr[2][2].shape)
# # 결과 1차원
# print(arr[2])
# print(arr[2].shape)
# # 결과 2차원
# print(arr[[2]])
# print(arr[[2]].shape)
# print("-"*50)
#
# # 결과 1차원
# print(arr[:, -1])
# print(arr[:, -1].shape)
# # 결과 2차원
# print(arr[:, [-1]])
# print(arr[:, [-1]].shape)
# print("-"*50)
#
# # boolean indexing
# print(arr[[True, False, False, True]])
# print("-"*50)
#
# print(arr)
# print(arr[1:3, -1])
# print(arr[1:3, [-1]])
# print("-"*50)
#
# # 행단위 unpacking
# r1, r2, r3, r4 = arr
# print(r1)
# print(r2)
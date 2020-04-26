import numpy as np

data = np.array([1, 2, 3, 4, 5, 6])

# Forward, Backward Indexing
#  0,  1,  2,  3,  4,  5
# -6, -5, -4, -3, -2, -1
print(data[0])
print(data[-1])

# Slicing
print(data[1:4])

# Index group
print(data[[1, 3, 4]])

# Boolean Indexing
# 열 개수만큼 True, False 를 입력해줘야한다.
print(data[[True, True, False, False, True, True]])

# 연산( 산술, 관계, 논리 )
# 산술연산
print(data + 1)
print(data * 2)

data2 = np.array([3, 3, 2, 2, 1, 1])
print(data + data2)

salary = np.array([1000, 2000, 3000, 4000, 5000])

salary = salary - (salary * 0.033)
print(salary)

# 관계연산
# 관계연산 후 결과는 True, False 값을 출력한다.
# np.where의 경우에는 index를 출력하므로 두 경우는 헷갈리지 말기.
print(data > 3)
print(data[data > 3])

# 논리연산
# and ==> &
# or ==> |
# not ==> ~p
# Array에 대한 논리연산은 & | ~ 으로 하므로 연산자 우선순위를 살펴보면 관계연산보다 우선순위가 높다. 괄호필요
print(data == 2)
print(data == 4)
print((data == 2) & (data == 4))
print((data == 2) | (data == 4))
print(~(data == 2))

# and, or 키워드는 Arrary에서는 에러발생
# print((data == 2) and (data == 4))
# print((data == 2) or (data == 4))

# not 키워드는 OK
print(not (data == 2))

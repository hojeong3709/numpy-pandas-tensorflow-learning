import numpy as np

data = np.array([1, 2, 3, 4, 5, 6])
# index
print(data[0])
print(data[-1])
# slicing
print(data[1:4])
# index group
print(data[[1, 3, 4]])

# boolean index
# 개수만큼 True, False를 입력해줘야한다.
print(data[[True, True, False, False, True, True]])

# 연산
print(data + 1)
print(data * 2)

data2 = np.array([3, 3, 2, 2, 1, 1])
print(data + data2)

salary = np.array([1000, 2000, 3000, 4000, 5000])

salary = salary - (salary * 0.033)
print(salary)

# 관계연산
print(data > 3)
print(data[data > 3])

# 논리연산
# and ==> &
# or ==> |
# not ==> ~p
# 비트연산자인 &를 사용하여 논리연산을 하기때문에 &가 관계연산보다 우선순위가 높다. 괄호필요
print(data == 2)
print(data == 4)
print((data == 2) & (data == 4))
print((data == 2) | (data == 4))
print(~(data == 2))

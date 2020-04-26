import numpy as np

# N-dimension array 생성방법 1
arr = np.array([1, 2, 3])
print(arr)

# N-dimension array 생성방법 2
arr = np.arange(1, 10, 1)
print(arr)

# 0~1 사이에 임의의 수 벡터값 5개 선정
r = np.random.rand(5)
print(r)

# 0~1 사이에 임의의 수 매트릭스(3X2) 생성
r = np.random.rand(3, 2)
print(r)

# 1~10 사이에 임의의 수 5개 선정
r = np.random.randint(1, 11, size=5)
print(r)

# 표준정규분포( 평균이 0 이고 편차가1 )에서 임의의 수 5개 선정
r = np.random.randn(5)

# 균등분할 -> 2분위, 3분위, 4분위
r = np.linspace(10, 20, 3)

# 복원추출
r = np.random.choice(np.arange(1, 11), 3)

# 비복원추출
r = np.random.choice(np.arange(1, 11), 3, replace=False)

# 1로 채워진 3X3 정방행렬
r = np.ones(shape=(3, 3), dtype=np.float32)

# 희소행렬, 다중 분류문제에서 One Hot Encoding으로 변환시 사용
# 정방행렬 3X3 -> 대각행렬
r = np.eye(3)

# 중복데이터 제거 ( feature or label 확인 )
r = np.unique([11, 11, 2, 2, 34, 34, 34])

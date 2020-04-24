
import numpy as np

kor = np.array([60, 55, 70, 30, 20, 90])

#1
ans1 = np.max(kor)

ans2 = kor[np.where(kor > 80)]

ans3 = kor[np.where((kor >= 50) & (kor < 80))]

ans4 = np.mean(kor)

ans5 = np.sum(kor)

ans6 = np.delete(kor, np.where(kor <= 40))

ans7 = kor[np.where(kor >= 50)] * 1.1

print(kor)
ans8 = np.sort(kor)[:-6:-1]


# print(ans1)
# print(ans2)
# print(ans3)
# print(ans4)
# print(ans5)
print(ans6)
print(ans7)
print(ans8)

for score in kor:
    print("합격" if score >= 70 else "불합격")
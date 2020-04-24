import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def cost(x, y, w):
    c = 0
    for i in np.arange(len(x)):
        hx = w * x[i]
        c = c + (hx-y[i]) ** 2

    return c/len(x)


x = [1, 2, 3]
y = [1, 2, 3]

print(cost(x, y, -1))
print(cost(x, y, 0))
print(cost(x, y, 1))
print(cost(x, y, 2))
print(cost(x, y, 3))

# 색상, 마커, 선의중요
# plt.plot(x, y, 'ro--')
# plt.show()

for w in np.linspace(-3, 5, 50):
    c = cost(x, y, w)
    print(w, c)
    plt.plot(w, c, "ro")

plt.show()

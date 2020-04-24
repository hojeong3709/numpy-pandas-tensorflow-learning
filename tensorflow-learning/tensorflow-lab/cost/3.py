import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def cost(x, y, w):
    c = 0
    for i in np.arange(len(x)):
        hx = w * x[i]
        # 오차제곱
        c = c + (hx-y[i]) ** 2

    return c/len(x)

def gradient_descent(x, y, w):
    c = 0
    for i in range(len(x)):
        hx = w * x[i]
        # W에 대한 미분, 편미분을 통해 x[i]가 도출
        c = c + (hx-y[i]) * x[i]
    return c/len(x)

def show_gradient():
    x = [1, 2, 3]
    y = [1, 2, 3]
    w = 10
    learning_rate = 0.1
    for i in range(200):
        c = cost(x, y, w)
        g = gradient_descent(x, y, w)
        w = w - learning_rate * g
        if c < 1e-15:
            break
        print(i, c)
    print("w=", w)

show_gradient()
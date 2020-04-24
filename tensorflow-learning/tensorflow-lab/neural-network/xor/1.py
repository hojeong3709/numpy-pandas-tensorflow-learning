import tensorflow as tf
import numpy as np


def common(w1, w2, theta, x1, x2):
    value = w1 * x1 + w2 * x2
    print('value:', value)
    return value > theta


def AND(x1, x2):
    return common(0.5, 0.5, 0.5, x1, x2)


def OR(x1, x2):
    return common(0.5, 0.5, 0.2, x1, x2)


def NAND(x1, x2):
    return common(-0.5, -0.5, -0.7, x1, x2)


def XOR(x1, x2):
    y1 = OR(x1, x2)
    y2 = NAND(x1, x2)
    return AND(y1, y2)


def show_operation(op):
    for x1, x2 in data:
        print(op(x1, x2))


data = [[1, 1], [1, 0], [0, 1], [0, 0]]

show_operation(AND)
show_operation(OR)
show_operation(XOR)
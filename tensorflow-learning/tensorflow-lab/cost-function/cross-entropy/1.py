import numpy as np
import tensorflow as tf

def fn(x):
    print(x/x.sum())

def softmax(x):
    # e = 2.71838
    e = np.exp(x) # 2.71**2.0 2.71**1.0, 2.71**0.1
    print(e)
    print(e / np.sum(e))


a = np.array([2.0, 1.0, 0.1])
fn(a)

softmax(a)
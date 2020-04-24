'''
패션이미지
8. 패션이미지에 따른 상품을 예측하시오(텐서플로우, 케라스)
'''
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.examples.tutorials.mnist import input_data
import warnings

warnings.filterwarnings('ignore')

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape)
print(train_labels.shape)

print(test_images.shape)
print(test_labels.shape)

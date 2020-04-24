import tensorflow as tf

# placeholder와 Variable은 변화하는 값이지만
# placeholder와 달리 variable은 초기값이 있음.
x = tf.placeholder(dtype=tf.float32)
w = tf.Variable([2], dtype=tf.float32)
y = w * x


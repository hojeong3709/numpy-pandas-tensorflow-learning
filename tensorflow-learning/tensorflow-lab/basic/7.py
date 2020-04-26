import tensorflow as tf

# placeholder와 Variable은 변화하는 값으로 공통점이 있지만
# placeholder는 초기값이 없고 Variable은 초기값이 있음.
x = tf.placeholder(dtype=tf.float32)
w = tf.Variable([2], dtype=tf.float32)
y = w * x

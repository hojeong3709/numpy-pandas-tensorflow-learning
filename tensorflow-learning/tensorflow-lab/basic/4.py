import tensorflow as tf

# 산술 관게 논리
a = tf.constant(10)
b = tf.constant(20)

sess = tf.Session()

z1 = tf.greater(a, b)
print(sess.run(z1))

z2 = tf.logical_and([True, False], [True, True])
print(sess.run(z2))

import tensorflow as tf

# Tensorflow 연산시 꼭 알아야 하는점 --.> Element Wise Operation
# Element Wise Operation 이 동작하면 매트릭스 연산시 Broadcasting
a = tf.constant([[1, 2, 3]])
# b = tf.constant(11)
b = tf.constant([11]) # broad casting
# b = tf.constant([[10, 11, 12]])
# b = tf.constant([[10, 11]]) --> Error
# b = tf.constant([[11]]) # broad casting

c = tf.constant([[1, 2, 3], [4, 5, 6 ]])
# d = tf.constant([[1, 2, 3]]) # broad casting
d = tf.constant([[1], [2]])
sess = tf.Session()

z = tf.add(a, b)
print(sess.run(z))

y = tf.add(c, d)
print(sess.run(y))

e = tf.constant([[1, 2], [3, 4]])
f = tf.constant([[5, 6], [7, 8]])

m = tf.matmul(e, f)
print(sess.run(m))


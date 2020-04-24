import tensorflow as tf

# 10행 1열짜리 텐서를 임의의 값을 넣어서 생성
a = tf.random_uniform([10, 1], minval=0, maxval=7, dtype=tf.int32)
sess = tf.Session()
print(sess.run(a))

b = tf.random_uniform([2, 2], -5.0, 5.0)
print(sess.run(b))

c = tf.random_normal([2, 2])
print(sess.run(c))

d = tf.constant([[1], [2]])
e = tf.constant(([3], [4]))
f = tf.concat([d, e], axis=0)

print(sess.run(f))
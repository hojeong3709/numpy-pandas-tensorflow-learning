import tensorflow as tf

input_data = [1, 2, 3, 4, 5]
x = tf.placeholder(dtype=tf.float32)
y = x * 2

sess = tf.Session()
print(sess.run(y, feed_dict={x: input_data}))

a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)
z = tf.multiply(a, b)

print(sess.run(z, feed_dict={a: [[3, 2], [1, 2]], b: [[10], [20]]}))

c = tf.placeholder(dtype=tf.float32, shape=[None, 2])
d = tf.placeholder(dtype=tf.float32, shape=[None, 1])

z = tf.matmul(c, d)

print(sess.run(z, feed_dict={c: [[3, 2], [1, 3], [2, 4]], d: [[3], [1]]}))

import tensorflow as tf

print(tf.__version__)

hello = tf.constant("Hello Tensorflow")
print(hello)

sess = tf.Session()
# bytes
result = sess.run(hello)
print(result)
print(result.decode())

a = tf.constant(10)
b = tf.constant(20)
_a, _b = sess.run([a, b])

print(_a, _b)
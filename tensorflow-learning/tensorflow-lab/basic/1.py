import tensorflow as tf

print(tf.__version__)

hello = tf.constant("Hello Tensorflow")
print(hello)

sess = tf.Session()
# 실행 후 결과값이 bytes 타입이므로 decode 필요
result = sess.run(hello)
print(result)
print(result.decode())

a = tf.constant(10)
b = tf.constant(20)
_a, _b = sess.run([a, b])

print(_a, _b)

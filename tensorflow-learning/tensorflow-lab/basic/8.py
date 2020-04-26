import tensorflow as tf

# 난수 생성방법
# 1. random_normal --> 정규분포
# 2. random_uniform --> 균일분포
# 3. truncated_normal --> 절단정규분포

# 참고자료
# https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/api_docs/python/constant_op.html

# 10행 1열짜리 텐서를 임의의 값을 넣어서 생성
a = tf.random_uniform([10, 1], minval=0, maxval=7, dtype=tf.int32)
sess = tf.Session()
print(sess.run(a))
print("="*50)

b = tf.random_uniform([2, 2], -5.0, 5.0)
print(sess.run(b))
print("="*50)

c = tf.random_normal([2, 2])
print(sess.run(c))
print("="*50)

d = tf.constant([[1], [2]])
e = tf.constant(([3], [4]))
f = tf.concat([d, e], axis=0)
print(sess.run(f))
print("="*50)

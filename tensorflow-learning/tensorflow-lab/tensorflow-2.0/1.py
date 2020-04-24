import tensorflow as tf

print(tf.__version__)

a = tf.constant(5)
b = tf.constant(3)

c = a * b

# 즉시실행
print(a.numpy())
print(b.numpy())
print(c.numpy())

# 세션이 사라져서 tf.global_initializer()를 실행해야 값이 들어갔던 부분이 사라짐.
x = [1.0, 0.6, -1.8]
w = tf.Variable(2.0)
b = tf.Variable(0.7)
z = w * x + b
print(z.numpy())

x = tf.Variable(tf.constant(3.0))

# 미분한 내용을 저장 --> GradientTape
with tf.GradientTape() as tape:
    y = tf.multiply(5, x)
    gradient = tape.gradient(y, x)
    print(gradient.numpy())

x = tf.Variable(tf.constant(3.0))

with tf.GradientTape() as tape:
    y = tf.multiply(x, x)
    gradient = tape.gradient(y, x)
    print(gradient.numpy())




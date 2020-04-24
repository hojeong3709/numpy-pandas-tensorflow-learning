import tensorflow as tf


def hxFn(x):
    return w * x


x = [1, 2, 3]
y = [1, 2, 3]

learning_rate = 0.1
w = tf.Variable(10.)
b = tf.Variable(10.)

for n in range(100):
    with tf.GradientTape() as tape:
        # hx = w * x # tf.multiply( w, x)
        hx = hxFn(x)
        cost = tf.reduce_mean(tf.square(hx - y))
        dw = tape.gradient(cost, w)
    w.assign_sub(learning_rate * dw)
    print(n, cost.numpy())

print(w.numpy())

# 예측값 추론시 Placeholder가 따로 없어서 Session.run()이 필요없음.
print(hxFn(10).numpy())

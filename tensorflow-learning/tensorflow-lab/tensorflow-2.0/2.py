import tensorflow as tf

x = [1, 2, 3]
y = [1, 2, 3]

learning_rate = 0.1
w = tf.Variable(10.)
b = tf.Variable(10.)

for i in range(10):
    # GradientDescentOptimizer -> GradientTape
    with tf.GradientTape() as tape:
        hx = w * x
        cost = tf.reduce_mean(tf.square(hx - y))
        # cost-function function을 w로 미분
        dw = tape.gradient(cost, w)
    # w = w - 미분한값
    w.assign_sub(learning_rate * dw)
    print(i, cost.numpy())

print(w.numpy())


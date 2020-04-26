import numpy as np
import tensorflow as tf
from sklearn import datasets

'''
x_data = data[:, :-1]
y_data = data[:, -1:]

print(x_data.shape)
print(y_data.shape)

IO = Dense(units=1, input_shape=[8], activation="sigmoid")
model = Sequential([IO])
model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])

history = model.fit(x_data, y_data, epochs=100)

print(model.predict(x_data))
print(model.predict_classes(x_data))
print(history.history['acc'][-1])
'''

iris = datasets.load_iris()
# print(iris.keys())
# print(iris["target"])
# print(iris["data"])

# e = np.eye(3)
# print(e[[0, 2, 2]])
# # One Hot Encoding
# print(e[iris["target"]])

x_data = iris["data"]
y_data = iris["target"]

print(x_data.shape)
print(y_data.shape)

y_one_hot = np.eye(3)[y_data]
# print(y_one_hot)

W = tf.Variable(tf.random_uniform([4, 3]))
b = tf.Variable(tf.random_uniform([3]))

X = tf.placeholder(dtype=tf.float32, shape=[None, 4])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 3])

z = tf.matmul(X, W) + b
hx = tf.nn.softmax(z)
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    sess.run(train, feed_dict={X: x_data})
    if not i % 100:
        print(i, sess.run(cost, feed_dict={X: x_data}))

print(sess.run(W))
print(sess.run(b))

predict = sess.run(z, feed_dict={X: [[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.7, 1.5], [7.7, 3.8, 6.7, 2.2]]})

# species = ["setosa", "versicolor", "virginica"]
# print(species[h[0].argmax()])
# print(species[h[1].argmax()])
# print(species[h[2].argmax()])

# 예측
print(iris["target_names"])
print(predict.argmax(axis=1))
for i in predict.argmax(axis=1):
    print(iris["target_names"][i])

print(predict.argmax(axis=1))

# 정확도
hx = sess.run(hx, feed_dict={X: x_data})
print(np.equal(iris["target"], hx.argmax(axis=1)).mean())



# IO = Dense(units=3, input_shape=[4], activation='cross-entropy')
# model = Sequential([IO])
# model.compile(loss='categorical_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])
# history = model.fit(x, y, epochs=1000)
# h = model.predict(np.array([[5.1,3.5,1.4,0.2], [6.7,3.1,4.7,1.5],[7.7,3.8,6.7,2.2]]))
# print(iris['target_names'][h.argmax(axis=1)])
# print(history.history['acc'][-1])
















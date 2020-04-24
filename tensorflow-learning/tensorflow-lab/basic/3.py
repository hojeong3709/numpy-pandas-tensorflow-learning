import tensorflow as tf

a = tf.constant(10, name="a")
print(tf.get_default_graph().get_operations())

b = tf.constant(10, name="b")
c = tf.constant(10, name="c")
print(tf.get_default_graph().get_operations())

# node 정보확인
print(a.op.node_def)

# tf.multiply(a,b)
d = a * b
print(tf.get_default_graph().get_operations())
print(d.op.node_def)

e = d + c
print(e.op.node_def)

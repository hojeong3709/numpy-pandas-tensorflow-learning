# 주사위 2개 10번 굴리기를 시행했을때
# 10X3 의 matrix 텐서를 생성하시오
# 1열과 2열은 주사위를 시행했을 때 값이며
# 3열은 두개열의 합으로 구성된다.

import tensorflow as tf
dice1 = tf.random_uniform([10, 1], minval=1, maxval=6, dtype=tf.int32)
dice2 = tf.random_uniform([10, 1], minval=1, maxval=6, dtype=tf.int32)
dice_sum = tf.add(dice1, dice2)

concat = tf.concat([dice1, dice2, dice_sum], axis=1)

sess = tf.Session()
print(sess.run(concat))
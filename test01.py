import tensorflow as tf

m1 = tf.constant([[3,3]])
m2 = tf.constant([[3],[3]])
product = tf.matmul(m1, m2)

with tf.Session() as sess:
    resualt = sess.run(product)
    print(resualt)
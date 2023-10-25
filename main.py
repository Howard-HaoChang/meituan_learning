import tensorflow as tf

ids = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
c = tf.concat(ids,1)

with tf.Session() as sess:
    print(sess.run(c))
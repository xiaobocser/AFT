import tensorflow as tf
import numpy as np



labels = np.array([[2,1,0,3,0]],dtype=np.int64).T
rec = tf.identity(labels)

rel = tf.constant(np.array([[0.1, 0.2, 0.6, 0.1],
                   [0.8, 0.06, 0.1, 0.04],
                   [0.3, 0.4, 0.1, 0.2],
                   [0.6, 0.25, 0.1, 0.05],
                   [0.15, 0.2, 0.6, 0.05],
                   ]).astype(np.float32))
# precision, _ = tf.metrics.precision_at_k(rec, rel, 3)

sess = tf.Session()
sess.run(tf.local_variables_initializer())

rel /= 5

r = sess.run(rel)
print(r)
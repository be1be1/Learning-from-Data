import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

x = np.array([[1.0, 2.0],[3.0, 4.0]])
print(x)
y = tf.reduce_mean(x,1)
a = tf.Print(y, [y], message='It is:')
b = tf.add(a, a).eval()
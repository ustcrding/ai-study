import numpy as np
import tensorflow as tf
import pandas as pd

datas = pd.read_excel("./数据源.xls", sheet_name="Sheet2")
datas = datas.reindex(np.random.permutation(datas.index))

batch_size = 7
learning_rate = 0.001
steps = 5000
dataset_size = datas.shape[0]
print(dataset_size)

w = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 3), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

y = tf.matmul(x, w)
loss_function = tf.matmul(tf.transpose(y_ - y), (y_ - y))

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

X = datas[["产品质量", "产品价格", "产品形象"]]
Y = datas[["用户满意度"]]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for i in range(steps):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        if i % 1000 == 0:
            total_loss = sess.run(loss_function, feed_dict={x: X, y_: Y})
            print("After {} training steps,loss is {}".format(i, total_loss))

    print(sess.run(w))

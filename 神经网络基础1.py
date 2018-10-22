import tensorflow as tf
# from numpy.random import RandomState
from sklearn import datasets
import numpy as np


def dataset_process():
    # rdm = RandomState(1)
    # dataset_size = 128
    # X = rdm.rand(dataset_size, 2)
    # Y = [[int(x1+x2<1)] for (x1, x2) in X]
    dataset = datasets.load_breast_cancer()
    dataset_x = dataset.data
    dataset_y = dataset.target

    np.random.seed(0)
    # 利用permutation排列组合，将数据集拆分为训练集和测试集
    indices = np.random.permutation(len(dataset_x))

    train_x = dataset_x[indices[:-50]]
    train_y = dataset_y[indices[:-50]]
    test_x = dataset_x[indices[-50:]]
    test_y = dataset_y[indices[-50:]]

    return train_x, train_y, test_x, test_y


def train(train_x, train_y):
    dataset_size = len(train_x)
    batch_size = 8

    w1 = tf.Variable(tf.random_normal([30, 15], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([15, 5], stddev=1, seed=1))
    w3 = tf.Variable(tf.random_normal([5, 1], stddev=1, seed=1))

    x = tf.placeholder(tf.float32, shape=(None, 30), name='x-input')
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

    a = tf.matmul(x, w1)
    b = tf.matmul(a, w2)
    y = tf.matmul(b, w3)
    y = tf.sigmoid(y)

    cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) +
                                    (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        steps = 5000
        for i in range(steps):
            start = (i*batch_size) % dataset_size
            end = min(start+batch_size, dataset_size)

            sess.run(train_step, feed_dict={x: train_x[start:end], y_: np.array(train_y[start:end]).reshape(end-start, 1)})
            if i % 1000 == 0:
                total_cross_entropy = sess.run(cross_entropy, feed_dict={x: train_x, y_: train_y.reshape(len(train_y), 1)})
                print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = dataset_process()
    print(train_x)
    print(train_y)
    train(train_x, train_y)

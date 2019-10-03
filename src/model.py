#
#
# Codebase copied from https://towardsdatascience.com/learn-word2vec-by-implementing-it-in-tensorflow-45641adaf2ac
#
#

import tensorflow as tf

class model:
    def __init__(self, vocab_size, embed_dim):
        self.x = tf.placeholder(tf.float32, shape=(None, vocab_size))
        self.y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

        self.vocab_size = vocab_size

        self.w1 = tf.Variable(tf.random_normal([vocab_size, embed_dim]))
        self.b1 = tf.Variable(tf.random_normal([embed_dim]))
        self.embedding = tf.add(tf.matmul(self.x, self.w1), self.b1)

        self.w2 = tf.Variable(tf.random_normal([embed_dim, vocab_size]))
        self.b2 = tf.Variable(tf.random_normal([vocab_size]))
        self.prediction = tf.nn.softmax(tf.add(tf.matmul(self.embedding, self.w2), self.b2))

    def train(lr, n_iter, x_train, y_label):
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy_loss)

        for _ in range(n_iter):
            sess.run(train_step, feed_dict={x: x_train, y: y_label})
            print(sess.run(cross_entropy_loss, feed_dict={x: x_train, y: y_label}))

    def get_embedding(vector):
        return tf.add(tf.matmul(vector, self.w1), self.b1)

    def get_embedding_matrix():
        return self.w1, self.b1

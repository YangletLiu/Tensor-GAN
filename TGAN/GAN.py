# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2018/10/22 10:41


import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import transform


learning_rate = 1e-3
LAMBDA = 10


def lrelu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def relu(x):
    return tf.nn.relu(x)


def elu(x):
    return tf.nn.elu(x)


def xavier_init(size):
    input_dim = size[0]
    stddev = 1. / tf.sqrt(input_dim / 2.)
    return tf.random_normal(shape=size, stddev=stddev)


def he_init(size, stride):
    input_dim = size[2]
    output_dim = size[3]
    filter_size = size[0]

    fan_in = input_dim * filter_size ** 2
    fan_out = output_dim * filter_size ** 2 / (stride ** 2)
    stddev = tf.sqrt(4. / (fan_in + fan_out))
    minval = -stddev * np.sqrt(3)
    maxval = stddev * np.sqrt(3)
    return tf.random_uniform(shape=size, minval=minval, maxval=maxval)


class Network(object):
    def __init__(self):
        self.layer_num = 0
        self.weights = []
        self.biases = []

    def conv2d(self, input, input_dim, output_dim, filter_size, stride, padding='SAME'):
        with tf.variable_scope('conv' + str(self.layer_num)):
            init_w = he_init([filter_size, filter_size, input_dim, output_dim], stride)
            weight = tf.get_variable(
                'weight',
                initializer=init_w
            )

            init_b = tf.zeros([output_dim])
            bias = tf.get_variable(
                'bias',
                initializer=init_b
            )

            output = tf.add(tf.nn.conv2d(
                input,
                weight,
                strides=[1, stride, stride, 1],
                padding=padding
            ), bias)

            self.layer_num += 1
            self.weights.append(weight)
            self.biases.append(bias)

        return output

    def deconv2d(self, input, output_dim, filter_size, stride, padding='SAME'):
        with tf.variable_scope('deconv' + str(self.layer_num)):
            input_shape = input.get_shape().as_list()
            init_w = he_init([filter_size, filter_size, output_dim, input_shape[3]], stride)
            weight = tf.get_variable(
                'weight',
                initializer=init_w
            )

            init_b = tf.zeros([output_dim])
            bias = tf.get_variable(
                'bias',
                initializer=init_b
            )

            output = tf.add(tf.nn.conv2d_transpose(
                value=input,
                filter=weight,
                output_shape=[
                    tf.shape(input)[0],
                    input_shape[1] * stride,
                    input_shape[2] * stride,
                    output_dim
                ],
                strides=[1, stride, stride, 1],
                padding=padding
            ), bias)
            output = tf.reshape(output,
                                [tf.shape(input)[0], input_shape[1] * stride, input_shape[2] * stride, output_dim])

            self.layer_num += 1
            self.weights.append(weight)
            self.biases.append(bias)

        return output

    def batch_norm(self, input, scale=False):
        ''' batch normalization
        ArXiv 1502.03167v3 '''
        with tf.variable_scope('batch_norm' + str(self.layer_num)):
            output = tf.contrib.layers.batch_norm(input, scale=scale)
            self.layer_num += 1

        return output

    def dense(self, input, output_dim):
        with tf.variable_scope('dense' + str(self.layer_num)):
            input_dim = input.get_shape().as_list()[1]

            init_w = xavier_init([input_dim, output_dim])
            weight = tf.get_variable('weight', initializer=init_w)

            init_b = tf.zeros([output_dim])
            bias = tf.get_variable('bias', initializer=init_b)

            output = tf.add(tf.matmul(input, weight), bias)

            self.layer_num += 1
            self.weights.append(weight)
            self.biases.append(bias)

        return output

    def residual_block(self, input, output_dim, filter_size, n_blocks=5):
        output = input
        with tf.variable_scope('residual_block'):
            for i in range(n_blocks):
                bypass = output
                output = self.deconv2d(output, output_dim, filter_size, 1)
                output = self.batch_norm(output)
                output = tf.nn.relu(output)

                output = self.deconv2d(output, output_dim, filter_size, 1)
                output = self.batch_norm(output)
                output = tf.add(output, bypass)

        return output

    def pixel_shuffle(self, x, r, n_split):
        def PS(x, r):
            bs, a, b, c = x.get_shape().as_list()
            bs = tf.shape(x)[0]
            x = tf.reshape(x, (bs, a, b, r, r))
            x = tf.transpose(x, (0, 1, 2, 4, 3))
            x = tf.split(x, a, 1)
            x = tf.concat([tf.squeeze(x_, axis=1) for x_ in x], 2)
            x = tf.split(x, b, 1)
            x = tf.concat([tf.squeeze(x_, axis=1) for x_ in x], 2)
            return tf.reshape(x, (bs, a * r, b * r, 1))

        xc = tf.split(x, n_split, 3)
        xc = tf.concat([PS(x_, r) for x_ in xc], 3)
        return xc


class GAN(object):
    def __init__(self, img_shape, latent_dim):
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        # self.learning_rate = learning_rate
        # self.vgg = VGG19(None, None, None)

        self.G_params = []
        self.D_params = []

        self.y = tf.placeholder(
            tf.float32,
            [None, self.img_shape[0]*2, self.img_shape[1]*2, self.img_shape[2]],
            name='x'
        )
        self.z = tf.placeholder(tf.float32, [None, self.latent_dim], name='z')
        self.x = self.downscale(self.y, 2)

        with tf.variable_scope('generator'):
            self.g = self.generator(self.z)
        with tf.variable_scope('discriminator') as scope:
            self.D_real = self.discriminator(self.x)
            scope.reuse_variables()
            self.D_fake = self.discriminator(self.g)

        disc_loss = -tf.reduce_mean(self.D_real) + tf.reduce_mean(self.D_fake)
        gen_loss = -tf.reduce_mean(self.D_fake)

        alpha = tf.random_uniform(
            # shape=[self.batch_size, 1],
            shape=(tf.shape(self.y)[0], 1),
            minval=0.,
            maxval=1.
        )

        x_ = tf.reshape(self.x, [-1, np.prod(self.img_shape)])
        g_ = tf.reshape(self.g, [-1, np.prod(self.img_shape)])

        differences = x_ - g_
        interpolates = x_ + alpha * differences
        interpolates = tf.reshape(interpolates, (-1, self.img_shape[0], self.img_shape[1], self.img_shape[2]))
        gradients = tf.gradients(self.discriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        self.D_loss = disc_loss + LAMBDA * gradient_penalty
        # self.G_loss = content_loss + self.SIGMA * gen_loss
        self.G_loss = gen_loss

        self.D_opt = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=0.5,
            beta2=0.9
        ).minimize(self.D_loss, var_list=self.D_params)
        self.G_opt = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=0.5,
            beta2=0.9
        ).minimize(self.G_loss, var_list=self.G_params)

    def generator(self, z):
        G = Network()
        # Network.deconv2d(input, input_shape, output_dim, filter_size, stride)
        # h = G.dense(z, np.prod((self.img_shape[0], self.img_shape[1], 64)))
        h = lrelu(G.dense(z, np.prod((4, 4, 256))))
        # h = tf.reshape(h, (tf.shape(h)[0], self.img_shape[0], self.img_shape[1], 64))
        h = tf.reshape(h, (tf.shape(h)[0], 4, 4, 256))
        h = lrelu(G.deconv2d(h, 128, 5, 2))
        h = h[:,:7,:7,:]
        h = lrelu(G.deconv2d(h, 64, 5, 2))

        # h = G.residual_block(h, 64, 3, 2)

        h = G.deconv2d(h, self.img_shape[2], 3, 1)
        h = tf.nn.sigmoid(h)

        self.G_params = G.weights + G.biases

        return h

    def discriminator(self, x):
        D = Network()
        # Network.conv2d(input, output_dim, filter_size, stride, padding='SAME')
        h = D.conv2d(x, self.img_shape[2], 32, 5, 2)
        h = lrelu(h)

        # h = D.conv2d(h, 64, 64, 3, 1)
        # h = lrelu(h)
        # h = D.batch_norm(h)

        map_nums = [32, 64, 128]

        for i in range(len(map_nums) - 1):
            h = D.conv2d(h, map_nums[i], map_nums[i + 1], 5, 2)
            h = lrelu(h)
            h = D.batch_norm(h)

        h_shape = h.get_shape().as_list()
        h = tf.reshape(h, [-1, h_shape[1] * h_shape[2] * h_shape[3]])
        h = D.dense(h, 1024)
        h = lrelu(h)

        h = D.dense(h, 1)

        self.D_params = D.weights + D.biases

        return h

    def downscale(self, x, K):
        mat = np.zeros([K, K, self.img_shape[2], self.img_shape[2]])
        for i in range(self.img_shape[2]):
            mat[:, :, i, i] = 1.0 / K ** 2
        filter = tf.constant(mat, dtype=tf.float32)
        return tf.nn.conv2d(x, filter, strides=[1, K, K, 1], padding='SAME')

    def vgg19_loss(self, x, g):
        _, real_phi = self.vgg.build_model(x, tf.constant(False), False)
        _, fake_phi = self.vgg.build_model(g, tf.constant(False), True)

        loss = None
        for i in range(len(real_phi)):
            l2_loss = tf.nn.l2_loss(real_phi[i] - fake_phi[i])
            if loss is None:
                loss = l2_loss
            else:
                loss += l2_loss

        return tf.reduce_mean(loss)

    @staticmethod
    def reconstruction_loss(x, g):
        return tf.reduce_sum(tf.square(x - g))


def show_result(xs, zs, gs):
    zs = np.squeeze(zs)
    xs = np.squeeze(xs)
    gs = np.squeeze(gs)
    fig = plt.figure(figsize=(5, 15))

    #graph = gridspec.GridSpec(1, 3)
    #graph.update(wspace=0.5, hspace=0.5)

    ax = fig.add_subplot(131)
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(zs, cmap='Greys_r')

    ax = fig.add_subplot(132)
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(gs, cmap='Greys_r')

    ax = fig.add_subplot(133)
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(xs, cmap='Greys_r')
    plt.savefig('./out/{}.png'.format(str(step).zfill(6)), bbox_inches='tight')


if __name__ == '__main__':
    batch_size = 32
    step_num = 10000
    latent_dim = 128

    from tensorflow.examples.tutorials.mnist import input_data

    data = input_data.read_data_sets("./MNIST_data", one_hot=True)

    g = GAN([14, 14, 1], latent_dim)

    if not os.path.exists('./backup/'):
        os.mkdir('./backup/')
    if not os.path.exists('./out/'):
        os.mkdir('./out/')

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    if tf.train.get_checkpoint_state('./backup/'):
        saver.restore(sess, './backup/')
        print('********Restore the latest trained parameters.********')

    for step in range(step_num):
        for _ in range(5):
            xs, _ = data.train.next_batch(batch_size)
            xs = np.reshape(xs, (-1, 28, 28, 1))
            zs = np.random.uniform(size=[batch_size, latent_dim])
            # xs = np.expand_dims(xs, axis=-1)
            _, dloss = sess.run([g.D_opt, g.D_loss], feed_dict={g.z:zs, g.y:xs})

        zs = np.random.uniform(size=[batch_size, latent_dim])
        xs, _ = data.train.next_batch(batch_size)
        xs = np.reshape(xs, (-1, 28, 28, 1))
        # xs = np.expand_dims(xs, axis=-1)
        _, gloss = sess.run([g.G_opt, g.G_loss], feed_dict={g.z:zs, g.y:xs})

        if step % 100 == 0:
            saver.save(sess, './backup/', write_meta_graph=False)
            zs = np.random.uniform(size=[3, latent_dim])
            gs = sess.run(g.g, feed_dict={g.z:zs})
            show_result(gs[0], gs[1], gs[2])
            print('step: {}, D_loss: {}, G_loss:{}'.format(step, dloss, gloss))




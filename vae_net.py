from utils import *
import tensorflow as tf
import numpy as np
from loadData import DataSet
import random

class vaeNet(object):

    def __init__(self, category='Mnist', vis_step=10, Train_Epochs=200, batch_size=128, z_size=100,
                 lr=0.001, history_dir='./', checkpoint_dir='./', logs_dir='./',
                 recon_dir='./', gen_dir='./'):
        self.test = False

        self.category = category
        self.epoch = Train_Epochs
        self.img_size = 28 if (category == 'Fashion-Mnist' or category == 'Mnist') else 64
        self.batch_size = batch_size
        self.z_size = z_size

        self.vis_step = vis_step

        self.lr = lr
        self.channel = 1 if (category == 'Fashion-Mnist' or category == 'Mnist') else 3

        self.history_dir = history_dir
        self.checkpoint_dir = checkpoint_dir
        self.logs_dir = logs_dir
        self.recon_dir = recon_dir
        self.gen_dir = gen_dir

        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_size], name='latent')

        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_size, self.img_size, self.channel],
                                name='image')

    def build_Model(self):
        self.mu, self.logvar = self.encoder(self.x, reuse=False)
        self.reparms_z = self.reparameterize(self.mu, self.logvar)
        self.recon = self.decoder(self.reparms_z, reuse=False)
        self.gen = self.decoder(self.z, reuse=True)
        """
        Loss and Optimizer
        """
        self.loss = self.loss_func(self.recon, self.x, self.mu, self.logvar)
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        """
        Logs
        """
        tf.summary.scalar('loss', tf.reduce_mean(self.loss))
        self.summary_op = tf.summary.merge_all()



    def encoder(self, x, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse):
            if self.category == 'Fashion-Mnist' or self.category == 'Mnist':
                x = tf.reshape(x, [self.batch_size, -1])

                fc1 = tf.nn.relu(tf.layers.dense(inputs=x, units=500, name='fc1'))

                fc2 = tf.nn.relu(tf.layers.dense(inputs=fc1, units=500, name='fc2'))

                # mu = tf.layers.dense(inputs=fc1, units=self.z_size, name='meanOfz')
                # logvar = tf.layers.dense(inputs=fc1, units=self.z_size, name='varOfz')
                output = tf.layers.dense(inputs=fc2, units=self.z_size*2, name='meanOfz')
                mu = output[:, :self.z_size]
                logvar = 1e-6 + tf.nn.softplus(output[:, self.z_size:])

        return mu, logvar

    def decoder(self, z, reuse=False):
        with tf.variable_scope('decoder', reuse=reuse):
            if self.category == 'Fashion-Mnist' or self.category == 'Mnist':

                fc1 = tf.nn.relu(tf.layers.dense(inputs=z, units=500, name='fc1'))

                fc2 = tf.nn.relu(tf.layers.dense(inputs=fc1, units=500, name='fc2'))

                fc3 = tf.nn.sigmoid(tf.layers.dense(inputs=fc2, units=784, name='fc3'))

        return fc3

    def reparameterize(self, mu, logvar):
        # std = tf.exp(0.5*logvar)
        # eps = tf.random_uniform(shape=tf.shape(std))
        # return mu+eps*std
        return mu + logvar * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    def loss_func(self, recon, x, mu, logvar):
        recon = tf.clip_by_value(recon, 1e-8, 1 - 1e-8)
        x = tf.reshape(x, [self.batch_size, -1])
        marginal_likelihood = tf.reduce_sum(x * tf.log(recon) + (1 - x) * tf.log(1 - recon), 1)
        KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(logvar) - tf.log(1e-8 + tf.square(logvar)) - 1, 1)

        marginal_likelihood = tf.reduce_mean(marginal_likelihood)
        KL_divergence = tf.reduce_mean(KL_divergence)

        ELBO = marginal_likelihood - KL_divergence

        loss = -ELBO


        return loss

    def train(self, sess):
        self.build_Model()

        data = DataSet(img_size=self.img_size, batch_size=self.batch_size, category=self.category)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=10)

        writer = tf.summary.FileWriter(self.logs_dir, sess.graph)

        start = 0
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)

        if latest_checkpoint:
            latest_checkpoint.split('-')
            start = int(latest_checkpoint.split('-')[-1])
            saver.restore(sess, latest_checkpoint)
            print('Loading checkpoint {}.'.format(latest_checkpoint))

        tf.get_default_graph().finalize()

        latent_gen = np.random.normal(size=(len(data), self.z_size))

        for epoch in range(start + 1, self.epoch):
            num_batch = int(len(data) / self.batch_size)
            losses = []
            for step in range(num_batch):
                obs = data.NextBatch(step)

                loss, summary, _ = sess.run([self.loss, self.summary_op, self.opt], feed_dict={self.x: obs})
                losses.append(loss)
                writer.add_summary(summary, global_step=epoch)

            print(epoch, ": Loss : ", np.mean(losses))
            if epoch % self.vis_step == 0:
                self.visualize(saver, sess, epoch, data)

    def visualize(self, saver, sess, epoch, data):
        saver.save(sess, "%s/%s" % (self.checkpoint_dir, 'model.ckpt'), global_step=epoch)
        idx = random.randint(0, int(len(data) / self.batch_size) - 1)

        """
                Recon
        """
        obs = data.NextBatch(idx)
        sys = sess.run(self.recon, feed_dict={self.x: obs})
        sys = np.reshape(sys, [self.batch_size, 28, 28, 1])
        sys = np.array(sys * 255.0, dtype=np.float)
        path = self.recon_dir + 'epoch' + str(epoch) + 'recon.jpg'
        # show_z_and_img(epoch, path, z, sys, self.row, self.col)
        show_in_one(path, sys, column=16, row=8)

        """
        Generation
        """
        # obs = data.NextBatch(idx, test=True)
        z = np.random.normal(size=(self.batch_size, self.z_size))
        # z = sess.run(self.langevin, feed_dict={self.z: z, self.x: obs})
        sys = sess.run(self.gen, feed_dict={self.z: z})
        sys = np.reshape(sys, [self.batch_size, 28, 28, 1])
        sys = np.array(sys * 255.0, dtype=np.float)
        path = self.gen_dir + 'epoch' + str(epoch) + 'gens.jpg'
        show_in_one(path, sys, column=16, row=8)
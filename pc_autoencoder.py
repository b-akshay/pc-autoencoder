import tensorflow as tf
import numpy as np
import time


class PCAE(object):
    def __init__(self, dim_input, dim_hidden, L1_eps=0.0, batch_size=256, 
                 optimizer=tf.train.AdagradOptimizer(0.1), enc_binary=False):
        tf.reset_default_graph()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.L1_eps = L1_eps
        self.enc_binary_tf = tf.Variable(enc_binary, name='enc_binary')
        self.alt_optimizer = optimizer
        self.W_tf = tf.Variable(np.random.randn(self.dim_input, self.dim_hidden), dtype=tf.float32, name='W_tf')
        self.sess = tf.Session()
        
        # Initialize data and weights.
        self.data_tf = tf.placeholder(tf.float32, [self.dim_input, None])
        self.mbatsize_tf = tf.shape(self.data_tf)[1]
        self.encs_tf = tf.Variable(tf.random_normal([dim_hidden, self.batch_size], dtype=tf.float32), name='encs_tf')
        self.projenc = tf.cond(self.enc_binary_tf, 
                               lambda: tf.assign(self.encs_tf, tf.clip_by_value(self.encs_tf, -1.0, 1.0)), 
                               lambda: tf.assign(self.encs_tf, tf.identity(self.encs_tf)))
        # Build computation graph of intermediate quantities.
        self.scores_tf = tf.matmul(self.W_tf, self.encs_tf)    # (V x n) matrix of logits.
        self.potentials_tf = tf.add(tf.nn.softplus(self.scores_tf), tf.nn.softplus(-self.scores_tf))    # Coordinatewise.
        self.B_tf = tf.div(tf.matmul(self.data_tf, self.encs_tf, transpose_b=True), tf.to_float(self.mbatsize_tf))
        self.tot_bias_tf = tf.trace(tf.matmul(self.B_tf, self.W_tf, transpose_a=True))
        avg_pot_tf = tf.reduce_mean(tf.reduce_sum(self.potentials_tf, 0))
        self.mmxloss_tf = tf.subtract(avg_pot_tf, self.tot_bias_tf)    # Slack, unregularized (>= actual loss).
        self.obj_tf = tf.add(self.mmxloss_tf, self.L1_eps*tf.reduce_sum(tf.abs(self.W_tf)))
        # Optimizers for alternating minimization.
        self.train_op_E = self.alt_optimizer.minimize(self.obj_tf, var_list=[self.encs_tf])
        self.train_op_W = self.alt_optimizer.minimize(self.obj_tf, var_list=[self.W_tf])
        # Compute actual reconstruction loss.
        realdata_zo = 0.5*tf.add(tf.ones(tf.shape(self.data_tf)), self.data_tf)
        losses_tf = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores_tf, labels=realdata_zo)
        self.celoss_tf = tf.reduce_mean(tf.reduce_sum(losses_tf, 0))
        # Gradients for diagnostics.
        self.grads_W_tf = tf.gradients(self.mmxloss_tf, self.W_tf)
        self.grads_encs = tf.gradients(self.mmxloss_tf, self.encs_tf)
        self.sess.run(tf.global_variables_initializer())

    def encode(self, data_mbatch, iters_encode=150, display_step=50):
        inittime = time.time()
        slacks_list = []
        losses_list = []
        print '--Encoding phase--'
        for step in xrange(iters_encode):
            step += 1
            _, _, celoss_alg, mmxloss_alg, corrs, encodings, grad_E = self.sess.run(
                [self.projenc, self.train_op_E, self.celoss_tf, self.mmxloss_tf, 
                 self.B_tf, self.encs_tf, self.grads_encs], 
                feed_dict={self.data_tf: data_mbatch.T})
            grad_E_max = np.max(np.abs(grad_E[0]))
            losses_list.append(celoss_alg)
            slacks_list.append(0.5*mmxloss_alg)
            if step % display_step == 0:
                print step, celoss_alg, time.time() - inittime, np.max(np.abs(encodings)), grad_E_max
        self.corrs = corrs
        return (self.sess.run(self.encs_tf, feed_dict={self.data_tf: data_mbatch.T}), 
                corrs, losses_list, slacks_list)
    
    def decode_fit(self, data_mbatch, encodings=None, iters_decode=350, display_step=50):
        inittime = time.time()
        grad_W_max = 10.0
        gradmaxes_W = []
        slacks_list = []
        losses_list = []
        if encodings is None:
            feed_dict = {self.data_tf: data_mbatch.T}
        else:
            feed_dict = {self.data_tf: data_mbatch.T, self.encs_tf: encodings.T}
        print '--Decoding phase--'
        for step in xrange(iters_decode):
        # while grad_W_max > pow(5, -epoch_ctr):
            step += 1
            _, celoss_alg, mmxloss_alg, W_mat, grad_W = self.sess.run(
                [self.train_op_W, self.celoss_tf, self.mmxloss_tf, self.W_tf, self.grads_W_tf], 
                feed_dict=feed_dict)
            grad_W_max = np.max(np.abs(grad_W[0]))
            gradmaxes_W.append(grad_W_max)
            losses_list.append(celoss_alg)
            slacks_list.append(0.5*mmxloss_alg)
            if step % display_step == 0:
                print step, celoss_alg, 0.5*mmxloss_alg, time.time() - inittime, grad_W_max    # np.max(np.abs(W_mat))
        return (losses_list, slacks_list, gradmaxes_W)
        
    def decode(self, encodings):
        return self.sess.run(2.0*tf.sigmoid(self.scores_tf) - 1.0, feed_dict={self.encs_tf: encodings})

    def get_weights(self):
        return self.sess.run(self.W_tf)
    
    def set_weights(self, W_mat):
        self.sess.run(tf.assign(self.W_tf, W_mat))
    
    def get_corrs(self, data_mbatch):
        # Assumes self.encs_tf are encodings of data_mbatch
        corrs, data, encs = self.sess.run(
            [self.B_tf, self.data_tf, self.encs_tf], feed_dict={self.data_tf: data_mbatch.T})
        return corrs, data, encs


"""
Example usage below: autoencoding MNIST.
"""
if __name__ == "__main__":
    import scipy.io
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    digits_train_all = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))).astype('float32') / 255.
    digits_test_all = x_test.reshape((len(x_test), np.prod(x_test.shape[1:]))).astype('float32') / 255.
    train_size = digits_train_all.shape[0]
    test_size = digits_test_all.shape[0]
    digits_train = digits_train_all[0:train_size, :]
    digits_test = digits_test_all[0:test_size, :]
    data_train = zo_to_pm(binarize_stoch(digits_train))     # Binarize stochastically
    data_test = zo_to_pm(binarize_stoch(digits_test))
    data_test = np.random.permutation(data_test) # Shuffle test data

    # Initialize autoencoder
    dim_hidden = 100
    batch_size = 250
    L1_eps = 0.0
    enc_binary = True
    tf.reset_default_graph()
    pc_ae = PCAE(dim_input, dim_hidden, enc_binary=enc_binary, batch_size=batch_size, L1_eps=L1_eps, 
                 optimizer=tf.train.AdagradOptimizer(0.3))
    num_epochs = 60
    losses_list = []
    slacks_list = []
    max_W_list = []
    inittime = time.time()
    for epoch_ctr in range(num_epochs):
        data_mbatch = samp_batch_from_data(data_train, batch_size)
        print 'Epoch: \t ' + str(epoch_ctr)
        # Train more accurately as we get closer to the optimum and W, B get better. This works fine but other settings work too.
        iters_encode = 35*(epoch_ctr + 1)
        iters_decode = 35*(epoch_ctr + 1)
        encs, corrs, _, _ = pc_ae.encode(data_mbatch, iters_encode=iters_encode, display_step=100)
        losslist, slacklist, maxWlist = pc_ae.decode_fit(
            data_mbatch, encodings=encs.T, iters_decode=iters_decode, display_step=100)
        losses_list.extend(losslist)
        slacks_list.extend(slacklist)
        max_W_list.extend(maxWlist)
    print 'Total time taken: \t' + str(time.time() - inittime)

    # Now compute test set encodings.
    encs_list = []
    corrs_list = []
    celosses = []
    inittime = time.time()
    for step in xrange(test_size / batch_size):
        print '-- Epoch %02d --' % (step + 1)
        offset = step * batch_size
        data_mb = data_test[offset:(offset + batch_size)]
        encs, corrs, losses_list, slacks_list = pc_ae.encode(data_mb, iters_encode=1000, display_step=200)
        encs_list.append(encs)
        corrs_list.append(corrs)
        celosses.append(losses_list[-1])
    test_encs = np.concatenate(tuple(encs_list), axis=1)      # Test set encodings
    test_corrs = np.mean(corrs_list, axis=0)                  # B matrix associated with test encodings
    print 'Mean loss: \t' + str(np.mean(celosses))
    print 'Total time taken: \t' + str(time.time() - inittime)
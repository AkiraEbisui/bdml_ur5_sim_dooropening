from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.utils import shuffle
# ROS
import rospy

import os
import logging
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)

obs_dim = rospy.get_param("/ML/obs_dim")
n_act = rospy.get_param("/ML/n_act")
clip_range = rospy.get_param("/ML/clip_range")
epochs = rospy.get_param("/ML/epochs")
policy_lr = rospy.get_param("/ML/policy_lr")
value_lr = rospy.get_param("/ML/value_lr")
hdim = rospy.get_param("/ML/hdim")
max_std = rospy.get_param("/ML/max_std")
seed = rospy.get_param("/ML/seed")
act_step = rospy.get_param("/ML/act_step")
save_step = rospy.get_param("/ML/save_step")

class PPOGAEAgent(object): 
    def __init__(self, obs_dim, n_act, clip_range, epochs, policy_lr, value_lr, hdim, max_std, seed):
#    def __init__(self, obs_dim, n_act, clip_range=0.2, epochs=10, policy_lr=3e-3, value_lr=7e-4, hdim=64, max_std=1.0, seed=0):
        
        self.seed=seed
        
        self.obs_dim = obs_dim
        self.act_dim = n_act
        
        self.clip_range = clip_range
        
        self.epochs = epochs
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.hdim = hdim
        self.max_std = max_std
        
        self._build_graph()
        self._init_session()
        self.counter = 0

        # load the parameters 
        self.saver.restore(self.sess, './results/ppo_with_gae_model-1500')

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn()
            self._value_nn()
            self._logprob()
            self._loss_train_op()
            self._kl_entropy()
            self._cnn_layer()
            self.init = tf.compat.v1.global_variables_initializer()
            self.variables = tf.compat.v1.global_variables()  
            # Create a saver object which will save all the variables
            self.saver = tf.compat.v1.train.Saver(max_to_keep=None)
            
    def _placeholders(self):
        # observations, actions and advantages:
        self.obs_ph = tf.compat.v1.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.act_ph = tf.compat.v1.placeholder(tf.float32, (None, self.act_dim), 'act')
        self.adv_ph = tf.compat.v1.placeholder(tf.float32, (None,), 'adv')
        self.ret_ph = tf.compat.v1.placeholder(tf.float32, (None,), 'ret')

        # learning rate:
        self.policy_lr_ph = tf.compat.v1.placeholder(tf.float32, (), 'policy_lr')
        self.value_lr_ph = tf.compat.v1.placeholder(tf.float32, (), 'value_lr')
        
        # place holder for old parameters
        self.old_std_ph = tf.compat.v1.placeholder(tf.float32, (None, self.act_dim), 'old_std')
        self.old_mean_ph = tf.compat.v1.placeholder(tf.float32, (None, self.act_dim), 'old_means')

        # place holder for CNN
        self.x = tf.compat.v1.placeholder(tf.float32, (28)) # input 4*7    self.x = tf.compat.v1.placeholder(tf.float32, (None, 28))
        self.y = tf.compat.v1.placeholder(tf.float32, (None, 10)) # output self.y = tf.compat.v1.placeholder(tf.float32, (None, 10))
        
    def _policy_nn(self):        
        hid1_size = self.hdim
        hid2_size = self.hdim
        with tf.compat.v1.variable_scope("policy"):
            # TWO HIDDEN LAYERS
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed), name="h1")
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed), name="h2")

            # MEAN FUNCTION
            self.mean = tf.layers.dense(out, self.act_dim,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed), name="mean")
            # UNI-VARIATE
            self.logits_std = tf.compat.v1.get_variable("logits_std",shape=(1,),initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed))
            self.std = self.max_std * tf.ones_like(self.mean) * tf.sigmoid(self.logits_std) # IMPORTANT TRICK

            # SAMPLE OPERATION
            self.sample_action = self.mean + tf.random.normal(tf.shape(self.mean),seed=self.seed) * self.std
    
    def _value_nn(self):
        hid1_size = self.hdim 
        hid2_size = self.hdim
        with tf.compat.v1.variable_scope("value"):
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name="h1")
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name="h2")
            value = tf.layers.dense(out, 1,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='output')
            self.value = tf.squeeze(value)
            
    def _logprob(self):
        # PROBABILITY WITH TRAINING PARAMETER
        y = self.act_ph 
        mu = self.mean
        sigma = self.std
        
        self.logp = tf.reduce_sum(-0.5 * tf.square((y - mu) / sigma) - tf.math.log(sigma) - 0.5 * np.log(2. * np.pi), axis=1)

        # PROBABILITY WITH OLD (PREVIOUS) PARAMETER
        old_mu_ph = self.old_mean_ph
        old_sigma_ph = self.old_std_ph
                
        self.logp_old = tf.reduce_sum(-0.5 * tf.square((y - old_mu_ph) / old_sigma_ph) - tf.math.log(old_sigma_ph) - 0.5 * np.log(2. * np.pi), axis=1)
        
    def _kl_entropy(self):
        delta = 1e-3 # add for preventing from "nan"
        mean, std = self.mean, self.std
        old_mean, old_std = self.old_mean_ph, self.old_std_ph
 
        log_std_old = tf.math.log(tf.clip_by_value(old_std, 1e-10, 1.0))
        log_std_new = tf.math.log(tf.clip_by_value(std, 1e-10, 1.0))
        frac_std_old_new = old_std / std

        # KL DIVERGENCE BETWEEN TWO GAUSSIAN
        kl = tf.reduce_sum(log_std_new - log_std_old + 0.5 * tf.square(frac_std_old_new) + 0.5 * tf.square((mean - old_mean) / std) - 0.5, axis=1)
        self.kl = tf.reduce_mean(kl) + 1e-7
        
        # ENTROPY OF GAUSSIAN
        entropy = tf.reduce_sum(log_std_new + 0.5 + 0.5 * np.log(2 * np.pi + delta), axis=1)
        self.entropy = tf.reduce_mean(entropy) + 1e-7
            
    def _loss_train_op(self):
        
        # REINFORCE OBJECTIVE
        ratio = tf.exp(self.logp - self.logp_old)
        cliped_ratio = tf.clip_by_value(ratio, clip_value_min = 1 - self.clip_range, clip_value_max = 1 + self.clip_range)
        self.policy_loss = -tf.reduce_mean(tf.minimum(self.adv_ph * ratio, self.adv_ph * cliped_ratio)) + 1e-7 # Compute 1_Policy  #add 1e-10 for preventing from "nan"
        
        # POLICY OPTIMIZER
        self.pol_var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="policy")
        optimizer = tf.compat.v1.train.AdamOptimizer(self.policy_lr_ph)
        self.train_policy = optimizer.minimize(self.policy_loss, var_list=self.pol_var_list)
            
        # L2 LOSS
        self.value_loss = tf.reduce_mean(0.5 * tf.square(self.value - self.ret_ph))
            
        # VALUE OPTIMIZER 
        self.val_var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="value")
        optimizer = tf.compat.v1.train.AdamOptimizer(self.value_lr_ph)
        self.train_value = optimizer.minimize(self.value_loss, var_list=self.val_var_list)
        
    def _init_session(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config, graph=self.g)
        self.sess.run(self.init)
        
    
    def get_value(self, obs):
        feed_dict = {self.obs_ph: obs}
        value = self.sess.run(self.value, feed_dict=feed_dict)
        return value
    
    def get_action(self, obs): # SAMPLE FROM POLICY
        feed_dict = {self.obs_ph: obs}
        sampled_action = self.sess.run(self.sample_action, feed_dict=feed_dict)
        return sampled_action[0] / act_step
# /10:   [action]', array([-0.01992016,  0.10500866,  0.00853405,  0.02726892, -0.12092558, 0.02108609])
# /500:   [action]', array([0.0010571,  -0.00022959,  -0.00092911,  -0.00055518, -0.0012934, -0.001190])  -> Maxdelta force(6.5, 1.8, 16.7), torque(0.65, 4.4, 0.17)
# /1000: [action]', array([-1.9920163e-04,  1.0500867e-03,  8.5340682e-05,  2.7268915e-04, -1.2092555e-03,  2.1086067e-04])
    
    def control(self, obs): # COMPUTE MEAN
        feed_dict = {self.obs_ph: obs}
        best_action = self.sess.run(self.mean, feed_dict=feed_dict)
        return best_action

    def _cnn_layer(self):
        img_hight = 7
        img_width = 4
        color_channel = 1
        img = tf.reshape(self.x, [-1, img_hight, img_width, color_channel]) # batch_size(-1 means auto), hight, width, color channel

        # convolution layer1
        with tf.name_scope('conv1'):
            conv1_f = tf.Variable(tf.truncated_normal([2, 2, 1, 32], stddev=0.1))      # filter hight, width, channel, number of filter(output dimension)
            conv1_c = tf.nn.conv2d(img, conv1_f, strides=[1, 1, 1, 1], padding='SAME') # strides: batch direct,  height direct, width direct, channel direct
            conv1_b = tf.Variable(tf.constant(0.1, shape=[32]))                        # shape=[output dimension]
            conv1_o = tf.nn.relu(conv1_c + conv1_b)                                    # 4x7 padding-> 6x9 conv-> 5x8 (5x8x32=1280)

        # pool layer1
        with tf.name_scope('pool1'):
            pool1_o = tf.nn.max_pool(conv1_o, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME') # ksize: batch direct,  height direct, width direct, channel direct
                                                                                                        # 5x8 padding-> 7x10 pool-> 6x9 (6x9x32=1792)
        # convolution layer2
        with tf.name_scope('conv2'):
            conv2_f = tf.Variable(tf.truncated_normal([2, 2, 32, 64], stddev=0.1))
            conv2_c = tf.nn.conv2d(pool1_o, conv2_f, strides=[1, 1, 1, 1], padding='SAME')
            conv2_b = tf.Variable(tf.constant(0.1, shape=[64]))
            conv2_o = tf.nn.relu(conv2_c + conv2_b)                                    # 6x9 padding-> 8x11 conv-> 7x10 (7x10x64=4480)
            #print("conv2_o", conv2_o) # <tf.Tensor 'conv2/Relu:0' shape=(1, 7, 4, 64) dtype=float32>

        # pool layer 2
        with tf.name_scope('pool2'):
            pool2_o = tf.nn.max_pool(conv2_o, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')  # 7x10 padding-> 9x12 pool-> 8x11 (8x11x64=5632)
            #print("pool2_o", pool2_o) # <tf.Tensor 'pool2/MaxPool:0' shape=(1, 7, 4, 64) dtype=float32>

        # flatten layer
        with tf.name_scope('flatten'):
            flatten_o = tf.reshape(pool2_o, [-1, 7 * 4 * 64]) # 8 * 11 * 64 = 5632 

        # fully connected layer
        with tf.name_scope('fully_connected'):
            fully_connected_w = tf.Variable(tf.truncated_normal([7 * 4 * 64, 1024], stddev=0.1))       # filter hight, width, channel, number of filter(output dimension)
            fully_connected_b = tf.Variable(tf.constant(0.1, shape=[1024]))                             # shape=[output dimension]
            fully_connected_o = tf.nn.relu(tf.matmul(flatten_o, fully_connected_w) + fully_connected_b) 

        # output layer
        with tf.name_scope('output'):
            output_w = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
            output_b = tf.Variable(tf.constant(0.1, shape=[10]))
            self.output_o = tf.nn.softmax(tf.matmul(fully_connected_o, output_w) + output_b) # output_o = self.y

    def update_cnn(self, delta_image):
        image_cnn = self.sess.run(self.output_o, {self.x: delta_image})
        return image_cnn

    def update(self, observes, actions, advantages, returns, batch_size): # TRAIN POLICY
        
        num_batches = max(observes.shape[0] // batch_size, 1) # compute the number of batches
        batch_size = observes.shape[0] // num_batches         # recalculate batch_size again
        
        old_means_np, old_std_np = self.sess.run([self.mean, self.std],{self.obs_ph: observes}) # COMPUTE OLD PARAMTER (self.mean = mean_action, self.std = a random value for trick?)

        for e in range(self.epochs): # repeat for self.epochs times
            observes, actions, advantages, returns, old_means_np, old_std_np = shuffle(observes, actions, advantages, returns, old_means_np, old_std_np, random_state=self.seed)
            for j in range(num_batches): # pick up the data set and compute "policy" and "value" for number of batches
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: observes[start:end,:],
                     self.act_ph: actions[start:end],
                     self.adv_ph: advantages[start:end],
                     self.ret_ph: returns[start:end],
                     self.old_std_ph: old_std_np[start:end,:],
                     self.old_mean_ph: old_means_np[start:end,:],
                     self.policy_lr_ph: self.policy_lr,
                     self.value_lr_ph: self.value_lr}        
                self.sess.run([self.train_policy, self.train_value], feed_dict) # compute "policy" and "value" by minimize the policy_loss and value_loss using Adamoptimizer
            
        feed_dict = {self.obs_ph: observes,
             self.act_ph: actions,
             self.adv_ph: advantages,
             self.ret_ph: returns,
             self.old_std_ph: old_std_np,
             self.old_mean_ph: old_means_np,
             self.policy_lr_ph: self.policy_lr,
             self.value_lr_ph: self.value_lr}        # update the feed_dict which added old_std, old_means, policy_lr, and value_lr
       
        policy_loss, value_loss, kl, entropy  = self.sess.run([self.policy_loss, self.value_loss, self.kl, self.entropy], feed_dict)
        
        # save the parameters
	self.counter = self.counter + 1

        if (self.counter%save_step) == 0:
            self.saver.save(self.sess, './results/ppo_with_gae_model', global_step= self.counter)

        return policy_loss, value_loss, kl, entropy
    
    def close_sess(self):
        self.sess.close()

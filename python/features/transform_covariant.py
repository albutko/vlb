"""
Transformation Convariant Feature Detector Implementation
https://github.com/ColumbiaDVMM/Transform_Covariant_Detector
original code author: Zhange et. Al
compiler: Alex Butenko
"""
import cv2
import numpy as np
from features.DetectorDescriptorTemplate import DetectorAndDescriptor
import features.feature_utils
import sys

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pickle
import cv2
import os
from skimage.transform import pyramid_gaussian

dirname = os.path.dirname(__file__)
class transform_covariant(DetectorAndDescriptor):
    def __init__(self):
        super(
            transform_covariant,
            self).__init__(
                name='transform_covariant',
                is_detector=True,
                is_descriptor=False,
                is_both=False)

        CNNConfig = {
            "patch_size": 32,
            "descriptor_dim" : 2,
            "batch_size" : 128,
            "alpha" : 1.0,
            "train_flag" : False
        }
        self.model = PatchCNN(CNNConfig)
        file = open(os.path.join(dirname, 'transform_covariant_misc/stats_mexico_tilde_p24_Mexico_train_point.pkl'), 'rb')
        self.mean, self.std = pickle.load(file, encoding='latin1')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        saver = tf.train.Saver()
        try:
            saver.restore(self.sess, os.path.join(dirname, 'transform_covariant_misc/mexico_tilde_p24_Mexico_train_point_translation_iter_20_model.ckpt'))
            print("Model restored.")
        except:
            print('No model found')
            exit()

    def detect_feature(self, image):
        print("extracting")
        image = features.feature_utils.all_to_BGR(image)

        #build image pyramid
        pyramid = pyramid_gaussian(image, max_layer = 4, downscale=np.sqrt(2))

        #predict transformation
        output_list = list()
        for (j, resized) in enumerate(pyramid) :
            fetch = {
                "o1": self.model.o1
            }

            resized = np.asarray(resized)

            resized = (resized-self.mean)/self.std
            resized = resized.reshape((1,resized.shape[0],resized.shape[1],resized.shape[2]))
            result = self.sess.run(fetch, feed_dict={self.model.patch: resized})
            result_mat = result["o1"].reshape((result["o1"].shape[1],result["o1"].shape[2],result["o1"].shape[3]))
            output_list.append(result_mat)


        pts = np.array(output_list)

        return pts

    def close(self):
        self.sess.close()

class PatchCNN:
    def __init__(self, CNNConfig):
        self.patch = tf.placeholder("float32", [1, None, None, 3])

        self.alpha = CNNConfig["alpha"]
        self.descriptor_dim = CNNConfig["descriptor_dim"]
        self._patch_size = CNNConfig["patch_size"]

        with tf.variable_scope("siamese") as scope:
            self.o1 = self.model(self.patch)

        self.o1_flat = tf.reshape(self.o1, [-1, self.descriptor_dim])

    def weight_variable(self, name, shape):
        weight = tf.get_variable(name = name+'_W', shape = shape, initializer = tf.random_normal_initializer(0, 1.0))
        return weight

    def bias_variable(self,name, shape):
        bias = tf.get_variable(name = name + '_b', shape = shape, initializer = tf.constant_initializer(0.0))
        return bias

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def conv2d_layer(self, name, shape, x):
        weight_init = tf.uniform_unit_scaling_initializer(factor=0.5)
        weight = self._variable_with_weight_decay(name=name + '_W', shape = shape, wd = 1e-5);
        bias = self._variable_with_weight_decay(name=name + '_b', shape = [shape[3]], wd = 1e-5);

        conv_val = tf.nn.relu(tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='VALID')+bias)
        return conv_val

    def conv2d_layer_no_relu(self, name, shape, x):
        weight = self._variable_with_weight_decay(name=name + '_W', shape = shape, wd = 1e-5);
        bias = self._variable_with_weight_decay(name=name + '_b', shape = [shape[3]], wd = 1e-5);

        conv_val = tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='VALID')+bias
        return conv_val

    def fc_layer(self, name, shape, x):
        weight = self._variable_with_weight_decay(name=name + '_W', shape = shape, wd = 1e-5);
        bias = self._variable_with_weight_decay(name=name + '_b', shape = [shape[1]], wd = 1e-5);

        fc_val = tf.matmul(x, weight)+bias
        return fc_val

    def _variable_with_weight_decay(self, name, shape, wd):
        dtype = tf.float32
        weight_init = tf.uniform_unit_scaling_initializer(factor=0.5)
        #weight_init = tf.truncated_normal_initializer(stddev=1.0)
        var = tf.get_variable(name=name, dtype = tf.float32, \
                shape=shape, initializer = weight_init)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')

    def model(self, x):
        if self._patch_size==16:
            h_conv1 = self.conv2d_layer('conv1', [5, 5, 3, 32], x)
            h_conv2 = self.conv2d_layer('conv2', [5, 5, 32, 32], h_conv1)
            h_conv3 = self.conv2d_layer('conv3', [5, 5, 32, 64], h_conv2)
            h_conv4 = self.conv2d_layer('conv4', [3, 3, 64, 64], h_conv3)
            h_conv5 = self.conv2d_layer('conv5', [2, 2, 64, 128], h_conv4)
            conv5_flatten = tf.reshape(h_conv5, [-1, 128])
            output = self.conv2d_layer_no_relu('fc1',[128,self.descriptor_dim],conv5_flatten)
        elif self._patch_size == 32:
            h_conv1 = self.conv2d_layer('conv1', [5, 5, 3, 32], x)
            h_pool1 = self.max_pool_2x2(h_conv1)
            h_conv2 = self.conv2d_layer('conv2', [5, 5, 32, 128], h_pool1)
            h_pool2 = self.max_pool_2x2(h_conv2)
            h_conv3 = self.conv2d_layer('conv3', [3, 3, 128, 128], h_pool2)
            h_conv4 = self.conv2d_layer('conv4', [3, 3, 128, 256], h_conv3)
            output = self.conv2d_layer_no_relu('fc1',[1, 1, 256, self.descriptor_dim],h_conv4)
        else:
            output = []
        return  output

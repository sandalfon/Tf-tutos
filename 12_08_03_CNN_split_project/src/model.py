# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:25:53 2018

@author: Sandalfon
"""
import tensorflow as tf

class Model(object):
    def __truncated_normal_var(self, name, shape, dtype):
        return(tf.get_variable(
                name=name, shape=shape, dtype=dtype, 
                initializer=tf.truncated_normal_initializer(stddev=0.05)))
    def __zero_var(self, name, shape, dtype):
        return(tf.get_variable(
                name=name, shape=shape, dtype=dtype, 
                initializer=tf.constant_initializer(0.0)))
    
    def run(self,input_images, options, train_logical=True):
        num_targets = options.getModelOptions('num_targets')
        batch_size = options.getRunOptions('batch_size')
        with tf.variable_scope('conv1') as scope:
            # Conv_kernel is 5x5 for all 3 colors and we will create 64 features
            conv1_kernel = self.__truncated_normal_var(name='conv_kernel1', 
                                                shape=options.getModelOptions('kernel1'), 
                                                dtype=tf.float32)
            # We convolve across the image with a stride size of 1
            conv1 = tf.nn.conv2d(input_images, conv1_kernel,
                                 options.getModelOptions('stride1a'), 
                                 padding=options.getModelOptions('paddTypeS'))
            # Initialize and add the bias term
            conv1_bias = self.__zero_var(name='conv_bias1', 
                                  shape=[options.getModelOptions('shape64')], 
                                  dtype=tf.float32)
            conv1_add_bias = tf.nn.bias_add(conv1, conv1_bias)
            # ReLU element wise
            relu_conv1 = tf.nn.relu(conv1_add_bias)
        
        # Max Pooling
        pool1 = tf.nn.max_pool(relu_conv1, 
                               ksize=options.getModelOptions('ksize1'), 
                               strides=options.getModelOptions('stride1b'),
                               padding=options.getModelOptions('paddTypeS'),
                               name='pool_layer1')
        
        # Local Response Normalization (parameters from paper)
        # paper: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
        norm1 = tf.nn.lrn(pool1, 
                          depth_radius=options.getModelOptions('depth_radius'), 
                          bias=options.getModelOptions('bias'), 
                          alpha=options.getModelOptions('alpha'), 
                          beta=options.getModelOptions('beta'), 
                          name='norm1')
    
        # Second Convolutional Layer
        with tf.variable_scope('conv2') as scope:
            # Conv kernel is 5x5, across all prior 64 features and we create 64 more features
            conv2_kernel = self.__truncated_normal_var(name='conv_kernel2', 
                                                shape=options.getModelOptions('kernel2'), 
                                                dtype=tf.float32)
            # Convolve filter across prior output with stride size of 1
            conv2 = tf.nn.conv2d(norm1, conv2_kernel,
                                 options.getModelOptions('stride1a'), 
                                 padding=options.getModelOptions('paddTypeS'))
            # Initialize and add the bias
            conv2_bias = self.__zero_var(name='conv_bias2', 
                                  shape=[options.getModelOptions('shape64')], 
                                  dtype=tf.float32)
            conv2_add_bias = tf.nn.bias_add(conv2, conv2_bias)
            # ReLU element wise
            relu_conv2 = tf.nn.relu(conv2_add_bias)
        
        # Max Pooling
        pool2 = tf.nn.max_pool(relu_conv2, 
                               ksize=options.getModelOptions('ksize1'), 
                               strides=options.getModelOptions('stride1b'),
                               padding=options.getModelOptions('paddTypeS'), 
                               name='pool_layer2')    
        
         # Local Response Normalization (parameters from paper)
        norm2 = tf.nn.lrn(pool2, 
                          depth_radius=options.getModelOptions('depth_radius'), 
                          bias=options.getModelOptions('bias'), 
                          alpha=options.getModelOptions('alpha'), 
                          beta=options.getModelOptions('beta'),
                          name='norm2')
        
        # Reshape output into a single matrix for multiplication for the fully connected layers
        reshaped_output = tf.reshape(norm2, [batch_size, -1])
        reshaped_dim = reshaped_output.get_shape()[1].value
        
        # First Fully Connected Layer
        with tf.variable_scope('full1') as scope:
            # Fully connected layer will have 384 outputs.
            full_weight1 = self.__truncated_normal_var(name='full_mult1', 
                                                shape=[reshaped_dim,
                                                       options.getModelOptions('shape384')], 
                                                dtype=tf.float32)
            full_bias1 = self.__zero_var(name='full_bias1', 
                                  shape=[options.getModelOptions('shape384')], 
                                  dtype=tf.float32)
            full_layer1 = tf.nn.relu(tf.add(
                    tf.matmul(reshaped_output, full_weight1), full_bias1))
    
        # Second Fully Connected Layer
        with tf.variable_scope('full2') as scope:
            # Second fully connected layer has 192 outputs.
            full_weight2 = self.__truncated_normal_var(name='full_mult2', 
                                                shape=[options.getModelOptions('shape384'),
                                                       options.getModelOptions('shape192')], 
                                                dtype=tf.float32)
            full_bias2 = self.__zero_var(name='full_bias2', 
                                  shape=[options.getModelOptions('shape192')], 
                                  dtype=tf.float32)
            full_layer2 = tf.nn.relu(tf.add(
                    tf.matmul(full_layer1, full_weight2), full_bias2))
    
        # Final Fully Connected Layer -> 10 categories for output (num_targets)
        with tf.variable_scope('full3') as scope:
            # Final fully connected layer has 10 (num_targets) outputs.
            full_weight3 = self.__truncated_normal_var(name='full_mult3', 
                                                shape=[options.getModelOptions('shape192'), 
                                                       num_targets], 
                                                       dtype=tf.float32)
            full_bias3 =  self.__zero_var(name='full_bias3', 
                                   shape=[num_targets], 
                                   dtype=tf.float32)
            final_output = tf.add(tf.matmul(full_layer2, full_weight3), full_bias3)
            
        return(final_output)
    
    def loss(self, logits, targets):
         # Get rid of extra dimensions and cast targets into integers
         targets = tf.squeeze(tf.cast(targets, tf.int32))
         # Calculate cross entropy from logits and targets
         cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                 logits=logits, labels=targets)
         # Take the average loss across batch size
         cross_entropy_mean = tf.reduce_mean(
                 cross_entropy, name='cross_entropy')
         return(cross_entropy_mean)
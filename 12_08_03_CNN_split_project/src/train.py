# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:30:18 2018

@author: Sandalfon
"""
import tensorflow as tf
from evaluator import Evaluator
from data import Data
from model import Model
from statPlot import StatPlot

class Train(object):
    
    def train(self, loss_value, options):
        generation_num = tf.Variable(0, trainable=False)
        learning_rate = options.getRunOptions('learning_rate')
        num_gens_to_wait = options.getRunOptions('num_gens_to_wait')
        lr_decay = options.getRunOptions('lr_decay')
        # Our learning rate is an exponential decay after we wait a fair number of generations
        model_learning_rate = tf.train.exponential_decay(learning_rate, generation_num,
                                                         num_gens_to_wait, lr_decay, staircase=True)
        # Create optimizer
        my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
        # Initialize train step
        train_step = my_optimizer.minimize(loss_value)
        return(train_step)
        
    def run(self, options):
        
        sess = tf.Session()
        evaluator = Evaluator()
        data = Data()
        model = Model()
        statPlot = StatPlot()
        
        print('Getting/Transforming Data.')
        # Initialize the data pipeline
        images, targets = data.batches(options, train_logical=True)
        # Get batch test images and targets from pipline
        test_images, test_targets = data.batches(options, train_logical=False)
        # Declare Model
        print('Creating the CIFAR10 Model.')
        with tf.variable_scope('model_definition') as scope:
            # Declare the training network model
            
            model_output = model.run(images, options)
            #model_output = cifar_cnn_model(images, batch_size)
            # This is very important!!!  We must set the scope to REUSE the variables,
            #  otherwise, when we set the test network model, it will create new random
            #  variables.  Otherwise we get random evaluations on the test batches.
            scope.reuse_variables()
            test_output = model.run(test_images, options)
        #    test_output = cifar_cnn_model(test_images, batch_size)
        # Declare loss function
        print('Declare Loss Function.')
        loss = model.loss(model_output, targets)
        
        # Create accuracy function
        accuracy = evaluator.accuracy(test_output, test_targets)
        
        # Create training operations
        print('Creating the Training Operation.')
        #generation_num = tf.Variable(0, trainable=False)
        train_op = self.train(loss, options)
        
        # Initialize Variables
        print('Initializing the Variables.')
        init = tf.global_variables_initializer()
        sess.run(init)
        
        # Initialize queue (This queue will feed into the model, so no placeholders necessary)
        tf.train.start_queue_runners(sess=sess)
        
        print('Starting Training')
        train_loss = []
        test_accuracy = []
        for i in range(options.getRunOptions('generations')):
            _, loss_value = sess.run([train_op, loss])
            
            if (i+1) % options.getRunOptions('output_every') == 0:
                train_loss.append(loss_value)
                output = 'Generation {}: Loss = {:.5f}'.format((i+1), loss_value)
                print(output)
            
            if (i+1) % options.getRunOptions('eval_every') == 0:
                [temp_accuracy] = sess.run([accuracy])
                test_accuracy.append(temp_accuracy)
                acc_output = ' --- Test Accuracy = {:.2f}%.'.format(100.*temp_accuracy)
                print(acc_output)
        statPlot.plot(train_loss, test_accuracy, options)
        return(train_loss, test_accuracy)
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:24:05 2018

@author: Sandalfon
"""

import tensorflow as tf

class Evaluator(object):
# Accuracy function
    def accuracy(self, logits, targets):
        # Make sure targets are integers and drop extra dimensions
        targets = tf.squeeze(tf.cast(targets, tf.int32))
        # Get predicted values by finding which logit is the greatest
        batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
        # Check if they are equal across the batch
        predicted_correctly = tf.equal(batch_predictions, targets)
        # Average the 1's and 0's (True's and False's) across the batch size
        accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
        return(accuracy)
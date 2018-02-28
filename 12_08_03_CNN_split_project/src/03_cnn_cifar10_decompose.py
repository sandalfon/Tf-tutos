# More Advanced CNN Model: CIFAR-10
#---------------------------------------
#
# In this example, we will download the CIFAR-10 images
# and build a CNN model with dropout and regularization
#
# CIFAR is composed ot 50k train and 10k test
# images that are 32x32.

import os
import sys
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from six.moves import urllib
from tensorflow.python.framework import ops
from options import Options
from data_dl import DataDl
from data import Data 
from model import Model
ops.reset_default_graph()

options=Options()
options.loadFromJson('options.json')
dataDl=DataDl()
dataDl.dl_end_extract(options)

# Change Directory
#abspath = os.path.abspath(__file__)
#dname = os.path.dirname(abspath)
#os.chdir(dname)

# Start a graph session
sess = tf.Session()
model = Model()
# Set model parameters
batch_size = options.getRunOptions('batch_size')
#data_dir = 'temp'
output_every = options.getRunOptions('output_every')
generations = options.getRunOptions('generations')
eval_every = options.getRunOptions('eval_every')
image_height = options.getDataOptions('img_height')
image_width = options.getDataOptions('img_width')
crop_height = options.getDataOptions('crop_height')
crop_width = options.getDataOptions('crop_width')
num_channels = options.getDataOptions('num_channels')
num_targets = options.getModelOptions('num_targets')
#extract_folder = 'cifar-10-batches-bin'

# Exponential Learning Rate Decay Params
learning_rate = options.getRunOptions('learning_rate')
lr_decay = options.getRunOptions('lr_decay')
num_gens_to_wait = options.getRunOptions('num_gens_to_wait')

# Extract model parameters
image_vec_length = image_height * image_width * num_channels
record_length = 1 + image_vec_length # ( + 1 for the 0-9 label)


# Train step
def train_step(loss_value, generation_num):
    # Our learning rate is an exponential decay after we wait a fair number of generations
    model_learning_rate = tf.train.exponential_decay(learning_rate, generation_num,
                                                     num_gens_to_wait, lr_decay, staircase=True)
    # Create optimizer
    my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
    # Initialize train step
    train_step = my_optimizer.minimize(loss_value)
    return(train_step)


# Accuracy function
def accuracy_of_batch(logits, targets):
    # Make sure targets are integers and drop extra dimensions
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    # Get predicted values by finding which logit is the greatest
    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    # Check if they are equal across the batch
    predicted_correctly = tf.equal(batch_predictions, targets)
    # Average the 1's and 0's (True's and False's) across the batch size
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    return(accuracy)

# Get data
print('Getting/Transforming Data.')
# Initialize the data pipeline
images, targets = Data().batches(options, train_logical=True)
# Get batch test images and targets from pipline
test_images, test_targets = Data().batches(options, train_logical=False)

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
accuracy = accuracy_of_batch(test_output, test_targets)

# Create training operations
print('Creating the Training Operation.')
generation_num = tf.Variable(0, trainable=False)
train_op = train_step(loss, generation_num)

# Initialize Variables
print('Initializing the Variables.')
init = tf.global_variables_initializer()
sess.run(init)

# Initialize queue (This queue will feed into the model, so no placeholders necessary)
tf.train.start_queue_runners(sess=sess)

# Train CIFAR Model
print('Starting Training')
train_loss = []
test_accuracy = []
for i in range(generations):
    _, loss_value = sess.run([train_op, loss])
    
    if (i+1) % output_every == 0:
        train_loss.append(loss_value)
        output = 'Generation {}: Loss = {:.5f}'.format((i+1), loss_value)
        print(output)
    
    if (i+1) % eval_every == 0:
        [temp_accuracy] = sess.run([accuracy])
        test_accuracy.append(temp_accuracy)
        acc_output = ' --- Test Accuracy = {:.2f}%.'.format(100.*temp_accuracy)
        print(acc_output)

# Print loss and accuracy
# Matlotlib code to plot the loss and accuracies
eval_indices = range(0, generations, eval_every)
output_indices = range(0, generations, output_every)

# Plot loss over time
plt.plot(output_indices, train_loss, 'k-')
plt.title('Softmax Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')
plt.show()

# Plot accuracy over time
plt.plot(eval_indices, test_accuracy, 'k-')
plt.title('Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.show()
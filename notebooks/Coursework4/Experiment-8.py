#!/disk/scratch/mlp/miniconda2/bin/python
# coding: utf-8

# # object recognition with CIFAR-100

# In[1]:

import os
import tensorflow as tf
import numpy as np
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider
import matplotlib.pyplot as plt
import time



# In[2]:

seed = 10102016
rng = np.random.RandomState(seed)
train_data = CIFAR100DataProvider('train', batch_size=50, rng = rng)
valid_data = CIFAR100DataProvider('valid', batch_size=50, rng = rng)


# # Regularisation

# ## Experiment Baseline
# ### CIFAR-100

# In[3]:

def fully_connected_layer(inputs, input_dim, output_dim, nonlinearity=tf.nn.relu, dropout=False, keep_prob=1.0, wd= None):
    with tf.device('/cpu:0'):
        weights = tf.Variable(
            tf.truncated_normal(
                [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5),
            'weights')
        biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
        weights = tf.add(weights,weight_decay)
    if dropout:
        outputs = tf.nn.dropout(nonlinearity(tf.matmul(inputs, weights) + biases), keep_prob)
    else:
        outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs


# In[4]:

def kernel(name, shape, stddev, wd = None):
    dtype = tf.float32
    with tf.device('/cpu:0'):
        kernel_weights = tf.get_variable(name, shape,
                                         initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
        kernel_weights = tf.add(kernel_weights,weight_decay)
    return kernel_weights


# In[5]:

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.contrib.layers.xavier_initializer_conv2d())
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


# In[ ]:

tf.reset_default_graph()

inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')

# for setting dropout
keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('data_augmentation'):
    reshape_inputs = tf.reshape(inputs, [50, 3, 32, 32])
    result = tf.image.random_flip_left_right(reshape_inputs)

with tf.name_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights_1',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=0.0001)
    conv = tf.nn.conv2d(result, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases_1', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name='cnn1')

pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')
# norm1
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')

# conv2
with tf.name_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights_2',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0001)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases_2', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name='cnn2')

norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
with tf.name_scope('hidden1') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [50, -1])
    hidden_1 = fully_connected_layer(reshape, 4096, 1024, dropout = True, keep_prob = keep_prob)

# local4
with tf.name_scope('outputs') as scope:
    #weights = _variable_with_weight_decay('weights', shape=[384, 192],
    #                                      stddev=0.04, wd=0.004)
    #biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    #local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    #_activation_summary(local4)
    outputs = fully_connected_layer(hidden_1, 1024, train_data.num_classes, tf.identity)


with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)),
            tf.float32))

with tf.name_scope('train'):
    train_step = tf.train.MomentumOptimizer(learning_rate = 0.001, momentum = 0.9).minimize(error)

init = tf.global_variables_initializer()


# In[ ]:

train_c100_exp3 = {'epoch':[], 'error':[], 'accuracy':[], 'time':[]}
valid_c100_exp3 = {'epoch':[], 'error':[], 'accuracy':[]}

train_data.reset()
valid_data.reset()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        start_time = time.time()
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy],
                feed_dict={inputs: input_batch, targets: target_batch, keep_prob:0.8})
            running_error += batch_error
            running_accuracy += batch_acc
        epoch_time = time.time() - start_time
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        train_c100_exp3['epoch'].append(e+1)
        train_c100_exp3['error'].append(running_error)
        train_c100_exp3['accuracy'].append(running_accuracy)
        train_c100_exp3['time'].append(epoch_time)
        print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f} running time={3:.2f}'
              .format(e + 1, running_error, running_accuracy, epoch_time))
        if (e + 1) % 1 == 0:
            valid_error = 0.
            valid_accuracy = 0.
            for input_batch, target_batch in valid_data:
                batch_error, batch_acc = sess.run(
                    [error, accuracy],
                    feed_dict={inputs: input_batch, targets: target_batch, keep_prob:1.0})
                valid_error += batch_error
                valid_accuracy += batch_acc
            valid_error /= valid_data.num_batches
            valid_accuracy /= valid_data.num_batches
            valid_c100_exp3['epoch'].append(e+1)
            valid_c100_exp3['error'].append(valid_error)
            valid_c100_exp3['accuracy'].append(valid_accuracy)
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[ ]:

import matplotlib.pyplot as plt


plt.style.use('ggplot')
fig1 = plt.figure(figsize=(12, 16))
ax1 = fig1.add_subplot(2, 1, 1)
ax2 = fig1.add_subplot(2, 1, 2)
ax1.plot(np.array(train_c100_exp3['error']), label='baseline')
ax1.plot(valid_c100_exp3['epoch'], valid_c100_exp3['error'], label='baseline')

ax2.plot(np.array(train_c100_exp3['accuracy']), label='baseline')
ax2.plot(valid_c100_exp3['epoch'], valid_c100_exp3['accuracy'], label='baseline')

ax1.legend(loc='best')
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Error')
ax2.legend(loc='best')
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Error')

plt.show()
fig1.tight_layout()
fig1.savefig('c100_3.png', dpi=200)


# In[ ]:

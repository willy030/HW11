import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average
import numpy as np
import os

import numpy as np
import random as rn
import tensorflow as tf
import threading
import time

global n_classes
n_classes = 50
_weights = {
        'wc1': tf.get_variable("wc1", [7, 7, 3, 96], initializer=tf.glorot_uniform_initializer()),
        'wc2': tf.get_variable('wc2',[5, 5, 96, 256], initializer=tf.glorot_uniform_initializer()),
        'wc3': tf.get_variable('wc3',[3, 3, 256, 384], initializer=tf.glorot_uniform_initializer()),
        'wc4': tf.get_variable('wc4',[3, 3, 384, 384], initializer=tf.glorot_uniform_initializer()),
        'wc5': tf.get_variable('wc5',[3, 3, 384, 256], initializer=tf.glorot_uniform_initializer()),
        'wd2': tf.get_variable('wd2',[4096, 4096], initializer=tf.glorot_uniform_initializer()),
        'out': tf.get_variable('out',[4096, n_classes], initializer=tf.glorot_uniform_initializer())
    }
_biases = {
        'bc1': tf.get_variable('bc1',[96], initializer=tf.glorot_uniform_initializer()),
        'bc2': tf.get_variable('bc2',[256], initializer=tf.glorot_uniform_initializer()),
        'bc3': tf.get_variable('bc3',[384], initializer=tf.glorot_uniform_initializer()),
        'bc4': tf.get_variable('bc4',[384], initializer=tf.glorot_uniform_initializer()),
        'bc5': tf.get_variable('bc5',[256], initializer=tf.glorot_uniform_initializer()),
        'bd2': tf.get_variable('db2',[4096], initializer=tf.glorot_uniform_initializer()),
        'out': tf.get_variable('bout',[n_classes], initializer=tf.glorot_uniform_initializer())
    }
def activation(x,name="activation"):
    return tf.nn.relu(x, name=name)
    
def conv2d(name, l_input, w, b, s, p, scope):
    l_input = tf.nn.conv2d(l_input, w, strides=[1,s,s,1], padding=p, name=name)
    l_input = activation(l_input+b)
    
    return l_input

def max_pool(name, l_input, k, s):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='VALID', name=name)

def norm(l_input, lsize=4, name="lrn"):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
   
def alex_net(_X, _dropout, batch_size):
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'], 3, 'VALID', 'conv1')
    pool1 = max_pool('pool1', conv1, k=3,s=2)
    conv2 = conv2d('conv2', pool1, _weights['wc2'], _biases['bc2'], 1, 'SAME', 'conv2')
    pool2 = max_pool('pool2', conv2, k=3,s=2)
    conv3 = conv2d('conv3', pool2, _weights['wc3'], _biases['bc3'], 1, 'SAME', 'conv3')
    conv4 = conv2d('conv4', conv3, _weights['wc4'], _biases['bc4'], 1, 'SAME', 'conv4')
    conv5 = conv2d('conv5', conv4, _weights['wc5'], _biases['bc5'], 1, 'SAME', 'conv5')
    pool5 = max_pool('pool2', conv5, k=3,s=2)
    # Find current size of conv5 to fit the requirement of FC1.
    sizes = pool5.get_shape().as_list()
    neurons =  sizes[1]*sizes[2]*sizes[3]
    dense1 = tf.reshape(pool5, [batch_size, neurons]) # Reshape conv5 output to fit dense layer input
    wei_den1 = tf.get_variable('wd3',[neurons, 4096], initializer=tf.glorot_uniform_initializer())
    b_den1 =  tf.get_variable('wd4',[4096], initializer=tf.glorot_uniform_initializer())
    
    dense1 = activation(tf.matmul(dense1, wei_den1) + b_den1, name='fc1') # Relu activation
    dd1=tf.nn.dropout(dense1, _dropout)
    dense2 = activation(tf.matmul(dd1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
    out = tf.matmul(dense2, _weights['out']) + _biases['out'] # Relu activation

    return out
#==========================================================================
#=============Reading data in multithreading manner========================
#==========================================================================
def read_labeled_image_list(image_list_file, training_img_dir):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []

    for line in f:
        filename, label = line[:-1].split(' ')
        filename = training_img_dir+filename
        filenames.append(filename)
        labels.append(int(label))
        
    return filenames, labels
    
    
def read_images_from_disk(input_queue, size1=256):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    fn=input_queue[0]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    
    #example = tf.image.decode_png(file_contents, channels=3, name="dataset_image") # png fo rlfw
    example=tf.image.resize_images(example, [size1,size1])
    return example, label, fn
    
def setup_inputs(sess, filenames, training_img_dir, image_size=256, crop_size=224, isTest=False, batch_size=128):
    
    # Read each image file
    image_list, label_list = read_labeled_image_list(filenames, training_img_dir)

    images = tf.cast(image_list, tf.string)
    labels = tf.cast(label_list, tf.int64)
     # Makes an input queue
    if isTest is False:
        isShuffle = True
    else:
        isShuffle = False
        
    input_queue = tf.train.slice_input_producer([images, labels], shuffle=isShuffle)
    image, y,fn = read_images_from_disk(input_queue)

    channels = 3
    image.set_shape([None, None, channels])
        
    # Crop and other random augmentations
    if isTest is False:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_saturation(image, .95, 1.05)
        image = tf.image.random_brightness(image, .05)
        image = tf.image.random_contrast(image, .95, 1.05)
        

    image = tf.random_crop(image, [crop_size, crop_size, 3])
    image = tf.cast(image, tf.float32)/255.0
    
    image, y,fn = tf.train.batch([image, y, fn], batch_size=batch_size, capacity=4,name='labels_and_images')

    tf.train.start_queue_runners(sess=sess)

    return image, y, fn, len(label_list)
# Training setting
batch_size = 128 
display_step = 80
dropout = 0.5# Dropout rate
keep_prob = tf.placeholder(tf.float32)          # Dropout rate to be fed
learning_rate = tf.placeholder(tf.float32)      # Learning rate to be fed
lr = 1e-3                                   # Learning rate start

# Setup the tensorflow...
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

print("Preparing the training & validation data...")
train_data, train_labels, filelist1, glen1 = setup_inputs(sess, "train.txt", "./", batch_size=batch_size)
val_data, val_labels, filelist2, tlen1 = setup_inputs(sess, "val.txt", "./", batch_size=batch_size)

max_iter = glen1*100

print("Preparing the training model with learning rate = %.5f..." % (lr))

with tf.variable_scope("alexnet", reuse=tf.AUTO_REUSE) as scope:
    pred = alex_net(train_data,keep_prob,batch_size)

with tf.name_scope('Loss_and_Accuracy'):
  cost = tf.losses.sparse_softmax_cross_entropy(labels=train_labels, logits=pred)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
  correct_prediction = tf.equal(tf.argmax(pred, 1), train_labels)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  top5=tf.reduce_mean(tf.cast(tf.nn.in_top_k(pred, train_labels, 5), tf.float32))
  
  tf.summary.scalar('Loss', cost)
  tf.summary.scalar('Training_Accuracy', accuracy)
  tf.summary.scalar('Top-5_accuracy', top5)
saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess.run(init)
step = 0
writer = tf.summary.FileWriter("/tmp/log2", sess.graph)
summaries = tf.summary.merge_all()

print("We are going to train the ImageNet model based on AlexNet!!!")
while (step * batch_size) < max_iter:
    epoch1=np.floor((step*batch_size)/glen1)
    if (((step*batch_size)%glen1 < batch_size) & (lr==1e-3) & (epoch1 >2)):
        lr /= 10

    sess.run(optimizer,  feed_dict={keep_prob: dropout, learning_rate: lr})

    if (step % 15000==1) & (step>15000):
        save_path = saver.save(sess, "checkpoint/tf_alex_model_iter" + str(step) + ".ckpt")
        print("Model saved in file at iteration %d: %s" % (step*batch_size,save_path))

    if step % display_step == 1:
        # calculate the loss
        loss, acc, top5acc, summaries_string = sess.run([cost, accuracy, top5, summaries], feed_dict={keep_prob: 1.})
        print("Iter=%d/epoch=%d, Loss=%.6f, Training Accuracy=%.6f, Top-5 Accuracy=%.6f, lr=%f" % (step*batch_size, epoch1 ,loss, acc, top5acc, lr))
        writer.add_summary(summaries_string, step)


    step += 1
print("Optimization Finished!")
save_path = saver.save(sess, "checkpoint/tf_alex_model.ckpt")
print("Model saved in file: %s" % save_path)
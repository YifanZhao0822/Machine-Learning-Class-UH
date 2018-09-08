# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 00:00:18 2018

@author: Administrator
"""
# imports
import tensorflow as tf
import numpy as np
import random
import pandas as pd
from ops import fc_layer

# record number of training samples
MFCC=pd.read_csv('C://Users//Administrator//Desktop//FrogsMFCCstrain.csv')
MFCC=np.mat(MFCC)
MFCC_case=MFCC[:,0:21]
MFCC_label=MFCC[:,21]
len_M_label=MFCC_label.shape[0]

    
def batch(in_data_case,in_data_label,batch_size,population_length):
    flag=random.sample(range(population_length),batch_size)
    batch_case=in_data_case[flag,:]
    data_without_batch_case=np.delete(in_data_case,flag,0)
    batch_label=in_data_label[:,flag]
    data_without_batch_label=np.delete(in_data_label,flag,1)
    return batch_case,batch_label,data_without_batch_case,data_without_batch_label

def MFCC_renew(input_data):
# the last column of input_data should be single label, 
# and input_data should be numpy type
    column_len=input_data.shape[1]
    data_case=input_data[:,0:column_len-1]
    data_label=input_data[:,column_len-1]
    data_label_len=data_label.shape[0]
    data_label1=np.zeros((3,data_label_len))
    for i in range(data_label_len):
        data_label1[int(data_label[i])-1,i]=1
    data_label=data_label1
    return data_case,data_label
def testf():
    W=np.array([[1,2,3],[1,2,3]])
    b=np.array([1,2,3])
    return W,b
# import test set
MFCC_test=pd.read_csv('C://Users//Administrator//Desktop//FrogsMFCCstest.csv')
MFCC_test=np.mat(MFCC_test)
MFCC_test_case,MFCC_test_label=MFCC_renew(MFCC_test)
flag_class1= np.where(MFCC_test[:,-1]==1)
flag_class1= flag_class1[0]
flag_class2= np.where(MFCC_test[:,-1]==2)
flag_class2= flag_class2[0]
flag_class3= np.where(MFCC_test[:,-1]==3)
flag_class3= flag_class3[0]
MFCC_test_case_class1=MFCC_test_case[flag_class1,:]
MFCC_test_label_class1=MFCC_test_label[:,flag_class1]
MFCC_test_case_class2=MFCC_test_case[flag_class2,:]
MFCC_test_label_class2=MFCC_test_label[:,flag_class2]
MFCC_test_case_class3=MFCC_test_case[flag_class3,:]
MFCC_test_label_class3=MFCC_test_label[:,flag_class3]
# input data:
# Number of classes, one class for each of 10 digits.
n_classes = 3

# number of units in the first hidden layer
h1 = 52
epochs = 10  # Total number of training epochs
batch_size = 108  # Training batch size
display_freq = 25  # Frequency of displaying the training results
num_tr_iter = int(len_M_label / batch_size)
# Create graph
# Placeholders for inputs (x), outputs(y)
x = tf.placeholder(tf.float32, shape=[None, 21], name='X')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')
learning_r= tf.placeholder(tf.float32, name='Epsilon')
fc1= fc_layer(x, h1, 'FC1', use_relu=True)
# initialize W_input2hidden and b_hidden to record Wn and Wn-1
W_input2hidden_former=tf.Variable(tf.zeros(shape=[x.get_shape()[1],h1]),name='W_input2hidden_former')
b_hidden_former=tf.Variable(tf.zeros(shape=[h1]),name='b_hidden_former')
output_logits= fc_layer(fc1[0], n_classes, 'OUT', use_relu=False)
# initialize W_hidden2output and b_output to record Wn and Wn-1
W_hidden2output_former=tf.Variable(tf.zeros(shape=[fc1[0].get_shape()[1],n_classes]),name='W_hidden2output_former')
b_output_former=tf.Variable(tf.zeros(shape=[n_classes]),name='b_output_former')

# Define the loss function, optimizer, and accuracy
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=output_logits[0]), name='loss')
    tf.summary.scalar('loss',loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_r).minimize(loss)
correct_prediction = tf.equal(tf.argmax(output_logits[0], 1), tf.argmax(y, 1), name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
# compute difference between Wn-1 and Wn
delta_W=tf.square(tf.norm(fc1[1]-W_input2hidden_former))+tf.square(tf.norm(output_logits[1]-W_hidden2output_former))
delta_b=tf.square(tf.norm(fc1[2]-b_hidden_former))+tf.square(tf.norm(output_logits[2]-b_output_former))
delta_Wb=tf.sqrt(delta_W+delta_b)
G_n=-delta_Wb*learning_r#gradient
# initialize G_n_record, rLoss_record, W_n_record:
G_n_record=tf.Variable(tf.zeros(shape=[epochs*num_tr_iter],name='G_n_record'))
rLoss_n_record=tf.Variable(tf.zeros(shape=[epochs*num_tr_iter],name='rLoss_n_record'))
W_n_record=tf.Variable(tf.zeros(shape=[epochs*num_tr_iter],name='W_n_record'))

# Network predictions
cls_prediction = tf.argmax(output_logits[0], axis=1, name='predictions')

# Initializing the variables
# hyper-parameters
# Network Parameters
init = tf.global_variables_initializer()
merged=tf.summary.merge_all()
# Launch the graph (session)
with tf.Session() as sess:
    train_writer=tf.summary.FileWriter(logdir='./tslogs/', graph=sess.graph)
    sess.run(init)
    counter_n = 0# to reduce the value of gradient step size
    # Number of training iterations in each epoch
    for epoch in range(epochs):
        print('Training epoch: {}'.format(epoch+1))
        MFCC_case,MFCC_label=MFCC_renew(MFCC)
        for iteration in range(num_tr_iter):
            counter_n +=1
            batch_x,batch_y,MFCC_case,MFCC_label=batch(MFCC_case,MFCC_label,batch_size,len_M_label-iteration*batch_size)
            # Run optimization op (backprop)
            feed_dict_batch = {x: batch_x, y: np.transpose(batch_y), learning_r: 0.9/counter_n}
            _, summary_tr = sess.run([optimizer, merged], feed_dict=feed_dict_batch)
            train_writer.add_summary(summary_tr, counter_n)
            delta_Wb_batch, G_n_batch = sess.run([delta_Wb,G_n],feed_dict=feed_dict_batch)
            loss_batch, acc_batch = sess.run([tf.sqrt(loss), accuracy],
                                                 feed_dict=feed_dict_batch)
            sess.run(tf.assign(G_n_record[counter_n-1],G_n_batch))
            sess.run(tf.assign(rLoss_n_record[counter_n-1],loss_batch))
            sess.run(tf.assign(W_n_record[counter_n-1],delta_Wb_batch))
            if iteration % display_freq == 0:
                # Calculate and display the batch loss and accuracy
                print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                      format(iteration, loss_batch, acc_batch))
                print("iter {0:3d}:\t Delta_W={1:.2f},\tG_n={2:.4f}".
                      format(iteration, delta_Wb_batch, G_n_batch))                
                
            W_input2hidden_present= sess.run(fc1,feed_dict={x:batch_x})[1]
            b_hidden_present= sess.run(fc1,feed_dict={x:batch_x})[2]
            W_hidden2output_present= sess.run(output_logits,feed_dict={x:batch_x})[1]
            b_output_present= sess.run(output_logits,feed_dict={x:batch_x})[2]
            sess.run(tf.assign(W_input2hidden_former,W_input2hidden_present))
            sess.run(tf.assign(b_hidden_former,b_hidden_present))
            sess.run(tf.assign(W_hidden2output_former,W_hidden2output_present))
            sess.run(tf.assign(b_output_former,b_output_present))

    steps=range(counter_n)+1
            
    # Test the network after training
    feed_dict_test = {x: MFCC_test_case_class1, y: np.transpose(MFCC_test_label_class1),
                      learning_r:0.9/counter_n}
    loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
    print('---------------------------------------------------------')
    print("Test loss class1: {0:.2f}, test accuracy class1: {1:.01%}".format(loss_test, acc_test))
    print('---------------------------------------------------------')
    feed_dict_test = {x: MFCC_test_case_class2, y: np.transpose(MFCC_test_label_class2),
                      learning_r:0.9/counter_n}
    loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
    print('---------------------------------------------------------')
    print("Test loss class2: {0:.2f}, test accuracy class2: {1:.01%}".format(loss_test, acc_test))
    print('---------------------------------------------------------')
    feed_dict_test = {x: MFCC_test_case_class3, y: np.transpose(MFCC_test_label_class3),
                      learning_r:0.9/counter_n}
    loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
    print('---------------------------------------------------------')
    print("Test loss class3: {0:.2f}, test accuracy class3: {1:.01%}".format(loss_test, acc_test))
    print('---------------------------------------------------------')
    feed_dict_test = {x: MFCC_test_case, y: np.transpose(MFCC_test_label),
                      learning_r:0.9/counter_n}
    loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
    print('---------------------------------------------------------')
    print("Test loss whole test: {0:.2f}, test accuracy whole test: {1:.01%}".format(loss_test, acc_test))
    print('---------------------------------------------------------')
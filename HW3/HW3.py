import tensorflow as tf
import numpy as np
import pandas as pd
from ops import fc_layer, batch
from matplotlib.mlab import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#import the data set
trains = [[], [], [], [], []]
tests= [[], [], [], [], []]
for k in range(1, 6):
    #the training set
    for i in range(1, 12):
        df = pd.read_csv('F:/Files/DataMining/HW3/' + str(k) + '/dataset' + str(i) + '.csv',
                         usecols=[1, 2, 3, 4, 5, 6],skiprows=[0, 1, 2, 3, 4], header=None)
        #insert the correct output behind each case
        for j in range(6, 11):
            if j == k + 5:
                df.insert(loc=j, column=str(j + 1), value=np.ones([len(df), 1]))
            else:
                df.insert(loc=j, column=str(j + 1), value=np.zeros([len(df), 1]))
        trains[k-1].extend(df.values)
    #the test set
    for i in range(12,16):
        df = pd.read_csv('F:/Files/DataMining/HW3/' + str(k) + '/dataset' + str(i) + '.csv',
                         usecols=[1, 2, 3, 4, 5, 6],skiprows=[0, 1, 2, 3, 4], header=None)
        for j in range(6, 11):
            if j == k + 5:
                df.insert(loc=j, column=str(j + 1), value=np.ones([len(df), 1]))
            else:
                df.insert(loc=j, column=str(j + 1), value=np.zeros([len(df), 1]))
        tests[k-1].extend(df.values)

#use PCA to decide the size of the hidden layer
d1 = []
d = [[], [], [], [], []]
pca1 = []
pca = [[], [], [], [], []]
count1 = 0
count = np.zeros(5)
color = ['b', 'g', 'r', 'c', 'm']
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
for k in range(1, 6):
    for i in range(1, 16):
        df = pd.read_csv('F:/Files/DataMining/HW3/' + str(k) + '/dataset' + str(i) + '.csv',
                         usecols=[1, 2, 3, 4, 5, 6],skiprows=[0, 1, 2, 3, 4], header=None)
        d[k - 1].extend(df.values)
    d[k - 1] = np.asarray(d[k - 1])
    pca[k - 1] = PCA(d[k - 1])  #do PCA for each class
    #ax.scatter(pca[k-1].Y[:, 0], pca[k-1].Y[:, 1], pca[k-1].Y[:, 2], c=color[k-1],alpha=0.125)
    d1.extend(d[k - 1])
    #compute the number of eigenvalues preserving 95% of the total sum of eigenvalues
    for m in range(6):
        count[k - 1] += pca[k - 1].fracs[m]
        if count[k - 1] >= 0.9:
            break
    count[k - 1] = m + 1
#plt.show()
d1 = np.asarray(d1)
pca1 = PCA(d1)  #do PCA for the whole data set
for m in range(6):
    count1 += pca1.fracs[m]
    if count1 >= 0.95:
        break
#compute the size of the hidden layer
s = m + 1
S = sum(count)
SL = 2 * S

train=[]
test=[]
for k in range(5):
    train.extend(trains[k])
    test.extend(tests[k])
trains=np.array(trains)
tests=np.array(tests)
train=np.asarray(train)
test=np.asarray(test)
np.random.shuffle(train)
np.random.shuffle(test)


n_des = 6
epochs = 10
display_freq = 60
batch_size = 100
test_size = 9000
n_classes = 5
h = SL

x = tf.placeholder(tf.float64, shape=[None, n_des], name='X')
y = tf.placeholder(tf.float64, shape=[None, n_classes], name='Y')
learning_rate = tf.placeholder(tf.float64, shape=None, name='epsilon')
W_hid2out_form=tf.Variable(tf.zeros(shape=[h,n_classes]),name='W_hid2out_form')
W_in2hid_form=tf.Variable(tf.zeros(shape=[n_des,h]),name='W_in2hid_form')
b_hid2out_form=tf.Variable(tf.zeros(shape=[n_classes]),name='b_hid2out_form')
b_in2hid_form=tf.Variable(tf.zeros(shape=[h]),name='b_in2hid_form')
W_hid2out_form=tf.cast(W_hid2out_form,tf.float64)
W_in2hid_form=tf.cast(W_in2hid_form,tf.float64)
b_hid2out_form=tf.cast(b_hid2out_form,tf.float64)
b_in2hid_form=tf.cast(b_in2hid_form,tf.float64)

fc1 = fc_layer(x, h, 'Hidden_layer', use_relu=True)
output_logits = fc_layer(fc1[0], n_classes, 'Output_layer', use_relu=False)

#define the loss
with tf.name_scope('RMSE'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=output_logits[0]), name='RMSE')
    tf.summary.scalar('RMSE',loss)
#define the optimizer
with tf.name_scope('Optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Op').minimize(loss)

#compute the number of correct predictions
correct_prediction = tf.equal(tf.argmax(output_logits[0], 1), tf.argmax(y, 1), name='co_pre')
#compute the percentage of correct predictions in the whole batch
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64), name='accuracy')

#the variable that signifies the class of denominator
indice=tf.placeholder(tf.int64,shape=[1],name='Index')
#initialization of the two matrices
entry_train=[[], [], [], [], []]
entry_test=[[], [], [], [], []]
#compare the output vector to the given class index and count the numbers in order to compute the numerator
prediction = tf.equal(tf.argmax(output_logits[0], 1), indice)
#compute the ratio of prediction and the number of cases in one class
ratio = tf.reduce_mean(tf.cast(prediction, tf.float64))

#compute the norm of the difference between the weights of the former and the present batches
delta_W=tf.square(tf.norm(fc1[1]-W_in2hid_form))+tf.square(tf.norm(output_logits[1]-W_hid2out_form))
delta_b=tf.square(tf.norm(fc1[2]-b_in2hid_form))+tf.square(tf.norm(output_logits[2]-b_hid2out_form))
with tf.name_scope('Delta_Wb'):
    difference=tf.sqrt(delta_W+delta_b,name='Delta_Wb')
    tf.summary.scalar('Delta_Wb',difference)
#compute the gradient
with tf.name_scope('Gradient'):
    gradient=tf.multiply(learning_rate,difference,name='Gradient')
    tf.summary.scalar('Gradient', gradient)

#compute the absolute values of the coordinates of the gradient and plot the histograms
with tf.name_scope('Grads_W_in2hid'):
    grads_W_in2hid=tf.stack(tf.abs(tf.gradients(loss, fc1[1],stop_gradients=fc1[1],name='Grads_W_in2hid')))
    tf.summary.histogram('Grads_W_in2hid',grads_W_in2hid)
with tf.name_scope('Grads_b_in2hid'):
    grads_b_in2hid=tf.stack(tf.abs(tf.gradients(loss, fc1[2],stop_gradients=fc1[2],name='Grads_b_in2hid')))
    tf.summary.histogram('Grads_b_in2hid',grads_b_in2hid)
with tf.name_scope('Grads_W_hid2out'):
    grads_W_hid2out=tf.stack(tf.abs(tf.gradients(loss, output_logits[1],stop_gradients=output_logits[1],name='Grads_W_hid2out')))
    tf.summary.histogram('Grads_W_hid2out',grads_W_hid2out)
with tf.name_scope('Grads_b_hid2out'):
    grads_b_hid2out=tf.stack(tf.abs(tf.gradients(loss, output_logits[2],stop_gradients=output_logits[2],name='Grads_b_hid2out')))
    tf.summary.histogram('Grads_b_hid2out',grads_b_hid2out)

cls_prediction = tf.argmax(output_logits[0], axis=1, name='predictions')

init_op = tf.global_variables_initializer()
merged=tf.summary.merge_all()

with tf.Session() as sess:
    W=[]
    G=[]
    sess.run(init_op)
    train_writer = tf.summary.FileWriter(logdir="./logs/", graph=sess.graph)
    test_writer = tf.summary.FileWriter(logdir="./logs/")
    num_iter = 263
    global_step = 0
    for epoch in range(epochs):
        train_index=np.arange(len(train))
        print("Training Epoch: {}".format(epoch + 1))
        for iteration in range(num_iter):
            global_step += 1
            x_batch,y_batch,train_index=batch(batch_size,train[train_index,:])
            feed_dict_batch={x:x_batch,y:y_batch,learning_rate:2 / global_step}
            _, summary_tr = sess.run([optimizer, merged], feed_dict=feed_dict_batch)
            _, summary_tr1 = sess.run([gradient, merged], feed_dict=feed_dict_batch)
            train_writer.add_summary(summary_tr, global_step)
            train_writer.flush()
            test_writer.add_summary(summary_tr1, global_step)
            test_writer.flush()

            diff_batch,grad_batch=sess.run([difference,gradient],feed_dict=feed_dict_batch)

            W_hid2out_present = sess.run(output_logits,feed_dict=feed_dict_batch)[1]
            W_in2hid_present = sess.run(fc1,feed_dict=feed_dict_batch)[1]
            b_hid2out_present = sess.run(output_logits,feed_dict=feed_dict_batch)[2]
            b_in2hid_present = sess.run(fc1,feed_dict=feed_dict_batch)[2]
            sess.run(W_hid2out_form,feed_dict={W_hid2out_form: W_hid2out_present})
            sess.run(W_in2hid_form,feed_dict={W_in2hid_form: W_in2hid_present})
            sess.run(b_hid2out_form, feed_dict={b_hid2out_form: b_hid2out_present})
            sess.run(b_in2hid_form, feed_dict={b_in2hid_form: b_in2hid_present})

            if (iteration+1) % display_freq == 0:
                loss_batch,acc_batch=sess.run([loss,accuracy],feed_dict=feed_dict_batch)
                print("Iteration: {0:3d}, Loss: {1:.2f}, Training Accuracy: {2:.01%}".
                      format((iteration + 1),loss_batch, acc_batch))
                print('Delta W: {0:.2f}  Gradient: {1:.4f}'.format(diff_batch,grad_batch))
            if diff_batch<10:
                break
        if diff_batch<10:
            break
    print("Global Step: {0:3d}".format(global_step))

    x_test,y_test=test[:,0:6],test[:,6:11]
    x_train, y_train = train[:, 0:6], train[:, 6:11]
    feed_dict_test = {x: x_test, y: y_test}
    feed_dict_train = {x: x_train, y: y_train,learning_rate:2 / global_step}
    loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
    loss_train, acc_train = sess.run([loss, accuracy], feed_dict=feed_dict_train)
    #_,_,_,_, summary_tr1=sess.run([grads_W_in2hid,grads_W_hid2out,grads_b_in2hid,grads_b_hid2out,merged],feed_dict=feed_dict_train)
    #train_writer.add_summary(summary_tr1)
    print('---------------------------------------------------------')
    print("Train loss: {0:.2f}, train accuracy: {1:.01%}".format(loss_train, acc_train))
    print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))
    print('---------------------------------------------------------')
'''
    hidden = sess.run(fc1, feed_dict=feed_dict_train)[0]
    hid_pca = PCA(hidden)
    label = [[], [], [], [], []]
    label_test = [[], [], [], [], []]
    color = ['b', 'g', 'r', 'c', 'm']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #
    #cov=np.cov(hidden)
    #hid_pca=np.linalg.eig(cov)
    #projection=np.dot(hidden,hid_pca.v)
    #hid_pca=PCA(hidden)

    for i in range(5):
        label[i] = y_train[:, i] == 1
        label_test[i] = y_test[:, i] == 1
        ax.scatter(hid_pca.Y[:,0],hid_pca.Y[:,1],hid_pca.Y[:,2],c=color[i])
        for j in range(5):
            entry_train[i].append(sess.run(ratio,feed_dict={x:x_train[label[i]],indice:[j]}))
            entry_test[i].append(sess.run(ratio, feed_dict={x: x_test[label_test[i]],indice:[j]}))
        print(sum(entry_train[i]),sum(entry_test[i]))
    print("Training set confusion matrix:")
    print(entry_train)
    print("Test set confusion matrix:")
    print(entry_test)
    #plt.show()
'''

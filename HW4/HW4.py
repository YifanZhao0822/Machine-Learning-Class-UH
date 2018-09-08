import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib.mlab import PCA
import matplotlib.pyplot as plt
from ops import fc_layer, batch_ae, batch
from mpl_toolkits.mplot3d import Axes3D

count=0
df=pd.read_csv('F:/Files/DataMining/HW4/companylist.csv',
                         usecols=[0],skiprows=[0], header=None)
list=df.values
for i in range(len(list)):
    filename = str(list[i])[2:-2] + '.csv'
    try:
        df = pd.read_csv('F:/Files/DataMining/HW4/data/' + filename,
                 usecols=[2], skiprows=[0, 1, 2, 3, 4, 5, 6], header=None)
    except FileNotFoundError:
        pass
    except pd.io.common.EmptyDataError:
        pass
    if len(df.values)>=5010:
        col=df.values[0:5010]
        col=np.nan_to_num(col)
        m=np.median(col)
        sigma = np.sqrt(np.var(col))
        col=(col-m)/sigma
        if count==0:
            data=col
            count=1
        else:
            data=np.append(data,col,axis=1)

label=np.empty([5000,4])
for j in range(5000):
    if data[j,755]<data[j+10,755]:
        if data[j,756]<data[j+10,756]:
            label[j,:]=[1,0,0,0]
        else:
            label[j, :] = [0, 1, 0, 0]
    else:
        if data[j,756]<data[j+10,756]:
            label[j,:]=[0,0,1,0]
        else:
            label[j, :] = [0, 0, 0, 1]

label_train=label[0:4000,:]
label_test=label[4000:5000,:]
data=data[0:5000,0:755]
data=np.nan_to_num(data)
n_des=data.shape[1]
n_cases=data.shape[0]
data_label=np.append(data,label,axis=1)
train=data[0:4000,:]
test=data[4000:5000,:]
label_train_index=[[],[],[],[],[]]
label_test_index=[[],[],[],[],[]]
for i in range(4):
    label_train_index[i]= label_train[:,i]==1
    label_test_index[i]=label_test[:,i]==1

pca=PCA(data)
co=np.corrcoef(np.transpose(data))
eig=np.linalg.eig(co)

for i in range(n_des):
    if np.sum(eig[0][0:i])>=0.5*np.sum(eig[0]):
        r1=i
        break
for i in range(n_des):
    if np.sum(eig[0][0:i])>=0.75*np.sum(eig[0]):
        r2=i
        break
for i in range(n_des):
    if np.sum(eig[0][0:i])>=0.9*np.sum(eig[0]):
        r3=i
        break
rat=[]
for i in range(755):
    rat.append(np.sum(pca.fracs[0:i]))


epochs=100
test_size=1000
batch_size=100
display_freq=10
h=r3
hh=20
n_classes=4

x = tf.placeholder(tf.float64, shape=[None, n_des], name='X')
y = tf.placeholder(tf.float64, shape=[None, n_des], name='Y')
x1=tf.placeholder(tf.float64, shape=[None, h], name='X1')
z=tf.placeholder(tf.float64, shape=[None, n_classes], name='Z')
learning_rate = tf.placeholder(tf.float64, shape=None, name='epsilon')
W_hid2out_form=tf.Variable(tf.zeros(shape=[h,n_des]),name='W_hid2out_form')
W_in2hid_form=tf.Variable(tf.zeros(shape=[n_des,h]),name='W_in2hid_form')
b_hid2out_form=tf.Variable(tf.zeros(shape=[n_des]),name='b_hid2out_form')
b_in2hid_form=tf.Variable(tf.zeros(shape=[h]),name='b_in2hid_form')
W_hid2out_form=tf.cast(W_hid2out_form,tf.float64)
W_in2hid_form=tf.cast(W_in2hid_form,tf.float64)
b_hid2out_form=tf.cast(b_hid2out_form,tf.float64)
b_in2hid_form=tf.cast(b_in2hid_form,tf.float64)

fc1 = fc_layer(x, h, 'Hidden_layer', use_relu=True)
output_logits = fc_layer(fc1[0], n_des, 'Output_layer', use_relu=False)
fc11 = fc_layer(x1, hh, 'Hidden_layer1', use_relu=True)
output_logits1=fc_layer(fc11[0], n_classes, 'Output_layer1', use_relu=False)

#define the loss defined as mean squared error
with tf.name_scope('MSE'):
    loss = tf.losses.mean_squared_error(labels=y, predictions=output_logits[0])
    tf.summary.scalar('MSE',loss)
loss1 = tf.losses.mean_squared_error(labels=z, predictions=output_logits1[0])
#compute the root MSE
with tf.name_scope('RMSE'):
    rmse=tf.sqrt(loss,name='RMSE')
    tf.summary.scalar('RMSE',rmse)
#define the optimizer
with tf.name_scope('Optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name='Op').minimize(loss)
optimizer1 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name='Op').minimize(loss1)
#compute the norm of the difference between the weights of the former and the present batches
delta_W=tf.square(tf.norm(fc1[1]-W_in2hid_form))+tf.square(tf.norm(output_logits[1]-W_hid2out_form))
delta_b=tf.square(tf.norm(fc1[2]-b_in2hid_form))+tf.square(tf.norm(output_logits[2]-b_hid2out_form))
with tf.name_scope('Delta_Wb'):
    difference=tf.sqrt(delta_W+delta_b,name='Delta_Wb')
    tf.summary.scalar('Delta_Wb',difference)
#compute the gradient
with tf.name_scope('Gradient'):
    gradient=tf.multiply(1/learning_rate,difference,name='Gradient')
    tf.summary.scalar('Gradient', gradient)

#compute the number of correct predictions
correct_prediction = tf.equal(tf.argmax(output_logits1[0], 1), tf.argmax(z, 1), name='co_pre')
#compute the percentage of correct predictions in the whole batch
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64), name='accuracy')

#the variable that signifies the class of denominator
indice=tf.placeholder(tf.int64,shape=[1],name='Index')
#initialization of the two matrices
entry_train=[[], [], [], [], []]
entry_test=[[], [], [], [], []]
#compare the output vector to the given class index and count the numbers in order to compute the numerator
prediction = tf.equal(tf.argmax(output_logits1[0], 1), indice)
#compute the ratio of prediction and the number of cases in one class
ratio = tf.reduce_mean(tf.cast(prediction, tf.float64))

W=[]
b=[]
grmse_train=[]
grmse_test=[]
init_op = tf.global_variables_initializer()
merged=tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init_op)
    train_writer = tf.summary.FileWriter(logdir="./logs/", graph=sess.graph)
    num_iter = 40
    global_step = 0
    for epoch in range(epochs):
        train_index = np.arange(len(train))
        print("Training Epoch: {}".format(epoch + 1))
        for iteration in range(num_iter):
            global_step += 1
            x_batch, train_index = batch_ae(batch_size, train[train_index, :])
            feed_dict_batch = {x: x_batch, y: x_batch, learning_rate: 2/(epoch+1)}
            _, summary_tr = sess.run([optimizer, merged], feed_dict=feed_dict_batch)
            train_writer.add_summary(summary_tr, global_step)

            diff_batch, grad_batch = sess.run([difference, gradient], feed_dict=feed_dict_batch)

            W_hid2out_present = sess.run(output_logits, feed_dict=feed_dict_batch)[1]
            W_in2hid_present = sess.run(fc1, feed_dict=feed_dict_batch)[1]
            b_hid2out_present = sess.run(output_logits, feed_dict=feed_dict_batch)[2]
            b_in2hid_present = sess.run(fc1, feed_dict=feed_dict_batch)[2]
            sess.run(W_hid2out_form, feed_dict={W_hid2out_form: W_hid2out_present})
            sess.run(W_in2hid_form, feed_dict={W_in2hid_form: W_in2hid_present})
            sess.run(b_hid2out_form, feed_dict={b_hid2out_form: b_hid2out_present})
            sess.run(b_in2hid_form, feed_dict={b_in2hid_form: b_in2hid_present})

            if (iteration + 1) % display_freq == 0:
                loss_batch = sess.run(loss, feed_dict=feed_dict_batch)
                print("Iteration: {0:3d}, Loss: {1:.2f}".format((iteration + 1), loss_batch))
                print('Delta W: {0:.2f}  Gradient: {1:.4f}'.format(diff_batch, grad_batch))

        x_train = train[:, 0:755]
        x_test=test[:,0:755]
        feed_dict_train = {x: x_train, y: x_train}
        feed_dict_test = {x: x_test, y: x_test}
        grmse_train.append(sess.run(rmse,feed_dict=feed_dict_train))
        grmse_test.append(sess.run(rmse,feed_dict=feed_dict_test))

    print("Global Step: {0:3d}".format(global_step))

    hidden,W,b=sess.run(fc1,feed_dict={x: train})
    _,W_out,b_out=sess.run(output_logits,feed_dict={x: train})
    hid_test=sess.run(fc1,feed_dict={x:test})[0]
    hidden=np.nan_to_num(hidden)
    hid_test = np.nan_to_num(hid_test)
    hid_pca=PCA(hidden)
    color = ['b', 'g', 'r', 'c']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(n_classes):
        ax.scatter(hid_pca.Y[:, 0], hid_pca.Y[:, 1], hid_pca.Y[:, 2], c=color[i])
    plt.show()

    hid_max=[]
    activity=[]
    class_act=[[],[],[],[],[]]
    for i in range(h):
        hid_max_index=np.argmax(hidden[:,i])
        hid_max.append(np.argwhere(label[hid_max_index,:]))
        activity.append(np.mean(hidden[:,i]))
        for j in range(n_classes):
            #class_max.append(hid_max.count(j))
            class_act[j].append(np.mean(hidden[label_train_index[j],i]))
    hid_max=np.asarray(hid_max)
    #print(hid_max)
    hid_max_sum=[]
    for i in range(n_classes):
        hid_max_sum.append(np.count_nonzero(hid_max==[[i]]))
    print(hid_max_sum)
    print(np.sort(activity))
    plt.plot(np.sort(activity))
    plt.show()


    act_sort=np.argsort(activity)
    inact_sort=(0.05*len(act_sort))
    act_sort=act_sort[int(inact_sort):]
    W_act=np.zeros([n_des,h])
    b_act=np.zeros([h])
    W_act[:,act_sort]=W[:,act_sort]
    b_act[act_sort]=b[act_sort]
    hid_act=sess.run(tf.sigmoid(np.matmul(train,W_act)+b_act))
    out_act=np.matmul(hid_act,W_out)+b_out
    loss_act = sess.run(tf.losses.mean_squared_error(labels=train, predictions=out_act))
    print("Loss after removing inactive nodes: {0:.2f}".format(loss_act))

#plt.plot(grmse_train,c='b')
#plt.plot(grmse_test,c='r')
#plt.show()

epochs=10
with tf.Session() as sess:
    sess.run(init_op)
    num_iter = 40
    global_step = 0
    for epoch in range(epochs):
        train_index=np.arange(len(train))
        print("Training Epoch: {}".format(epoch + 1))
        for iteration in range(num_iter):
            global_step += 1
            x_batch,z_batch,train_index=batch(batch_size,hidden[train_index,:],label_train[train_index,:])
            feed_dict_batch={x1:x_batch,z:z_batch,learning_rate:2 / (epoch+1)}
            sess.run(optimizer1,feed_dict=feed_dict_batch)
            if (iteration + 1) % display_freq == 0:
                loss_batch,accuracy_batch = sess.run([loss1,accuracy], feed_dict=feed_dict_batch)
                print("Iteration: {0:3d}, Loss: {1:.2f}, Accuracy: {2:.01%}".format((iteration + 1), loss_batch,accuracy_batch))
    con_train = [[], [], [], []]
    con_test = [[], [], [], []]
    for i in range(n_classes):
        con_train[i] = label_train[:, i] == 1
        con_test[i] = label_test[:, i] == 1
        for j in range(n_classes):
            entry_train[i].append(sess.run(ratio,feed_dict={x1:hidden[con_train[i]],indice:[j]}))
            entry_test[i].append(sess.run(ratio, feed_dict={x1: hid_test[con_test[i]],indice:[j]}))
        print(sum(entry_train[i]), sum(entry_test[i]))
    print("Training set confusion matrix:")
    print(entry_train)
    print("Test set confusion matrix:")
    print(entry_test)

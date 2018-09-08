import tensorflow as tf
import numpy as np


# weight and bais wrappers
def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01) #generate the normal distributed initialized weights
    return tf.get_variable('W_' + name,
                           dtype=tf.float64,
                           shape=shape,
                           initializer=initer)


def bias_variable(name, shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float64)
    return tf.get_variable('b_' + name,
                           dtype=tf.float64,
                           initializer=initial)


def fc_layer(x, num_units, name, use_relu=True):
    """
    Create a fully-connected layer
    :param x: input from previous layer
    :param num_units: number of hidden units in the fully-connected layer
    :param name: layer name
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    with tf.variable_scope(name):
        in_dim = x.get_shape()[1]
        W = weight_variable(name, shape=[in_dim, num_units])
        b = bias_variable(name, [num_units])
        tf.summary.histogram('W', W)
        tf.summary.histogram('b', b)
        layer = tf.matmul(x, W)
        layer += b
        if use_relu:
            layer = tf.nn.sigmoid(layer)
        return layer,W,b


def fn_layer(x, num_units,W,b, name, use_relu=True):
    with tf.variable_scope(name):
        in_dim=x.get_shape()[1]
        layer=tf.matmul(x, W)
        layer+=b
        if use_relu:
            layer=tf.nn.sigmoid(layer)
        return layer


def batch(batch_size, feature,label):
    index=np.random.choice(a=len(feature),size=batch_size,replace=False)    #generate the random sample
    features=feature[index,:]
    labels=label[index,:]
    rest=np.setdiff1d(np.arange(len(feature)), index)
    return features,labels,rest

def batch_ae(batch_size, dataset):
    index=np.random.choice(a=len(dataset),size=batch_size,replace=False)    #generate the random sample
    results=dataset[index,:]
    rest=np.setdiff1d(np.arange(len(dataset)), index)
    return results,rest
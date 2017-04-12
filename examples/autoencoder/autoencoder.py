import tensorflow as tf

from layers import *

def encoder(input):
    # Create a conv network with 3 conv layers and 1 FC layer
    # Conv 1: filter: [3, 3, 1], stride: [2, 2], relu
    
    # Conv 2: filter: [3, 3, 8], stride: [2, 2], relu
    
    # Conv 3: filter: [3, 3, 8], stride: [2, 2], relu
    
    # FC: output_dim: 100, no non-linearity
    with tf.variable_scope('encoder'):
        net = tf.nn.conv2d(input, tf.get_variable('conv1_weights', [3, 3, 1, 8]), [1, 2, 2, 1], 'SAME')
        net = tf.nn.relu(net + tf.get_variable('conv1_bias', [8]))
        net = tf.nn.conv2d(net, tf.get_variable('conv2_weights', [3, 3, 8, 8]), [1, 2, 2, 1], 'SAME')
        net = tf.nn.relu(net + tf.get_variable('conv2_bias', [8]))
        net = tf.nn.conv2d(net, tf.get_variable('conv3_weights', [3, 3, 8, 8]), [1, 2, 2, 1], 'SAME')
        net = tf.nn.relu(net + tf.get_variable('conv3_bias', [8]))

        size = net.get_shape().as_list()
        batch_size, size = size[0], size[1] * size[2] * size[3]
        net = tf.reshape(net, [batch_size, size])
        net = tf.matmul(net, tf.get_variable('fc_weights', [size, 100])) + tf.get_variable('fc_bias', [100])
    return net


def decoder(input):
    # Create a deconv network with 1 FC layer and 3 deconv layers
    # FC: output dim: 128, relu
    
    # Reshape to [batch_size, 4, 4, 8]
    
    # Deconv 1: filter: [3, 3, 8], stride: [2, 2], relu
    
    # Deconv 2: filter: [8, 8, 1], stride: [2, 2], padding: valid, relu
    
    # Deconv 3: filter: [7, 7, 1], stride: [1, 1], padding: valid, sigmoid
    with tf.variable_scope('decoder'):
        batch_size = input.get_shape().as_list()[0]
        net = tf.matmul(input, tf.get_variable('fc_weights', [batch_size, 128]))
        net = tf.nn.relu(net + tf.get_variable('fc_bias', [128]))
        net = tf.reshape(net, [batch_size, 4, 4, 8])
        net = tf.nn.conv2d_transpose(net, tf.get_variable('conv1_weights', [3, 3, 8, 8]),
                                     [batch_size, 8, 8, 8], [1, 2, 2, 1], 'SAME')
        net = tf.nn.relu(net + tf.get_variable('conv1_bias', [8]))
        net = tf.nn.conv2d_transpose(net, tf.get_variable('conv2_weights', [8, 8, 1, 8]),
                                     [batch_size, 22, 22, 1], [1, 2, 2, 1], 'VALID')
        net = tf.nn.relu(net + tf.get_variable('conv2_bias', [1]))
        net = tf.nn.conv2d_transpose(net, tf.get_variable('conv3_weights', [7, 7, 1, 1]),
                                     [batch_size, 28, 28, 1], [1, 1, 1, 1], 'VALID')
        net = tf.nn.sigmoid(net + tf.get_variable('conv3_bias', [1]))
    return net

def autoencoder(input_shape):
    # Define place holder with input shape
    input = tf.placeholder(tf.float32, input_shape)
    # Define variable scope for autoencoder
    with tf.variable_scope('autoencoder') as scope:
        # Pass input to encoder to obtain encoding
        encoding = encoder(input)

        # Pass encoding into decoder to obtain reconstructed image
        reconstructed = decoder(encoding)
        
        # Return input image (placeholder) and reconstructed image
        return input, reconstructed

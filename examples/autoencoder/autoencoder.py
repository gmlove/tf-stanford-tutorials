import tensorflow as tf

from layers import *

def encoder(input):
    # Create a conv network with 3 conv layers and 1 FC layer
    # Conv 1: filter: [3, 3, 1], stride: [2, 2], relu
    
    # Conv 2: filter: [3, 3, 8], stride: [2, 2], relu
    
    # Conv 3: filter: [3, 3, 8], stride: [2, 2], relu
    
    # FC: output_dim: 100, no non-linearity
    with tf.variable_scope('encoder'):
        net = conv(input, 'conv1', [3, 3, 1], [2, 2])
        net = conv(net, 'conv2', [3, 3, 8], [2, 2])
        net = conv(net, 'conv3', [3, 3, 8], [1, 1])
        # we don't need a fc layer
        # net = fc(net, 'fc', 100, non_linear_fn=None)
    return net


def decoder(input):
    # Create a deconv network with 1 FC layer and 3 deconv layers
    # FC: output dim: 128, relu
    
    # Reshape to [batch_size, 4, 4, 8]
    
    # Deconv 1: filter: [3, 3, 8], stride: [2, 2], relu
    
    # Deconv 2: filter: [8, 8, 1], stride: [2, 2], padding: valid, relu
    
    # Deconv 3: filter: [7, 7, 1], stride: [1, 1], padding: valid, sigmoid
    with tf.variable_scope('decoder'):
        # removed the fc layer and change the network structure to the same as encoder
        net = input
        # batch_size = net.get_shape().as_list()[0]
        # net = fc(net, 'fc', 128)
        # net = tf.reshape(net, [batch_size, 4, 4, 8])
        net = deconv(net, 'deconv1', [3, 3, 8], [1, 1])
        net = deconv(net, 'deconv2', [3, 3, 8], [2, 2])
        net = deconv(net, 'deconv3', [3, 3, 1], [2, 2], non_linear_fn=tf.nn.sigmoid)
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

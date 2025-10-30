import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    z = tf.size(x)
    z = tf.cast(z, tf.float32)
    sum_square = tf.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)/z
    

    
    return tf.math.sqrt(sum_square)


class GaussianNoiseLayer(layers.Layer):
    def __init__(self, stddev, **kwargs):
        super(GaussianNoiseLayer, self).__init__(**kwargs)
        self.stddev = stddev
    
    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.stddev)
            return inputs + noise
        else:
            noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.stddev)
            return inputs + noise
    
    def get_config(self):
        config = super(GaussianNoiseLayer, self).get_config()
        config.update({'noise_var': self.stddev})
        return config


def getnoisevariance(SNR,rate,P=1):
    # the SNR in args[0] is actually EbN0
    snrdB = SNR + 10*np.log10(rate)
    snr = 10.0**(snrdB/10.0)
    #P_avg = 1
    N0 = P/snr
    return (N0/2)

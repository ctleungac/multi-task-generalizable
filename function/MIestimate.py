###### import sys
import sys
assert sys.version_info >= (3, 5)
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import tensorflow_probability as tfp
np.random.seed(42)
tf.random.set_seed(42)

randN_05 = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)


def get_network(dim_x,dim_y):
    tf.keras.backend.clear_session()
    
    input_A = keras.layers.Input(shape=[dim_x+dim_y])
    input_B = keras.layers.Input(shape=[dim_x+dim_y])

    transform = keras.models.Sequential([
    layers.Dense(20, kernel_initializer=randN_05, activation="relu"),
    keras.layers.Dropout(rate=0.3), # To regularize higher dimensionality
#     layers.Dense(30, kernel_initializer=randN_05, activation="relu"),
#     keras.layers.Dropout(rate=0.3), # To regularize higher dimensionality
    layers.Dense(1, kernel_initializer=randN_05, activation=None)])

    output_A = transform(input_A)
    output_B = transform(input_B)
    output_C = tf.reduce_mean(output_A) - tf.math.log(tf.reduce_mean(tf.exp(output_B))) # MINE
    #output_C = tf.reduce_mean(output_A) - tf.reduce_mean(tf.exp(output_B))+1 # MINE-f
    MI_mod = keras.models.Model(inputs=[input_A, input_B], outputs=output_C)
    MI_mod.compile(loss=loss_func, optimizer=keras.optimizers.Nadam(lr=0.001))
    return MI_mod

    

def loss_func(inp, outp):
    '''Calculate the loss: scaled negative estimated mutual information'''
    return -outp

def MINE_ready(x_sample, y_sample):
    x_sample1, x_sample2 = tf.split(x_sample, num_or_size_splits=2)
    y_sample1, y_sample2 = tf.split(y_sample, num_or_size_splits=2)
    
     # Ensure both tensors are of type float32
    x_sample1 = tf.cast(x_sample1, dtype=tf.float32)
    x_sample2 = tf.cast(x_sample2, dtype=tf.float32)
    y_sample1 = tf.cast(y_sample1, dtype=tf.float32)
    y_sample2 = tf.cast(y_sample2, dtype=tf.float32)
    
    joint_sample = tf.concat([x_sample1, y_sample1], axis=1)
    marg_sample = tf.concat([x_sample2, y_sample1], axis=1)
    return joint_sample,marg_sample

def MINE_MI(x_sample,y_sample,total_epochs=50):
    joint_sample,marg_sample = MINE_ready(x_sample,y_sample)
    MI_mod = get_network(x_sample.shape[-1],y_sample.shape[-1])
    MI_mod.compile(loss=loss_func, optimizer=keras.optimizers.Adam(lr=0.001,weight_decay=5e-4))
    history_mi = MI_mod.fit((joint_sample, marg_sample), x_sample[0:int(x_sample.shape[0]/2)], epochs=total_epochs,batch_size=200,verbose=0)
    return -np.log2(np.exp(1))*history_mi.history['loss'][-1],history_mi


def variance_normalization(data, C):
    # Reshape data as before
    data = np.reshape(data, (data.shape[0], int(data.size / data.shape[0])))
    
    # Subtract the mean
    means = np.mean(data, axis=0)
    data = data - means
    
    # Compute the normalization factor
    norm = np.sqrt(np.mean(np.sum(data ** 2, axis=1)))
    
    # Normalize the data
    data *= C / norm
    
    return data, norm ** 2

def global_normalize_mine(data_orig, C, p=2,dim_correction= False):
    # Reshape data as before
    data_orig = np.reshape(data_orig, (data_orig.shape[0], int(data_orig.size / data_orig.shape[0])))
    
    # Subtract the mean
    means = np.mean(data_orig, axis=0)
    data = np.abs(data_orig - means)
    
    # Compute the normalization factor
    # norm = np.sqrt(np.mean(np.sum(data ** 2, axis=1)))
    # linalg.norm(x, ord=None, 
  
    norm = np.sqrt(np.mean(np.sum(data ** 2, axis=1)))
    norm = norm/ np.sqrt(data.shape[1])
    # print(norm)
    # Normalize the data
    
    data = C*(data_orig-means) / norm
    return data


def feature_normalization(data,C):
    data = np.reshape(data, (data.shape[0],int(data.size/data.shape[0])))
    
    means =  np.mean(data, axis=0) # find the mean for each dimension 
    data = data - means # data - means for each dimension
    
    norm = np.tile(np.sqrt(np.mean(data ** 2 ,axis=0)),(data.shape[0],1))
#     norm =  np.sqrt(np.mean(np.sum(sqz,axis=1)))
    normalized_data = C*data / (norm+(0.0000001))
    
    return normalized_data,norm**2


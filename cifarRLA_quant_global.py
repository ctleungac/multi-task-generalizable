#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import pandas as pd
import argparse
import math
import os


# For demonstration only (if you're using it): 
# from matplotlib import pyplot as plt


# ----------------------------
# 1) Define custom layers/classes/functions
# ----------------------------

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

def get_data_phase1(LABEL_FIRST_HALF):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    train_filter = np.where(np.in1d(y_train, LABEL_FIRST_HALF))
    test_filter =  np.where(np.in1d(y_test, LABEL_FIRST_HALF))
    
    x_train = x_train[train_filter]
    y_train = y_train[train_filter]
    x_test = x_test[test_filter]
    y_test = y_test[test_filter]
    
    Y_train = tf.keras.utils.to_categorical(y_train, 10)
    Y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return x_train, x_test, Y_train, Y_test

def getnoisevariance(SNR, rate, P=1):
    """
    Given SNR in dB (as EbN0 actually) and a rate,
    returns variance (stddev^2) of noise.
    """
    snrdB = SNR + 10*np.log10(rate)
    snr = 10.0**(snrdB/10.0)
    # P is the average signal power, default = 1
    N0 = P/snr
    return (N0/2)
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, final_lr, warmup_epochs, total_epochs):
        super(WarmupCosineDecay, self).__init__()
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.warmup_epochs = tf.cast(warmup_epochs, tf.float32)
        # total_epochs is the overall epochs. The decay period is total_epochs - warmup_epochs.
        self.total_decay_epochs = tf.cast(total_epochs - warmup_epochs, tf.float32)
        self.pi = tf.constant(math.pi, dtype=tf.float32)

    def __call__(self, step):
        # Convert the current step (epoch) to float.
        epoch = tf.cast(step, tf.float32)
        # Warmup learning rate: linearly increase from 0 to initial_lr over warmup_epochs.
        warmup_lr = self.initial_lr * (epoch / self.warmup_epochs)
        # Cosine decay learning rate: decay from initial_lr down to final_lr.
        cosine_decay = 0.5 * (1 + tf.cos(self.pi * (epoch - self.warmup_epochs) / self.total_decay_epochs))
        cosine_lr = self.final_lr + (self.initial_lr - self.final_lr) * cosine_decay

        # Use tf.cond to select warmup or decay based on the epoch.
        return tf.cond(epoch < self.warmup_epochs,
                       lambda: warmup_lr,
                       lambda: cosine_lr)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "final_lr": self.final_lr,
            "warmup_epochs": self.warmup_epochs.numpy(),
            "total_decay_epochs": self.total_decay_epochs.numpy(),
        }



def main():
    # ----------------------------
    # Parse arguments
    # ----------------------------
    parser = argparse.ArgumentParser(description="Train with variable SNR settings and lambdas")
    
    # SAMPLE_NUM
    parser.add_argument("--sample_num", type=int, default=15000,
                        help="Number of samples to use for training (default=15000)")
    
    # lambda_list as multiple integers
    # e.g. --lambda_list 0 1 2 5 10
    parser.add_argument("--lambda_list", nargs="+", type=float, default=[0, 0.5, 1, 2, 5, 10],
                        help="List of lambda values (default=[0, 1, 2, 5, 10])")
    
    # SNR_train_list as multiple integers
    # e.g. --snr_train_list 0 5 10 15
    parser.add_argument("--snr_train_list", nargs="+", type=float, default=[0, 5, 10, 15],
                        help="List of training SNR values (default=[0, 5, 10, 15])")
    
    # SNR_test_list can be a range, but we’ll treat it as multiple integers
    # e.g. --snr_test_list -5  -2  1  4  7 ...
    # If you specifically want the default range: np.arange(-5,20,3)
    parser.add_argument("--snr_test_list", nargs="+", type=float,
                        default=list(np.arange(-5, 20, 3)),
                        help="List of testing SNR values (default=-5:3:20)")
    
    args = parser.parse_args()

    SAMPLE_NUM = args.sample_num
    lambda_list = args.lambda_list
    SNR_train_list = args.snr_train_list
    SNR_test_list = args.snr_test_list
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # For demonstration: print them
    print("SAMPLE_NUM=", SAMPLE_NUM)
    print("lambda_list=", lambda_list)
    print("SNR_train_list=", SNR_train_list)
    print("SNR_test_list=", SNR_test_list)

    # ------------------------------------------------
    # The rest of your training script
    # ------------------------------------------------

    # label permutation
    label_permutation = np.arange(10)
    LABEL_FIRST_HALF = label_permutation[:5]

    x_train, x_test, Y_train, Y_test = get_data_phase1(LABEL_FIRST_HALF)

    # Overlapping label sets
    label2_list = np.zeros((6, 5))
    for i in range(6):
        label2_list[i][:] = label_permutation[i:i+5]

    acc_list = []
    datalist = []
    
        
    num_runs = 2  # Number of full re-trainings
    embeddingDim = 80
    batchsize = 256

    # ----------------------------
    # Training Loop
    # ----------------------------
    
    for run_id in range(num_runs):
        print(f"\n=== Starting Run {run_id+1}/{num_runs} ===")
        for snr_train in SNR_train_list:
            print("current training snr:", snr_train)
            for lambda_val in lambda_list:
                noise_sd = getnoisevariance(snr_train, 1)
                print("current lambda:", lambda_val)

                # Clear previous model / graph
                K.clear_session()
                tf.keras.utils.set_random_seed(42)

                # ----------------------------
                # Build the model
                # ----------------------------
                input_layer = layers.Input(shape=(32, 32, 3))
                x = layers.Conv2D(32, (3, 3), padding='same')(input_layer)
                x = layers.LeakyReLU(alpha=0.1)(x)
                x = layers.BatchNormalization()(x)

                x = layers.Conv2D(64, (3, 3), padding='same')(x)
                x = layers.LeakyReLU(alpha=0.1)(x)
                x = layers.BatchNormalization()(x)

                x = layers.MaxPooling2D(pool_size=(2, 2))(x)

                x = layers.Conv2D(128, (3, 3), padding='same')(x)
                x = layers.LeakyReLU(alpha=0.1)(x)
                x = layers.BatchNormalization()(x)

                encoded = layers.GlobalAveragePooling2D()(x)
                #encoded = layers.Dense(256)(x) # shape: (batch_size, 256)


                # Fake quantization
    #             encoder_2_quantized = tf.keras.layers.Lambda(
    #                 lambda x: tf.quantization.fake_quant_with_min_max_vars(x, min=0.0, max=6.0, num_bits=8)
    #             )(maxpool_2)

                encoder_2_quantized = tf.keras.layers.Lambda(lambda x:
                            tf.quantization.fake_quant_with_min_max_vars(x, min=tf.reduce_min(x), max=tf.reduce_max(x), num_bits=8)
                            )(encoded)
#                 encoder_2_quantized = tf.keras.layers.Lambda(
#                     lambda x: tf.quantization.fake_quant_with_min_max_vars_per_channel(
#                         x,
#                         min=tf.reduce_min(x, axis=[0, 1, 2]),
#                         max=tf.reduce_max(x, axis=[0, 1, 2]),
#                         num_bits=8,
#                         narrow_range=False
#                     )
#                )(encoded)

                embedding_network = tf.keras.Model(inputs=input_layer, outputs=encoder_2_quantized)

                # Normalization + Noise
                normalized_x = tf.keras.layers.Lambda(lambda x: K.tanh(x))(encoder_2_quantized)
                noise_layer = GaussianNoiseLayer(stddev=noise_sd)(normalized_x)

                # Classifier head
#                 CE_cnn_1 = layers.Conv2D(32, (3, 3), padding='same', strides=2)(noise_layer)
#                 #CE_cnn_1 = layers.BatchNormalization()(CE_cnn_1)
#                 CE_cnn_1 = layers.Activation('relu')(CE_cnn_1)
                flatten_1 = layers.Flatten()(noise_layer)
                CE_dense_2 = layers.Dense(256)(flatten_1)
                #CE_dense_2 = layers.BatchNormalization()(CE_dense_2)
                CE_dense_2 = layers.LeakyReLU(alpha=0.1)(CE_dense_2)
                CE_output = layers.Dense(10, activation='softmax', name='CE')(CE_dense_2)

                # Reconstruction head
                decoder_input = layers.Dense(16 * 16 * 64)(noise_layer)
                decoder_input = layers.LeakyReLU(alpha=0.1)(decoder_input)
                x = layers.Reshape((16, 16, 64))(decoder_input)
                x = layers.Conv2DTranspose(64, kernel_size=3, padding='same')(x)
                x = layers.LeakyReLU(alpha=0.1)(x)
                x = layers.UpSampling2D((2, 2))(x)
                mse_output = layers.Conv2D(3, kernel_size=3, padding='same', activation='sigmoid', name='mse')(x)  # Reconstruct the image

                # Combine into model
                model = tf.keras.Model(inputs=input_layer, outputs=[CE_output, mse_output])

                # Optimizer and compile
                lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=0.03, decay_steps=10000
                )

#                 lr_schedule = WarmupCosineDecay(initial_lr=0.06, final_lr=0.001, warmup_epochs=5, total_epochs=200)

                opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)

                model.compile(optimizer=opt,
                              loss={'CE': 'categorical_crossentropy', 
                                    'mse': 'mse'},
                              metrics={'CE': 'accuracy', 'mse': tf.keras.metrics.RootMeanSquaredError()},
                              loss_weights=[1, lambda_val])

                # Train first phase
                history = model.fit(
                    x=x_train, 
                    y=(Y_train, x_train),
                    batch_size=batchsize,
                    epochs=200,     # shortened to 2 for demonstration
                    verbose=0,
                    validation_data=(x_test, (Y_test, x_test))
                )

                first_acc = history.history['val_CE_accuracy'][-1]
                val_mse_loss = history.history['val_mse_loss'][-1]

                print("current acc:", first_acc)
                print("training phase 2")

                # Save the embedding network
                #embedding_network.save("embedding_network2.h5")

                # ----------------------------
                # Second training loop
                # ----------------------------
                for i in range(label2_list.shape[0]):
                    print("current overlap:", i)
                    print(label2_list[5 - i])
                    x_train_2, x_test_2, Y_train_2, Y_test_2 = get_data_phase1(label2_list[5 - i])

                    for snr_test in SNR_test_list:
                        print("test snr:", snr_test)
                        noise_sd_test = getnoisevariance(snr_test, 1)

                        # Load embedding network
                        embedding_network_2 = embedding_network#tf.keras.models.load_model("embedding_network2.h5", compile=False)
                        for layer in embedding_network_2.layers:
                            layer.trainable = False

                        # Rebuild new classifier head for second training
                        # We must re-use the 'normalized_x' from the original pipeline
                        input_layer_2  = layers.Input(shape=(32, 32, 3))
                        encoder_1_2 = embedding_network_2(input_layer_2) 
                        normalized_x_2 = tf.keras.layers.Lambda(lambda x: K.tanh(x))(encoder_1_2)
                        noise_layer_2 = GaussianNoiseLayer(stddev=noise_sd_test)(normalized_x_2)

                        # Classifier branch
                        #CE_cnn_2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(noise_layer_2)
                        flatten_2 = layers.Flatten()(noise_layer_2)
                        CE_dense_3 = layers.Dense(256)(flatten_2)
                        CE_dense_3 = layers.LeakyReLU(alpha=0.1)(CE_dense_3)
                        CE_output2 = tf.keras.layers.Dense(10, activation='softmax', name='CE2')(CE_dense_3)


                        #flatten_2 = layers.Flatten()(noise_layer_2)
                        mse2 = layers.Dense(16 * 16 * 64, activation='relu')(flatten_2)  # Expand to a 16x16 feature map with 64 channels
                        mse2 = layers.Reshape((16, 16, 64))(mse2)  # Reshape to (16, 16, 64)
                        mse2 = layers.Conv2DTranspose(64, kernel_size=3, padding='same', activation='relu')(mse2)  # Refine features
                        mse2 = layers.UpSampling2D((2, 2))(mse2)  # Upsample from 16x16 to 32x32
                        mse_output2 = layers.Conv2D(3, kernel_size=3, activation='sigmoid', padding='same', name='mse2')(mse2)

                        reconstructed_model = tf.keras.models.Model(
                            inputs=input_layer_2,
                            outputs=[CE_output2, mse_output2]
                        )

                        # Freeze embedding part
#                         for layer_idx in range(len(embedding_network_2.layers)):
#                             reconstructed_model.layers[layer_idx + 1].trainable = False

                        # Second-phase compile
    #                     lr_schedule_2 = tf.keras.optimizers.schedules.ExponentialDecay(
    #                         initial_learning_rate=3e-2,
    #                         decay_steps=10000,
    #                         decay_rate=0.8
    #                     )

                        lr_schedule_2 = tf.keras.optimizers.schedules.CosineDecay(
                        initial_learning_rate=0.05, decay_steps=20000
                        )

                        opt_2 = tf.keras.optimizers.SGD(learning_rate=lr_schedule_2)

                        reconstructed_model.compile(
                            optimizer=opt_2,
                            loss={'CE2': 'categorical_crossentropy', 'mse2': 'mse'},
                            loss_weights=[1, 0],  # so the 'mse' branch won't affect training
                            metrics={'CE2': 'accuracy', 'mse2': 'mean_squared_error'}
                        )

                        # Fit second phase
                        history1 = reconstructed_model.fit(
                            x=x_train_2,
                            y=(Y_train_2, x_train_2),
                            batch_size=batchsize,
                            epochs=150,  # shortened for demonstration
                            verbose=0,
                            validation_data=(x_test_2, (Y_test_2, x_test_2))
                        )

                        # Evaluate
                        results = reconstructed_model.evaluate(x=x_test_2, y=(Y_test_2, x_test_2), verbose=0)
                        # 'results' structure is [loss_total, CE2_loss, mse_loss, CE2_accuracy, mse_mse]
                        second_accuracy = results[3]
                        second_mse = results[4]

                        savedata = {
                            "lambda": lambda_val,
                            "first lr": 0.05,
                            "first batchsize": 256,
                            "first accuracy": first_acc,
                            "train snr": snr_train,
                            "test snr": snr_test,
                            "second loss (mse)": val_mse_loss,
                            "second accuracy": second_accuracy
                        }

                        datalist.append(savedata)

                        print("final acc:", second_accuracy)

    # ----------------------------
    # 3) Convert results to DataFrame and save to Excel or CSV
    # ----------------------------
    df = pd.DataFrame(datalist)
    # df.to_excel("RLA_cifar_quanti.xlsx", index=False)
    # If you prefer CSV:
    df.to_csv("./results/RLA_cifar_quant_global_snr332.csv", index=False)

    print("Training complete. Results saved to 'results.xlsx'.")


if __name__ == "__main__":
    main()

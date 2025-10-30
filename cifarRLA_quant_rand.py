#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import pandas as pd
import argparse

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
    parser.add_argument("--lambda_list", nargs="+", type=float, default=[0, 1, 2, 5, 10],
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

    # For demonstration: print them
    print("SAMPLE_NUM=", SAMPLE_NUM)
    print("lambda_list=", lambda_list)
    print("SNR_train_list=", SNR_train_list)
    print("SNR_test_list=", SNR_test_list)
    
    num_runs = 5  # Number of full re-trainings
    embeddingDim = 80
    batchsize = 256
    datalist = []

    # ------------------------------------------------
    # The rest of your training script
    # ------------------------------------------------
    
    for run_id in range(num_runs):
        print(f"\n=== Starting Run {run_id+1}/{num_runs} ===")

        # label permutation
        label_permutation = np.random.permutation(10)
        LABEL_FIRST_HALF = label_permutation[:5]

        x_train, x_test, Y_train, Y_test = get_data_phase1(LABEL_FIRST_HALF)

        # Overlapping label sets
        label2_list = np.zeros((6, 5))
        for i in range(6):
            label2_list[i][:] = label_permutation[i:i+5]

        acc_list = []
        datalist = []

        # ----------------------------
        # Training Loop
        # ----------------------------
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
                input_layer  = layers.Input(shape=(32, 32, 3))

                # Encoder
                encoder_1 = layers.Conv2D(96, (3, 3), activation='relu', padding='same')(input_layer)
                encoder_2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_1)
                maxpool_2 = layers.MaxPooling2D((2, 2))(encoder_2)

                # Fake quantization
    #             encoder_2_quantized = tf.keras.layers.Lambda(
    #                 lambda x: tf.quantization.fake_quant_with_min_max_vars(x, min=0.0, max=6.0, num_bits=8)
    #             )(maxpool_2)

                encoder_2_quantized = tf.keras.layers.Lambda(lambda x:
                            tf.quantization.fake_quant_with_min_max_vars(x, min=tf.reduce_min(x), max=tf.reduce_max(x), num_bits=8)
                            )(encoder_2)

                embedding_network = tf.keras.Model(inputs=input_layer, outputs=encoder_2_quantized)

                # Normalization + Noise
                normalized_x = tf.keras.layers.Lambda(lambda x: K.tanh(x))(encoder_2_quantized)
                noise_layer = GaussianNoiseLayer(stddev=noise_sd)(normalized_x)

                # Classifier head
                CE_cnn_1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(noise_layer)
                flatten_1 = layers.Flatten()(CE_cnn_1)
                CE_dense_2 = layers.Dense(256, activation='relu')(flatten_1)
                CE_output = tf.keras.layers.Dense(10, activation='softmax', name='CE')(CE_dense_2)

                # Reconstruction head
                mse_1 = layers.Conv2DTranspose(32, kernel_size=3, activation='relu', padding='same')(noise_layer)
                upsample1 = layers.UpSampling2D((2,2))(mse_1)
                mse_2 = layers.Conv2DTranspose(32, kernel_size=3, activation='relu', padding='same')(upsample1)
                mse_output = layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same', name='mse')(mse_2)

                # Combine into model
                model = tf.keras.Model(inputs=input_layer, outputs=[CE_output, mse_output])

                # Optimizer and compile
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=5e-2,
                    decay_steps=10000,
                    decay_rate=0.8
                )
                opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
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
                print("current acc:", first_acc)
                print("training phase 2")

                # Save the embedding network
                embedding_network.save("embedding_network2.h5")

                # ----------------------------
                # Second training loop
                # ----------------------------
                for i in range(label2_list.shape[0]):
                    print("current overlap:", i)
                    x_train_2, x_test_2, Y_train_2, Y_test_2 = get_data_phase1(label2_list[5 - i])

                    for snr_test in SNR_test_list:
                        print("test snr:", snr_test)
                        noise_sd_test = getnoisevariance(snr_test, 1)

                        # Load embedding network
                        embedding_network_2 = tf.keras.models.load_model("embedding_network2.h5", compile=False)
                        for layer in embedding_network_2.layers:
                            layer.trainable = False

                        # Rebuild new classifier head for second training
                        # We must re-use the 'normalized_x' from the original pipeline
                        input_layer_2  = layers.Input(shape=(32, 32, 3))
                        encoder_1_2 = embedding_network_2(input_layer_2) 
                        normalized_x_2 = tf.keras.layers.Lambda(lambda x: K.tanh(x))(encoder_1_2)
                        noise_layer_2 = GaussianNoiseLayer(stddev=noise_sd_test)(normalized_x_2)

                        # Classifier branch
                        CE_cnn_2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(noise_layer_2)
                        flatten_2 = layers.Flatten()(CE_cnn_2)
                        CE_dense_3 = layers.Dense(256, activation='relu')(flatten_2)
                        CE_output2 = tf.keras.layers.Dense(10, activation='softmax', name='CE2')(CE_dense_3)

                        # Reconstruction branch 
                        mse_1_2 = layers.Conv2DTranspose(32, kernel_size=3, activation='relu', padding='same')(noise_layer_2)
                        upsample1_2 = layers.UpSampling2D((2,2))(mse_1_2)
                        mse_2_2 = layers.Conv2DTranspose(32, kernel_size=3, activation='relu', padding='same')(upsample1_2)
                        mse_output2 = layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same', name='mse')(mse_2_2)

                        reconstructed_model = tf.keras.models.Model(
                            inputs=input_layer_2,
                            outputs=[CE_output2, mse_output2]
                        )

                        # Freeze embedding part
                        for layer_idx in range(len(embedding_network_2.layers)):
                            reconstructed_model.layers[layer_idx + 1].trainable = False

                        # Second-phase compile
                        lr_schedule_2 = tf.keras.optimizers.schedules.ExponentialDecay(
                            initial_learning_rate=3e-2,
                            decay_steps=10000,
                            decay_rate=0.8
                        )
                        opt_2 = tf.keras.optimizers.SGD(learning_rate=lr_schedule_2)

                        reconstructed_model.compile(
                            optimizer=opt_2,
                            loss={'CE2': 'categorical_crossentropy', 'mse': 'mse'},
                            loss_weights=[1, 0],  # so the 'mse' branch won't affect training
                            metrics={'CE2': 'accuracy', 'mse': 'mean_squared_error'}
                        )

                        # Fit second phase
                        history1 = reconstructed_model.fit(
                            x=x_train_2,
                            y=(Y_train_2, x_train_2),
                            batch_size=batchsize,
                            epochs=200,  # shortened for demonstration
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
                            "second loss (mse)": second_mse,
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
    df.to_csv("RLA_cifar_quanti_dynamic.csv", index=False)

    print("Training complete. Results saved to 'results.xlsx'.")


if __name__ == "__main__":
    main()

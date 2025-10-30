#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import pandas as pd
import argparse
import cifar100label
import ssl
import math


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

    (x_train, y_train), (x_test, y_test) =  tf.keras.datasets.cifar100.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    train_filter = np.where(np.in1d(y_train, LABEL_FIRST_HALF))
    test_filter =  np.where(np.in1d(y_test, LABEL_FIRST_HALF))
    
    x_train = x_train[train_filter]
    y_train = y_train[train_filter]
    x_test = x_test[test_filter]
    y_test = y_test[test_filter]
    
    Y_train = tf.keras.utils.to_categorical(y_train, 100)
    Y_test = tf.keras.utils.to_categorical(y_test, 100)
    
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

def datagenerate():
    
    ssl._create_default_https_context = ssl._create_unverified_context
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    
    LABEL_FIRST_HALF =  cifar100label.coarse_id_fine_id[0]
    LABsEL_SECOND_HALF = []
    for i in range(5):
        LABEL_SECOND_HALF.append(cifar100label.coarse_id_fine_id[1][i])

    label_permutation = LABEL_FIRST_HALF + LABEL_SECOND_HALF

    label2_list = np.zeros((6,5))
    for i in range(6):
        label2_list[i][:] = label_permutation[i:i+5]

    print(LABEL_FIRST_HALF)
    print(label2_list)
    
    train_filter = np.where(np.in1d(train_labels, LABEL_FIRST_HALF))
    test_filter =  np.where(np.in1d(test_labels, LABEL_FIRST_HALF))

    x_1train = train_images[train_filter]
    Y_train = train_labels[train_filter]
    x_1test = test_images[test_filter]
    Y_test = test_labels[test_filter]

    Y_1train = tf.keras.utils.to_categorical(Y_train, 100)
    Y_1test = tf.keras.utils.to_categorical(Y_test, 100)

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
    parser.add_argument("--lambda_list", nargs="+", type=float, default=[0,  1, 2, 5, 10],
                        help="List of lambda values (default=[0, 0.5, 1, 2, 5, 10])")
    
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

    # ------------------------------------------------
    # The rest of your training script
    # ------------------------------------------------

    # label permutation

    LABEL_FIRST_HALF =  cifar100label.coarse_id_fine_id[0]
    x_train, x_test, Y_train, Y_test = get_data_phase1(LABEL_FIRST_HALF)
    
    LABEL_SECOND_HALF = []
    for i in range(5):
        LABEL_SECOND_HALF.append(cifar100label.coarse_id_fine_id[i+1][i])

    label_permutation = LABEL_FIRST_HALF + LABEL_SECOND_HALF

    label2_list = np.zeros((6,5))
    for i in range(6):
        label2_list[i][:] = label_permutation[i:i+5]
    
    acc_list = []
    datalist = []

    embeddingDim = 80
    batchsize = 128

    # ----------------------------
    # Training Loop
    # ----------------------------
    num_runs = 3
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
                # Encoder
                input_layer = layers.Input(shape=(32, 32, 3))
                x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
                x = layers.BatchNormalization()(x)
                x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
                x = layers.BatchNormalization()(x)
                x = layers.MaxPooling2D(pool_size=(2, 2))(x)
                x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
                x = layers.BatchNormalization()(x)
                x = layers.GlobalAveragePooling2D()(x)
                encoded = layers.Dense(256, activation='relu')(x) # shape: (batch_size, 256)

                # Fake quantization on the latent vector
#                 encoder_2_quantized = layers.Lambda(
#                     lambda x: tf.quantization.fake_quant_with_min_max_vars(x, min=0.0, max=6.0, num_bits=8)
#                 )(encoded)
                encoder_2_quantized = tf.keras.layers.Lambda(lambda x:
                            tf.quantization.fake_quant_with_min_max_vars(x, min=tf.reduce_min(x), max=tf.reduce_max(x), num_bits=8)
                            )(encoded)

                embedding_network = tf.keras.Model(inputs=input_layer, outputs=encoder_2_quantized)

                # Normalization + Noise
                normalized_x = tf.keras.layers.Lambda(lambda x: K.tanh(x))(encoder_2_quantized)
                noise_layer = GaussianNoiseLayer(stddev=noise_sd)(normalized_x)

                latent_1D = layers.Flatten()(noise_layer)  # forces shape (None, 256) if it was 4D
                # --- DENSE -> RESHAPE ---
                expanded = layers.Dense(8 * 8 * 64, activation='relu')(latent_1D)  # shape (None, 4096)
                reshaped = layers.Reshape((8, 8, 64))(expanded)                    # shape (None, 8, 8, 64)

                # --- CLASSIFIER HEAD ---

                flatten_1 = layers.Flatten()(noise_layer)
                CE_dense_2 = layers.Dense(256)(latent_1D)
                CE_dense_2 = layers.LeakyReLU(alpha=0.1)(CE_dense_2)
                CE_output = layers.Dense(100, activation='softmax', name='CE')(CE_dense_2)


                # Reconstruction head               
                # (None, 8, 8, 64)

                rec = layers.Conv2DTranspose(64, 3, padding='same', activation='relu')(reshaped)
                rec = layers.UpSampling2D((2, 2))(rec)  # 8 -> 16
                rec = layers.Conv2DTranspose(32, 3, padding='same', activation='relu')(rec)
                rec = layers.UpSampling2D((2, 2))(rec)  # 16 -> 32
                mse_output = layers.Conv2D(3, kernel_size=3, activation='sigmoid', padding='same', name='mse')(rec)


                # Combine into model
                model = tf.keras.Model(inputs=input_layer, outputs=[CE_output, mse_output])

                # Optimizer and compile
    #             lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    #                 initial_learning_rate=0.03, decay_steps=20000
    #             )

#                 lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#                 initial_learning_rate=5e-2,
#                 decay_steps=10000,
#                 decay_rate=0.8)
                lr_schedule = WarmupCosineDecay(initial_lr=0.06, final_lr=0.001, warmup_epochs=5, total_epochs=200)

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
                print("current acc:", first_acc)
                print("training phase 2")

                # Save the embedding network
                embedding_network.save("embedding_network_cifar100_2.h5")

                # ----------------------------
                # Second training loop
                # ----------------------------
                for i in range(label2_list.shape[0]):
                    print("current overlap:", i)
                    # print(label2_list[5 - i])
                    x_train_2, x_test_2, Y_train_2, Y_test_2 = get_data_phase1(label2_list[5 - i])

                    for snr_test in SNR_test_list:
                        print("test snr:", snr_test)
                        noise_sd_test = getnoisevariance(snr_test, 1)

                        # Load embedding network
                        embedding_network_2 = embedding_network #tf.keras.models.load_model("embedding_network_100.h5", compile=False)
                        for layer in embedding_network_2.layers:
                            layer.trainable = False


                        # Rebuild new classifier head for second training
                        # We must re-use the 'normalized_x' from the original pipeline
                        input_layer_2  = layers.Input(shape=(32, 32, 3))
                        encoder_1_2 = embedding_network_2(input_layer_2) 
                        normalized_x_2 = layers.Lambda(lambda x: K.tanh(x))(encoder_1_2)
                        noise_layer_2 = GaussianNoiseLayer(stddev=noise_sd_test)(normalized_x_2)

                        # Flatten the noise layer to force a 2D shape (batch, features)
                        latent_1D_2 = layers.Flatten()(noise_layer_2)

                        # Now use a Dense layer to project to the correct number of units (8*8*64 = 4096)
                        expanded2 = layers.Dense(8 * 8 * 64, activation='relu')(latent_1D_2)

                        # Reshape to a 2D spatial map of shape (8, 8, 64)
                        reshaped2 = layers.Reshape((8, 8, 64))(expanded2)

                        # --- Classifier Branch ---
                        flatten_2 = layers.Flatten()(noise_layer_2)
                        CE_dense_3 = layers.Dense(256)(flatten_2)
                        CE_dense_3 = layers.LeakyReLU(alpha=0.1)(CE_dense_3)
                        CE_output2 = layers.Dense(100, activation='softmax', name='CE2')(CE_dense_3)
                        rec2 = layers.Conv2DTranspose(64, 3, padding='same', activation='relu')(reshaped2)
                        rec2 = layers.UpSampling2D((2, 2))(rec2)  # Upsample: 8x8 -> 16x16
                        rec2 = layers.Conv2DTranspose(32, 3, padding='same', activation='relu')(rec2)
                        rec2 = layers.UpSampling2D((2, 2))(rec2)  # Upsample: 16x16 -> 32x32
                        mse_output2 = layers.Conv2D(3, kernel_size=3, activation='sigmoid', padding='same', name='mse2')(rec2)

                        reconstructed_model = tf.keras.models.Model(
                            inputs=input_layer_2,
                            outputs=[CE_output2, mse_output2]
                        )

    #                     # Freeze embedding part
    #                     for layer_idx in range(len(embedding_network_2.layers)):
    #                         reconstructed_model.layers[layer_idx + 1].trainable = False

                        # Second-phase compile
    #                     lr_schedule_2 = tf.keras.optimizers.schedules.ExponentialDecay(
    #                         initial_learning_rate=3e-2,
    #                         decay_steps=10000,
    #                         decay_rate=0.8
    #                     )

    #                     lr_schedule_2 = tf.keras.optimizers.schedules.CosineDecay(
    #                     initial_learning_rate=0.03, decay_steps=20000
    #                     )
                        lr_schedule_2 = tf.keras.optimizers.schedules.ExponentialDecay(
                            initial_learning_rate=3e-2,
                            decay_steps=5000,
                            decay_rate=0.9)

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
                            epochs=100,  # shortened for demonstration
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
    df.to_csv("./results/RLA_cifar100_quant_snr1616.csv", index=False)

    print("Training complete. Results saved to 'RLA_cifar100_quant_global.xlsx'.")


if __name__ == "__main__":
    main()

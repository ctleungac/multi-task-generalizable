#!/usr/bin/env python3

import argparse
import datetime
import os
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers


# ----------------------------
# 1) Define custom layers/classes/functions
# ----------------------------


class GaussianNoiseLayer(layers.Layer):
    """Applies additive white Gaussian noise using the provided standard deviation."""

    def __init__(self, stddev, **kwargs):
        super(GaussianNoiseLayer, self).__init__(**kwargs)
        self.stddev = stddev

    def call(self, inputs, training=None):
        noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.stddev)
        return inputs + noise

    def get_config(self):
        config = super(GaussianNoiseLayer, self).get_config()
        config.update({'noise_var': self.stddev})
        return config


def augment_mnist(image: tf.Tensor) -> tf.Tensor:
    """Performs a lightweight SimCLR-style augmentation for MNIST images."""

    image = tf.reshape(image, [28, 28, 1])
    # Random crop with padding similar to SimCLR's random resized crop
    image = tf.image.resize_with_crop_or_pad(image, 32, 32)
    image = tf.image.random_crop(image, size=[28, 28, 1])
    # Random horizontal flip helps for MNIST even if digits might change orientation
    image = tf.image.random_flip_left_right(image)
    # Slight random brightness for contrastive learning
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return tf.reshape(image, [28 * 28])


def simclr_nt_xent_loss(proj_1: tf.Tensor, proj_2: tf.Tensor, temperature: float = 0.5) -> tf.Tensor:
    """Computes the NT-Xent contrastive loss used by SimCLR."""

    batch_size = tf.shape(proj_1)[0]
    proj_1 = tf.math.l2_normalize(proj_1, axis=1)
    proj_2 = tf.math.l2_normalize(proj_2, axis=1)

    representations = tf.concat([proj_1, proj_2], axis=0)
    similarity_matrix = tf.matmul(representations, representations, transpose_b=True)

    # Remove similarity with itself
    mask = tf.linalg.set_diag(tf.ones_like(similarity_matrix), tf.zeros_like(tf.linalg.diag_part(similarity_matrix)))
    logits = similarity_matrix / temperature
    logits = logits - 1e9 * (1.0 - mask)

    labels = tf.concat([
        tf.range(batch_size, batch_size * 2),
        tf.range(batch_size)
    ], axis=0)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss)


def build_base_encoder(input_shape: Tuple[int, ...]) -> tf.keras.Model:
    """Builds the shared encoder used by both phases."""

    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(128, activation='relu')(input_layer)
    x = tf.keras.layers.Dense(40, activation='relu')(x)
    quantized = tf.keras.layers.Lambda(
        lambda t: tf.quantization.fake_quant_with_min_max_vars(
            t, min=tf.reduce_min(t), max=tf.reduce_max(t), num_bits=8
        )
    )(x)
    return tf.keras.Model(inputs=input_layer, outputs=quantized, name="base_encoder")


def build_projection_head(input_tensor: tf.Tensor) -> tf.Tensor:
    """Creates a simple 2-layer projection head as used in SimCLR."""

    proj = tf.keras.layers.Dense(80, activation='relu')(input_tensor)
    proj = tf.keras.layers.Dense(64)(proj)
    return proj

def get_data_phase1(LABEL_FIRST_HALF):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train / 255.0).astype('float32')
    x_test = (x_test / 255.0).astype('float32')
    
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
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
    
    parser.add_argument("--num_run", type=int, default=3,
                        help="Number of independent runs for the experiment")
    
    args = parser.parse_args()
    
    # # Set the environment variable to make only GPU 0 visible
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    SAMPLE_NUM = args.sample_num
    lambda_list = args.lambda_list
    SNR_train_list = args.snr_train_list
    SNR_test_list = args.snr_test_list
    num_runs = args.num_run

    # For demonstration: print them
    print("SAMPLE_NUM=", SAMPLE_NUM)
    print("lambda_list=", lambda_list)
    print("SNR_train_list=", SNR_train_list)
    print("SNR_test_list=", SNR_test_list)
    
    # num_runs = 5  # Number of full re-trainings
    batchsize = 256
    datalist = []

    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------
    # The rest of your training script
    # ------------------------------------------------
    
    for run_id in range(num_runs):
        print(f"\n=== Starting Run {run_id+1}/{num_runs} ===")

        # label permutation
        label_permutation = np.arange(10)
        LABEL_FIRST_HALF = label_permutation[:5]

        x_train, x_test, Y_train, Y_test = get_data_phase1(LABEL_FIRST_HALF)

        # Overlapping label sets
        label2_list = np.zeros((6, 5))
        for i in range(6):
            label2_list[i][:] = label_permutation[i:i+5]

        # ----------------------------
        # Training Loop
        # ----------------------------
        for snr_train in SNR_train_list:
            print("current training snr:", snr_train)
            for lambda_val in lambda_list:
                # The lambda parameter is kept for logging to mirror the benchmark settings
                noise_sd = getnoisevariance(snr_train, 1)
                print("current lambda:", lambda_val)

                # Clear previous model / graph
                K.clear_session()
                tf.keras.utils.set_random_seed(42)

                # ----------------------------
                # Phase 1: SimCLR pre-training
                # ----------------------------
                base_encoder = build_base_encoder((x_train.shape[1],))

                simclr_input = tf.keras.layers.Input(shape=x_train[0].shape)
                encoded = base_encoder(simclr_input)
                normalized_embed = tf.keras.layers.Lambda(lambda t: tf.math.tanh(t))(encoded)
                noisy_embed = GaussianNoiseLayer(stddev=noise_sd)(normalized_embed)
                projection = build_projection_head(noisy_embed)
                simclr_model = tf.keras.Model(inputs=simclr_input, outputs=projection, name="simclr_model")

                optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
                AUTOTUNE = tf.data.AUTOTUNE

                def _make_dataset(data: np.ndarray) -> tf.data.Dataset:
                    ds = tf.data.Dataset.from_tensor_slices(data.astype(np.float32))
                    ds = ds.shuffle(min(10000, data.shape[0]), seed=run_id)
                    ds = ds.map(lambda sample: (augment_mnist(sample), augment_mnist(sample)),
                                num_parallel_calls=AUTOTUNE)
                    ds = ds.batch(batchsize)
                    ds = ds.prefetch(AUTOTUNE)
                    return ds

                x_phase1 = x_train[:min(SAMPLE_NUM, x_train.shape[0])]
                simclr_dataset = _make_dataset(x_phase1)

                simclr_epochs = 150
                simclr_losses = []

                for epoch in range(simclr_epochs):
                    epoch_losses = []
                    for aug_1, aug_2 in simclr_dataset:
                        with tf.GradientTape() as tape:
                            proj_1 = simclr_model(aug_1, training=True)
                            proj_2 = simclr_model(aug_2, training=True)
                            loss = simclr_nt_xent_loss(proj_1, proj_2)
                        gradients = tape.gradient(loss, simclr_model.trainable_variables)
                        optimizer.apply_gradients(zip(gradients, simclr_model.trainable_variables))
                        epoch_losses.append(float(loss))
                    mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
                    simclr_losses.append(mean_loss)
                    if epoch % 10 == 0:
                        print(f"SimCLR epoch {epoch+1}/{simclr_epochs} - loss: {mean_loss:.4f}")

                pretrain_loss = simclr_losses[-1] if simclr_losses else 0.0
                print(f"Finished SimCLR pre-training with final loss {pretrain_loss:.4f}")

                base_encoder.save("embedding_network2.h5")

                # ----------------------------
                # Phase 2: Downstream classification (benchmark decoder)
                # ----------------------------
                overlap_count = 0
                for i in range(label2_list.shape[0]):
                    print("current overlap:", i)
                    x_train_2, x_test_2, Y_train_2, Y_test_2 = get_data_phase1(label2_list[5 - i])

                    x_train_2 = x_train_2[:min(SAMPLE_NUM, x_train_2.shape[0])]
                    Y_train_2 = Y_train_2[:x_train_2.shape[0]]

                    for snr_test in SNR_test_list:
                        print("test snr:", snr_test)
                        noise_sd_test = getnoisevariance(snr_test, 1)

                        input_layer = tf.keras.layers.Input(shape=x_train[0].shape)
                        embedding_network_2 = tf.keras.models.load_model("embedding_network2.h5", compile=False)
                        embedding_network_2.trainable = False

                        embedded_output = embedding_network_2(input_layer)
                        normalized_embedded = tf.keras.layers.Lambda(lambda t: tf.math.tanh(t))(embedded_output)
                        noise_layer_1 = GaussianNoiseLayer(stddev=noise_sd_test)(normalized_embedded)

                        CE_dense_2 = tf.keras.layers.Dense(40, activation='relu')(noise_layer_1)
                        CE_dense_3 = tf.keras.layers.Dense(40, activation='relu')(CE_dense_2)
                        CE_output2 = tf.keras.layers.Dense(10, activation='softmax', name='CE2')(CE_dense_3)

                        classifier_model = tf.keras.models.Model(inputs=input_layer, outputs=CE_output2)

                        lr_schedule_2 = tf.keras.optimizers.schedules.ExponentialDecay(
                            initial_learning_rate=3e-2, decay_steps=10000, decay_rate=0.8)
                        opt_2 = tf.keras.optimizers.SGD(learning_rate=lr_schedule_2)

                        classifier_model.compile(
                            optimizer=opt_2,
                            loss='categorical_crossentropy',
                            metrics=['accuracy']
                        )

                        history1 = classifier_model.fit(
                            x=x_train_2,
                            y=Y_train_2,
                            batch_size=batchsize,
                            epochs=100,
                            verbose=0,
                            validation_data=(x_test_2, Y_test_2)
                        )

                        final_val_accuracy = history1.history['val_accuracy'][-1]

                        results = classifier_model.evaluate(x=x_test_2, y=Y_test_2, verbose=0)
                        second_accuracy = results[1]

                        savedata = {
                            "lambda": lambda_val,
                            "simclr_final_loss": pretrain_loss,
                            "train snr": snr_train,
                            "test snr": snr_test,
                            "second accuracy": second_accuracy,
                            "val_accuracy_history": history1.history['val_accuracy'],
                            "overlap": overlap_count
                        }

                        datalist.append(savedata)

                        overlap_count += 1

                        print(f"final acc: {second_accuracy:.4f} (val acc final {final_val_accuracy:.4f})")

    # ----------------------------
    # 3) Convert results to DataFrame and save to Excel or CSV
    # ----------------------------
    if datalist:
        df = pd.DataFrame(datalist)
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        filename = f"./results/RLA_mnist_simclr_benchmark_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"Training complete. Results saved to '{filename}'.")
    else:
        print("Training complete but no results were generated.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""SimCLR benchmark on Fashion-MNIST with the same encoder/decoder
architecture used in FmnistRLA_quant.py.

Phase 1 performs SimCLR pre-training on the subset of labels used in the
communication benchmark, ensuring the encoder mirrors the dense-quantized
network. Phase 2 freezes the encoder and trains the downstream classifier
under the same noise/SNR settings as the original benchmark to guarantee a
fair comparison.
"""

import argparse
import os
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K


# ---------------------------------------------------------------------------
# Utility layers and helpers reused from the original benchmark
# ---------------------------------------------------------------------------

class GaussianNoiseLayer(layers.Layer):
    """Gaussian noise layer identical to the benchmark implementation."""

    def __init__(self, stddev: float, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, inputs, training=None):
        del training  # Noise is always injected to match the benchmark script
        noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.stddev)
        return inputs + noise

    def get_config(self):
        config = super().get_config()
        config.update({"noise_var": self.stddev})
        return config


def getnoisevariance(snr_db: float, rate: float, power: float = 1.0) -> float:
    """Compute the noise variance from SNR (in dB) and channel rate."""

    snr_linear = 10.0 ** ((snr_db + 10 * np.log10(rate)) / 10.0)
    noise_psd = power / snr_linear
    return noise_psd / 2.0


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def load_fashion_mnist_subset(labels: Iterable[int], sample_num: int) -> np.ndarray:
    """Return Fashion-MNIST images restricted to the requested labels."""

    (x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
    mask = np.isin(y_train, list(labels))
    x_train = x_train[mask][:sample_num]
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    return x_train


def get_data_phase(labels: Iterable[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_train = (x_train.astype("float32") / 255.0).reshape(len(x_train), -1)
    x_test = (x_test.astype("float32") / 255.0).reshape(len(x_test), -1)

    train_filter = np.where(np.in1d(y_train, labels))
    test_filter = np.where(np.in1d(y_test, labels))

    x_train = x_train[train_filter]
    x_test = x_test[test_filter]
    y_train = y_train[train_filter]
    y_test = y_test[test_filter]

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return x_train, x_test, y_train, y_test


# ---------------------------------------------------------------------------
# Encoder definition (shared between SimCLR and downstream task)
# ---------------------------------------------------------------------------


def build_encoder(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(80, activation="relu")(x)
    x = layers.Lambda(
        lambda t: tf.quantization.fake_quant_with_min_max_vars(
            t, min=0.0, max=6.0, num_bits=8
        )
    )(x)
    return tf.keras.Model(inputs, x, name="benchmark_encoder")


def build_projection_head() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            layers.Dense(80, activation="relu"),
            layers.Dense(64, activation=None),
        ],
        name="projection_head",
    )


# ---------------------------------------------------------------------------
# SimCLR model
# ---------------------------------------------------------------------------


def random_crop(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.resize_with_crop_or_pad(image, 32, 32)
    image = tf.image.random_crop(image, size=[28, 28, 1])
    return image


def augment_image(image: tf.Tensor) -> tf.Tensor:
    image = random_crop(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def augment_batch(images: tf.Tensor) -> tf.Tensor:
    return tf.map_fn(augment_image, images)


def nt_xent_loss(z_i: tf.Tensor, z_j: tf.Tensor, temperature: float = 0.1) -> tf.Tensor:
    batch_size = tf.shape(z_i)[0]
    z = tf.concat([z_i, z_j], axis=0)
    z = tf.math.l2_normalize(z, axis=1)

    logits = tf.matmul(z, z, transpose_b=True) / temperature
    mask = tf.eye(2 * batch_size)
    logits = logits - mask * 1e9  # mask self-similarity

    positives = tf.concat([tf.range(batch_size, 2 * batch_size), tf.range(batch_size)], axis=0)
    loss = tf.keras.losses.sparse_categorical_crossentropy(positives, logits, from_logits=True)
    return tf.reduce_mean(loss)


class SimCLRModel(tf.keras.Model):
    def __init__(self, encoder: tf.keras.Model, projection_head: tf.keras.Model, temperature: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head
        self.temperature = temperature
        self.loss_metric = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            xi = augment_batch(data)
            xj = augment_batch(data)

            xi = tf.reshape(xi, [tf.shape(xi)[0], -1])
            xj = tf.reshape(xj, [tf.shape(xj)[0], -1])

            hi = self.encoder(xi, training=True)
            hj = self.encoder(xj, training=True)

            zi = self.projection_head(hi, training=True)
            zj = self.projection_head(hj, training=True)

            loss = nt_xent_loss(zi, zj, self.temperature)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}


# ---------------------------------------------------------------------------
# Downstream classifier mirroring the benchmark decoder
# ---------------------------------------------------------------------------


def build_classifier(
    encoder: tf.keras.Model,
    noise_std: float,
    learning_rate: float,
) -> tf.keras.Model:
    input_layer = tf.keras.Input(shape=(784,))
    embedded = encoder(input_layer, training=False)
    normalized = layers.Lambda(lambda t: K.tanh(t))(embedded)
    noised = GaussianNoiseLayer(stddev=noise_std)(normalized)

    x = layers.Dense(80, activation="relu")(noised)
    x = layers.Dense(80, activation="relu")(x)
    output = layers.Dense(10, activation="softmax", name="CE")(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=5000,
        decay_rate=0.9,
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# ---------------------------------------------------------------------------
# Main experiment pipeline
# ---------------------------------------------------------------------------


def run_experiment(args: argparse.Namespace):
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "Fmnist_SimCLR_benchmark.csv")

    # Ensure the results file starts fresh on each invocation to avoid mixing
    # with prior runs.
    if os.path.exists(csv_path):
        os.remove(csv_path)

    SAMPLE_NUM = args.sample_num
    lambda_list = args.lambda_list
    SNR_train_list = args.snr_train_list
    SNR_test_list = args.snr_test_list
    num_runs = args.num_run
    batch_size = args.batch_size
    pretrain_epochs = args.pretrain_epochs
    temperature = args.temperature

    print("SAMPLE_NUM=", SAMPLE_NUM)
    print("lambda_list=", lambda_list)
    print("SNR_train_list=", SNR_train_list)
    print("SNR_test_list=", SNR_test_list)
    print("pretrain_epochs=", pretrain_epochs)

    results = []

    for run_id in range(num_runs):
        print(f"\n=== Starting Run {run_id + 1}/{num_runs} ===")

        run_results = []

        label_permutation = np.random.permutation(10)
        label_first_half = label_permutation[:5]

        x_simclr = load_fashion_mnist_subset(label_first_half, SAMPLE_NUM)
        dataset = (
            tf.data.Dataset.from_tensor_slices(x_simclr)
            .shuffle(buffer_size=len(x_simclr))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        encoder = build_encoder(784)
        projector = build_projection_head()
        simclr_model = SimCLRModel(encoder=encoder, projection_head=projector, temperature=temperature)
        simclr_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4))

        simclr_model.fit(dataset, epochs=pretrain_epochs, verbose=1)

        encoder.save("simclr_encoder.h5")

        label_windows = np.zeros((6, 5), dtype=int)
        for i in range(6):
            label_windows[i, :] = label_permutation[i : i + 5]

        for snr_train in SNR_train_list:
            print("Training SNR:", snr_train)
            for lambda_val in lambda_list:
                # Lambda is logged to keep the same experimental grid as
                # FmnistRLA_quant.py. The SimCLR baseline does not include a
                # reconstruction loss, so the value does not influence
                # optimisation but ensures the results tables remain
                # comparable.
                noise_sd = np.sqrt(getnoisevariance(snr_train, rate=1))

                x_train, x_test, y_train, y_test = get_data_phase(label_first_half)

                classifier_encoder = tf.keras.models.load_model("simclr_encoder.h5", compile=False)
                classifier_encoder.trainable = False

                classifier = build_classifier(
                    encoder=classifier_encoder,
                    noise_std=noise_sd,
                    learning_rate=args.decoder_lr,
                )

                history = classifier.fit(
                    x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=args.decoder_epochs,
                    verbose=0,
                    validation_data=(x_test, y_test),
                )

                first_acc = history.history["val_accuracy"][-1]
                print("First-phase validation accuracy:", first_acc)

                for overlap_id in range(label_windows.shape[0]):
                    labels_phase2 = label_windows[5 - overlap_id]
                    x_train_2, x_test_2, y_train_2, y_test_2 = get_data_phase(labels_phase2)

                    for snr_test in SNR_test_list:
                        noise_sd_test = np.sqrt(getnoisevariance(snr_test, rate=1))

                        classifier_encoder_2 = tf.keras.models.load_model("simclr_encoder.h5", compile=False)
                        classifier_encoder_2.trainable = False

                        downstream_model = build_classifier(
                            encoder=classifier_encoder_2,
                            noise_std=noise_sd_test,
                            learning_rate=args.decoder_lr,
                        )

                        downstream_model.fit(
                            x_train_2,
                            y_train_2,
                            batch_size=batch_size,
                            epochs=args.decoder_epochs,
                            verbose=0,
                            validation_data=(x_test_2, y_test_2),
                        )

                        loss, acc = downstream_model.evaluate(
                            x_test_2, y_test_2, verbose=0
                        )

                        results.append(
                            {
                                "run": run_id,
                                "lambda": lambda_val,
                                "train_snr": snr_train,
                                "test_snr": snr_test,
                                "overlap_id": overlap_id,
                                "first_phase_val_accuracy": first_acc,
                                "downstream_accuracy": acc,
                            }
                        )
                        run_results.append(results[-1])

                        print(
                            f"Overlap {overlap_id}, test SNR {snr_test}: accuracy={acc:.4f}"
                        )

        df_run = pd.DataFrame(run_results)
        write_header = not os.path.exists(csv_path)
        df_run.to_csv(csv_path, mode="a", index=False, header=write_header)
        print(f"Saved results for run {run_id + 1} to {csv_path}")

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Saved aggregated results to {csv_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimCLR Fashion-MNIST benchmark")
    parser.add_argument("--sample_num", type=int, default=15000)
    parser.add_argument("--lambda_list", nargs="+", type=float, default=[0, 1, 2, 5, 10])
    parser.add_argument("--snr_train_list", nargs="+", type=float, default=[0, 5, 10, 15])
    parser.add_argument(
        "--snr_test_list",
        nargs="+",
        type=float,
        default=list(np.arange(-5, 20, 3)),
    )
    parser.add_argument("--num_run", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--decoder_epochs", type=int, default=100)
    parser.add_argument("--decoder_lr", type=float, default=5e-2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--gpu", type=str, default="0")
    return parser.parse_args()


def main():
    args = parse_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    tf.keras.utils.set_random_seed(42)
    run_experiment(args)


if __name__ == "__main__":
    main()

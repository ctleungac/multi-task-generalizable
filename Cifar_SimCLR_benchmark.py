#!/usr/bin/env python3
"""CIFAR-10 SimCLR benchmark aligned with cifarRLA_quant_global.py.

This script trains a SimCLR encoder (phase 1) using the same convolutional
backbone as cifarRLA_quant_global.py, then trains a downstream classifier
(phase 2) with the same decoder and optimizer schedule. It supports multiple
runs, configurable SNR noise injection, GPU selection, and saving artifacts
for each run.
"""

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 SimCLR benchmark")
    parser.add_argument("--snr", type=float, default=0.0, help="SNR value in dB")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--phase1_epochs", type=int, default=200, help="SimCLR pretraining epochs")
    parser.add_argument("--phase2_epochs", type=int, default=150, help="Downstream training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    parser.add_argument("--temperature", type=float, default=0.1, help="NT-Xent temperature")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--gpu", type=str, default="0", help="GPU index to use")
    parser.add_argument("--linear_probe", action="store_true", help="Freeze encoder during downstream training")
    parser.add_argument("--output_dir", type=str, default="outputs/cifar_simclr_benchmark", help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras import layers
    import tensorflow.keras.backend as K

    @dataclass
    class RunConfig:
        snr: float
        runs: int
        phase1_epochs: int
        phase2_epochs: int
        batch_size: int
        temperature: float
        seed: int
        gpu: str
        linear_probe: bool
        output_dir: str

    def getnoisevariance(snr_db: float, rate: float, power: float = 1.0) -> float:
        snr_linear = 10.0 ** ((snr_db + 10 * np.log10(rate)) / 10.0)
        noise_psd = power / snr_linear
        return noise_psd / 2.0

    class GaussianNoiseLayer(layers.Layer):
        def __init__(self, stddev: float, **kwargs):
            super().__init__(**kwargs)
            self.stddev = stddev

        def call(self, inputs, training=None):
            del training
            noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.stddev)
            return inputs + noise

        def get_config(self):
            config = super().get_config()
            config.update({"noise_var": self.stddev})
            return config

    def set_determinism(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

    def build_encoder() -> tf.keras.Model:
        inputs = layers.Input(shape=(32, 32, 3))
        x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(64, (3, 3), padding="same")(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = layers.Conv2D(128, (3, 3), padding="same")(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.BatchNormalization()(x)

        x = layers.GlobalAveragePooling2D()(x)

        encoded = layers.Lambda(
            lambda t: tf.quantization.fake_quant_with_min_max_vars(
                t, min=tf.reduce_min(t), max=tf.reduce_max(t), num_bits=8
            )
        )(x)
        return tf.keras.Model(inputs, encoded, name="encoder")

    def build_projection_head() -> tf.keras.Model:
        inputs = layers.Input(shape=(128,))
        x = layers.Dense(256, activation="relu")(inputs)
        x = layers.Dense(128, activation="relu")(x)
        outputs = layers.Dense(64, activation=None)(x)
        return tf.keras.Model(inputs, outputs, name="projection_head")

    def build_classifier_head(encoder: tf.keras.Model, noise_std: float, train_encoder: bool) -> tf.keras.Model:
        encoder.trainable = train_encoder
        inputs = layers.Input(shape=(32, 32, 3))
        features = encoder(inputs)
        normalized = layers.Lambda(lambda t: K.tanh(t))(features)
        noised = GaussianNoiseLayer(stddev=noise_std)(normalized)
        flat = layers.Flatten()(noised)
        x = layers.Dense(256)(flat)
        x = layers.LeakyReLU(alpha=0.1)(x)
        outputs = layers.Dense(10, activation="softmax", name="CE")(x)
        model = tf.keras.Model(inputs, outputs, name="classifier")

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.05,
            decay_steps=20000,
        )
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def simclr_augment(image: tf.Tensor) -> tf.Tensor:
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.1)
        image = tf.image.random_crop(image, size=[28, 28, 3])
        image = tf.image.resize(image, [32, 32])
        make_gray = tf.random.uniform(()) < 0.2
        image = tf.cond(
            make_gray,
            lambda: tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image)),
            lambda: image,
        )
        apply_blur = tf.random.uniform(()) < 0.5
        def blur_fn(img: tf.Tensor) -> tf.Tensor:
            kernel_size = 3
            kernel = tf.ones((kernel_size, kernel_size, 3, 1)) / float(kernel_size * kernel_size)
            return tf.nn.depthwise_conv2d(img[None, ...], kernel, strides=[1, 1, 1, 1], padding="SAME")[0]

        image = tf.cond(apply_blur, lambda: blur_fn(image), lambda: image)
        return tf.clip_by_value(image, 0.0, 1.0)

    def generate_views(image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        view1 = simclr_augment(image)
        view2 = simclr_augment(image)
        return view1, view2

    def nt_xent_loss(z_i: tf.Tensor, z_j: tf.Tensor, temperature: float) -> tf.Tensor:
        batch_size = tf.shape(z_i)[0]
        z = tf.concat([z_i, z_j], axis=0)
        z = tf.math.l2_normalize(z, axis=1)
        logits = tf.matmul(z, z, transpose_b=True) / temperature
        mask = tf.eye(2 * batch_size)
        logits = logits - mask * 1e9
        labels = tf.concat([tf.range(batch_size, 2 * batch_size), tf.range(batch_size)], axis=0)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        return tf.reduce_mean(loss)

    def simclr_pretrain(
        encoder: tf.keras.Model,
        projector: tf.keras.Model,
        dataset: tf.data.Dataset,
        epochs: int,
        temperature: float,
    ) -> List[float]:
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.9, nesterov=True)
        losses: List[float] = []

        for epoch in range(epochs):
            epoch_losses = []
            for batch in dataset:
                with tf.GradientTape() as tape:
                    xi, xj = batch
                    hi = encoder(xi, training=True)
                    hj = encoder(xj, training=True)
                    zi = projector(hi, training=True)
                    zj = projector(hj, training=True)
                    loss = nt_xent_loss(zi, zj, temperature)
                variables = encoder.trainable_variables + projector.trainable_variables
                grads = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(grads, variables))
                epoch_losses.append(loss.numpy())
            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            losses.append(mean_loss)
        return losses

    def finetune_or_linear_probe(
        encoder: tf.keras.Model,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        noise_std: float,
        epochs: int,
        linear_probe: bool,
    ) -> Tuple[tf.keras.Model, Dict[str, List[float]]]:
        classifier = build_classifier_head(encoder, noise_std=noise_std, train_encoder=not linear_probe)
        history = classifier.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0)
        logs = {k: [float(vv) for vv in values] for k, values in history.history.items()}
        return classifier, logs

    def evaluate(model: tf.keras.Model, dataset: tf.data.Dataset) -> Dict[str, float]:
        loss, acc = model.evaluate(dataset, verbose=0)
        return {"loss": float(loss), "accuracy": float(acc)}

    def save_run_artifacts(
        run_dir: Path,
        config: RunConfig,
        seed: int,
        pretrain_losses: List[float],
        downstream_logs: Dict[str, List[float]],
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        encoder: tf.keras.Model,
        classifier: tf.keras.Model,
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump({**asdict(config), "seed": seed}, f, indent=2)

        curve = pd.DataFrame({"phase1_loss": pretrain_losses})
        for key, values in downstream_logs.items():
            curve[f"phase2_{key}"] = pd.Series(values)
        curve.to_csv(run_dir / "training_curves.csv", index=False)

        with (run_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "train": train_metrics,
                    "val": val_metrics,
                    "test": test_metrics,
                },
                f,
                indent=2,
            )

        encoder.save(run_dir / "encoder_final")
        classifier.save(run_dir / "classifier_final")

    def load_cifar10() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        return x_train, y_train, x_test, y_test

    def prepare_simclr_dataset(images: np.ndarray, batch_size: int) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices(images)
        ds = ds.shuffle(buffer_size=len(images), reshuffle_each_iteration=True)
        ds = ds.map(lambda x: generate_views(x), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def prepare_classifier_datasets(
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        val_split = int(0.1 * len(images))
        permutation = np.random.permutation(len(images))
        images = images[permutation]
        labels = labels[permutation]
        x_val, y_val = images[:val_split], labels[:val_split]
        x_train, y_train = images[val_split:], labels[val_split:]
        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(buffer_size=len(x_train))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return train_ds, val_ds

    def print_run_header(run_idx: int, seed: int, config: RunConfig) -> None:
        print(
            f"Run {run_idx + 1}/{config.runs} | seed={seed} | snr={config.snr} | "
            f"phase1_epochs={config.phase1_epochs} | phase2_epochs={config.phase2_epochs} | "
            f"batch_size={config.batch_size} | linear_probe={config.linear_probe}"
        )

    def main(config: RunConfig) -> None:
        x_train, y_train, x_test, y_test = load_cifar10()
        noise_std = getnoisevariance(config.snr, rate=1.0)
        output_root = Path(config.output_dir) / f"snr_{int(config.snr)}"
        all_metrics: List[Dict[str, float]] = []

        for run_idx in range(config.runs):
            seed = config.seed + run_idx
            set_determinism(seed)
            print_run_header(run_idx, seed, config)

            encoder = build_encoder()
            projector = build_projection_head()

            simclr_ds = prepare_simclr_dataset(x_train, config.batch_size)
            pretrain_losses = simclr_pretrain(
                encoder=encoder,
                projector=projector,
                dataset=simclr_ds,
                epochs=config.phase1_epochs,
                temperature=config.temperature,
            )

            train_ds, val_ds = prepare_classifier_datasets(x_train, y_train, config.batch_size)
            classifier, downstream_logs = finetune_or_linear_probe(
                encoder=encoder,
                train_ds=train_ds,
                val_ds=val_ds,
                noise_std=noise_std,
                epochs=config.phase2_epochs,
                linear_probe=config.linear_probe,
            )

            test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(config.batch_size)
            train_metrics = evaluate(classifier, train_ds)
            val_metrics = evaluate(classifier, val_ds)
            test_metrics = evaluate(classifier, test_ds)
            all_metrics.append({"train_acc": train_metrics["accuracy"], "val_acc": val_metrics["accuracy"], "test_acc": test_metrics["accuracy"]})

            run_dir = output_root / f"run_{run_idx + 1:03d}"
            save_run_artifacts(
                run_dir=run_dir,
                config=config,
                seed=seed,
                pretrain_losses=pretrain_losses,
                downstream_logs=downstream_logs,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                encoder=encoder,
                classifier=classifier,
            )

        summary_path = output_root / "summary.csv"
        df = pd.DataFrame(all_metrics)
        metrics = ["train_acc", "val_acc", "test_acc"]
        summary = pd.DataFrame(
            {
                "metric": metrics,
                "mean": [df[m].mean() for m in metrics],
                "std": [df[m].std() for m in metrics],
            }
        )
        summary.to_csv(summary_path, index=False)

    main(
        RunConfig(
            snr=args.snr,
            runs=args.runs,
            phase1_epochs=args.phase1_epochs,
            phase2_epochs=args.phase2_epochs,
            batch_size=args.batch_size,
            temperature=args.temperature,
            seed=args.seed,
            gpu=args.gpu,
            linear_probe=args.linear_probe,
            output_dir=args.output_dir,
        )
    )

from __future__ import annotations

import tensorflow as tf

from ml.config import TrainingConfig


def conv_block(
    x: tf.Tensor,
    filters: int,
    dropout_rate: float,
    regularizer: tf.keras.regularizers.Regularizer,
) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.SpatialDropout2D(dropout_rate)(x)
    return x


def build_basic_cnn(num_classes: int, config: TrainingConfig) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*config.image_size, 3), name="image")
    regularizer = tf.keras.regularizers.L2(1e-4)

    x = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.04),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)

    x = conv_block(x, 32, 0.05, regularizer)
    x = conv_block(x, 64, 0.10, regularizer)
    x = conv_block(x, 128, 0.15, regularizer)
    x = conv_block(x, 256, 0.20, regularizer)

    x = tf.keras.layers.Conv2D(
        384,
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(384, use_bias=False, kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(config.dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="damage_class")(x)

    return tf.keras.Model(inputs, outputs, name="professional_cnn_baseline")


def build_resnet50(num_classes: int, config: TrainingConfig) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*config.image_size, 3), name="image")
    x = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.04),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )
    base_model.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(config.dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="damage_class")(x)

    model = tf.keras.Model(inputs, outputs, name="resnet50_transfer_learning")
    model.base_model = base_model
    return model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc", multi_label=True),
        ],
    )


def unfreeze_resnet_top(model: tf.keras.Model, trainable_layers: int = 40) -> None:
    base_model = getattr(model, "base_model", None)
    if base_model is None:
        raise ValueError("This model does not have a ResNet50 base_model attached.")

    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

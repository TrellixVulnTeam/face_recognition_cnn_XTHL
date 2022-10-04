#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri June 22 16:15:10 2022
training.py python file
@author: Im-Rises
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

if __name__ == "__main__":
    DATASET_PATH = "../datasets/lfw_funneled"
    BATCH_SIZE = 8
    SEED = 123
    w, h, l = 94, 125, 3
    IMG_SHAPE = (h, w, l)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=(h, w),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(h, w),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    class_names = train_ds.class_names
    n_classes = len(class_names)

    # print(class_names)
    print(f"Image dimensions: {w}x{h}x{l}")
    print(f"Number of classes: {n_classes}")

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(8):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[np.argmax(labels[i])])
            plt.axis("off")

    print(
        f"Number of validation batches: {tf.data.experimental.cardinality(train_ds):d}"
    )
    print(f"Number of test batches: {tf.data.experimental.cardinality(val_ds):d}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_ds.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = val_ds.prefetch(buffer_size=AUTOTUNE)

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomTranslation(
                height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode="nearest"
            ),
            tf.keras.layers.RandomBrightness((-0.1, 0.1)),
            # tf.keras.layers.Random
        ]
    )

    for image, _ in train_dataset.take(1):
        plt.figure(figsize=(10, 10))
        first_image = image[0]
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
            plt.imshow(augmented_image[0] / 255)
            plt.axis("off")
    plt.show()

    preprocess_input = tf.keras.applications.resnet50.preprocess_input

    # Change input shape to add image preprocessing
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    y = preprocess_input(inputs)

    # Create ResNet50 model
    resnet = ResNet50(
        input_shape=[h, w, 3],
        weights="imagenet",
        include_top=False,
        classes=n_classes,
    )

    # Freeze layers for transfer learning
    for layer in resnet.layers[:10]:
        layer.trainable = False
    # resnet.trainable = False

    # Add ResNet50 to the final model output
    outputs = resnet(y)

    # Change the ResNet50 output to be the number of class of the dataset
    x = Flatten()(outputs)
    x = Dense(n_classes, activation="softmax")(x)

    # Assemble model
    model = tf.keras.Model(inputs, x)

    LEARNING_RATE = 0.001
    NESTEROV = True
    MOMENTUM = 0.9

    opti = SGD(
        learning_rate=LEARNING_RATE,
        momentum=MOMENTUM,
        nesterov=NESTEROV,
    )

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=opti,
        metrics=["accuracy"],
    )

    model.summary()

    early_stop = EarlyStopping(monitor="val_accuracy", patience=2)
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=50, callbacks=[early_stop]
    )
    # model.save("../models/resnet50_dl_lfw")
    model.save("../models/resnet50_dl_lfw.h5")

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.ylabel("Accuracy")
    plt.ylim([min(plt.ylim()), 1])
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.ylabel("Cross Entropy")
    plt.ylim([0, 1.0])
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    plt.show()

    image_batch, label_batch = val_ds.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch)

    plt.figure(figsize=(10, 10))
    for i in range(8):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].astype("uint8"))
        plt.title(
            f"Real: {class_names[np.argmax(label_batch[i])]}"
            f"\nPred: {class_names[np.argmax(predictions[i])]}"
        )
        plt.axis("off")
    plt.show()

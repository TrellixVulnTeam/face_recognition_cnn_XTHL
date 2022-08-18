#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri August 5 16:46:00 2022
test_prediction.py python file
@author: Im-Rises
description: Deep Learning script using ResNet50 and Transfer Learning for the LFW dataset.
"""

from keras.applications import ResNet50
from matplotlib import pyplot as plt
from tensorflow import keras

from common_functions import *

if __name__ == "__main__":
    parameters = {
        "shape": [80, 80],
        "nbr_classes": 5749,
        "train_path": "../datasets/lfw_funneled/",
        "test_path": "../datasets/lfw_funneled/",
        "batch_size": 8,
        "epochs": 50,
        "number_of_last_layers_trainable": 10,
        "learning_rate": 0.001,
        "nesterov": True,
        "momentum": 0.9,
    }

    model = ResNet50
    preprocess_input = keras.applications.resnet.preprocess_input
    filename = "resnet50"

    train_files, test_files, train_generator, test_generator = get_data(
        preprocess_input=preprocess_input, parameters=parameters
    )

    model = create_model(architecture=model, parameters=parameters)

    history = fit(
        model=model,
        train_generator=train_generator,
        test_generator=test_generator,
        train_files=train_files,
        test_files=test_files,
        parameters=parameters,
    )

    score = evaluation_model(model, test_generator)

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

######################################################################################

# IMSIZE = [80, 80]
# NBCLASSES = 100
# resnet = ResNet50(input_shape=IMSIZE + [3], weights='imagenet', include_top=False)
#
# for layer in resnet.layers:
#     layer.trainable = False
#
# out = resnet.output
#
# x = Flatten()(out)
# x = Dense(NBCLASSES, activation='softmax')(x)
#
# model = Model(inputs=resnet.input, outputs=x)
#
# model.compile(loss="binary_crossentropy",
#               optimizer="adam",
#               metrics=['accuracy'])
#
# model.summary()

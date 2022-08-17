#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri August 5 16:46:00 2022
test_prediction.py python file
@author: Im-Rises
description: Deep Learning script using ResNet50 and Transfer Learning for the LFW dataset.
"""

from keras import Model
from keras.applications import ResNet50
from keras.layers import Flatten, Dense

if __name__ == "__main__":
    IMSIZE = [80, 80]
    NBCLASSES = 100
    resnet = ResNet50(input_shape=IMSIZE + [3], weights='imagenet', include_top=False)

    for layer in resnet.layers:
        layer.trainable = False

    out = resnet.output

    x = Flatten()(out)
    x = Dense(NBCLASSES, activation='softmax')(x)

    model = Model(inputs=resnet.input, outputs=x)

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])

    model.summary()

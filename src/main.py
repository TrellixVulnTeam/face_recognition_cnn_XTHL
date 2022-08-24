#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri June 22 16:15:10 2022
main.py python file
@author: Im-Rises
"""

import user_interface as ui

if __name__ == "__main__":
    PROJECT_NAME = "face_recognition_cnn"
    HAAR_CASCADE_WEIGHTS = (
        "../face_detection_weights/haarcascade_frontalface_default.xml"
    )
    MODEL_PATH = "../models/resnet50_dl_lfw"

    user_interface = ui.UserInterface(PROJECT_NAME, HAAR_CASCADE_WEIGHTS, MODEL_PATH)
    user_interface.start()

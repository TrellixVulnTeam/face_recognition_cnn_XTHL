#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri June 22 16:15:10 2022
main.py python file
@author: Im-Rises
"""

import UserInterface as ui

if __name__ == "__main__":
    PROJECT_NAME = "face_recognition_cnn"
    HAAR_CASCADE_WEIGHTS = (
        "../face_detection_weights/haarcascade_frontalface_default.xml"
    )
    MODEL_PATH = "../models/resnet50_dl_lfw"

    user_interface = ui.UserInterface(PROJECT_NAME, HAAR_CASCADE_WEIGHTS, MODEL_PATH)
    user_interface.start()
    
    # image1 = cv.imread("../datasets/clement/clement (1).jpg", cv.CASCADE_SCALE_IMAGE)
    # image2 = cv.imread("../datasets/clement/clement (2).jpg", cv.CASCADE_SCALE_IMAGE)
    # image3 = cv.imread("../datasets/clement/clement (3).jpg", cv.CASCADE_SCALE_IMAGE)
    # image4 = cv.imread("../datasets/clement/clement (4).jpg", cv.CASCADE_SCALE_IMAGE)
    # # clementList = [image1, image2, image3, image4]
    # model = load_model(MODEL_PATH)
    # predictions = model.predict(image1)
    # for i in predictions:
    #     print(person_dic.id_10_person_dic[np.argmax(i)])

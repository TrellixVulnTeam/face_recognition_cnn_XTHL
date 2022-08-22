#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri June 22 16:15:10 2022
main.py python file
@author: Im-Rises
"""

import cv2 as cv


def facial_detection_and_mark(_image, class_cascade):
    """
    Function to detect and mark faces in an image
    :param _image: image to detect faces in
    :param class_cascade: classifier to use for face detection
    :return: image with marked faces
    """
    imgreturn = _image.copy()
    gray = cv.cvtColor(imgreturn, cv.COLOR_BGR2GRAY)
    faces = class_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE,
    )
    for (x_axis, y_axis, w_axis, h_axis) in faces:
        cv.rectangle(
            imgreturn,
            (x_axis, y_axis),
            (x_axis + w_axis, y_axis + h_axis),
            (0, 255, 0),
            2,
        )
    return imgreturn


def video_detection(haarclass):
    """
    Function to detect faces in a video
    :param haarclass: classifier to use for face detection
    :return:
    """
    webcam = cv.VideoCapture(0)
    if webcam.isOpened():
        while True:
            b_img_ready, imageframe = webcam.read()
            if b_img_ready:
                face = facial_detection_and_mark(imageframe, haarclass)
                cv.imshow("My webcam", face)
            else:
                print("No image available")
            keystroke = cv.waitKey(20)
            if keystroke == 27:
                break

        webcam.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    PROJECT_NAME = "face_recognition_cnn"
    HAAR_CASCADE_WEIGHTS = "faceDetectionWeights/haarcascade_frontalface_default.xml"

    # print(cv.getBuildInformation())
    classCascadefacial = cv.CascadeClassifier(HAAR_CASCADE_WEIGHTS)
    video_detection(classCascadefacial)

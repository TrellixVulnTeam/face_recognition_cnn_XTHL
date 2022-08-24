#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri June 22 16:15:10 2022
main.py python file
@author: Im-Rises
"""

import cv2 as cv
import numpy as np
import tensorflow as tf
from PIL import Image
from numpy import expand_dims

import person_dictionary as person_dic


def facial_detection_and_mark(_image, class_cascade):
    """
    Function to detect and mark faces in an image
    :param _image: image to detect faces in
    :param class_cascade: classifier to use for face detection
    :return: image with marked faces
    """
    frame = _image.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = class_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE,
    )
    for (x_axis, y_axis, width, height) in faces:
        face = crop_face(frame, (x_axis, y_axis, width, height))
        cv.rectangle(
            frame,
            (x_axis, y_axis),
            (x_axis + width, y_axis + height),
            (0, 255, 0),
            2,
        )
        return frame, face
    return frame, None


def crop_face(image, face_coordinates):
    """
    Function to crop the face from an image
    :param image: image to crop the face from
    :param face_coordinates: coordinates of the face to crop
    :return: cropped face
    """
    x_axis, y_axis, width, height = face_coordinates
    face_image = Image.fromarray(
        image[y_axis : y_axis + height, x_axis : x_axis + width]
    ).resize((94, 125))
    face_array = np.asarray(face_image)
    return expand_dims(np.asarray(face_array), 0)


def video_detection(haarclass, classificator, window_name):
    """
    Function to detect faces in a video
    :param haarclass: classifier to use for face detection
    :param classificator: classifier to use for face recognition
    :param window_name: name of the window to display the video in
    :return:
    """
    cam = cv.VideoCapture(0)
    if cam.isOpened():
        face = None
        while True:
            b_img_ready, image_frame = cam.read()
            if b_img_ready:
                camera_image, face = facial_detection_and_mark(image_frame, haarclass)
                cv.imshow(window_name, camera_image)
            else:
                print("No image available")
            keystroke = cv.waitKey(20)
            if keystroke == 27:
                break
            if keystroke == ord("s") and face is not None:
                predict(classificator, face)
            if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
                break
        cam.release()
        cv.destroyAllWindows()


def load_model(model_path):
    """
    Function to load the prediction model with its weights
    :return: loaded model
    """
    return tf.keras.models.load_model(model_path)


def predict(model, face):
    """
    Function to predict the emotion of an image
    :param model: model to use for prediction
    :param face: face to predict the name from
    :return: predicted face
    """
    print(person_dic.id_10_person_dic[np.argmax(model.predict(face))])


if __name__ == "__main__":
    PROJECT_NAME = "face_recognition_cnn"
    HAAR_CASCADE_WEIGHTS = (
        "../face_detection_weights/haarcascade_frontalface_default.xml"
    )
    MODEL_PATH = "../models/resnet50_dl_lfw"

    # print(cv.getBuildInformation())
    classification_model = load_model(MODEL_PATH)
    class_cascadefacial = cv.CascadeClassifier(HAAR_CASCADE_WEIGHTS)
    video_detection(class_cascadefacial, classification_model, PROJECT_NAME)

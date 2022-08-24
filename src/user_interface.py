#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri June 22 16:15:10 2022
user_interface.py python file
@author: Im-Rises
"""

import tkinter as tk

import cv2 as cv
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk


# Put everything in private scope


class UserInterface:
    VIDEO_LABEL_SHAPE = (300, 250)
    IMAGE_FACE_LABEL_SHAPE = (94, 125)

    def __init__(self, window_name, haar_cascade_weights_path, model_path):
        self.camera_label = None
        self.face_label = None
        self.window = self.create_window(window_name)
        self.camera = self.set_up_opencv()
        self.class_cascadefacial = cv.CascadeClassifier(haar_cascade_weights_path)
        self.face = None
        # self.classification_model = self.load_prediction_model(model_path)

    def start(self):
        self.camera_label.after(20, self.show_frames)
        self.window.mainloop()

    def set_init_image_label(self, shape, image_label):
        """
        Set the image label to a blank image. It prevents a bug where the label is
        created without image which consequently create a window to the maximum size.
        :param shape: the shape of the image
        :param image_label: the image label to set
        :return: None
        """
        arr = np.asarray([[255] * shape[0]] * shape[1])
        face_image = Image.fromarray(arr)
        face_image = ImageTk.PhotoImage(face_image)
        image_label.imgtk = face_image
        image_label.configure(image=face_image)

    def show_frames(self):
        b_img_ready, image_frame = self.camera.read()
        if b_img_ready:
            camera_image, self.face = self.facial_detection_and_mark(
                image_frame, self.class_cascadefacial
            )
            cv2image = cv.cvtColor(camera_image, cv.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image).resize(self.VIDEO_LABEL_SHAPE)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)
            self.camera_label.after(20, self.show_frames)

    def create_window(self, window_name):
        # Window creation
        root = tk.Tk()
        root.title(window_name)
        # window_width = 800
        # window_height = 340
        # screen_width = root.winfo_screenwidth()
        # screen_height = root.winfo_screenheight()
        # center_x = int(screen_width / 2 - window_width / 2)
        # center_y = int(screen_height / 2 - window_height / 2)
        # root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        root.resizable(False, False)

        # Frames creation
        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=0)
        frame_camera = self.create_video_frame(root)
        frame_camera.grid(column=0, row=0)
        frame_face_prediction = self.create_prediction_frame(root)
        frame_face_prediction.grid(column=1, row=0)

        return root

    def create_video_frame(self, container):
        frame = tk.Frame(container)
        frame.columnconfigure(0, weight=0)

        self.camera_label = tk.Label(
            frame, width=self.VIDEO_LABEL_SHAPE[0], height=self.VIDEO_LABEL_SHAPE[1]
        )
        self.camera_label.grid(row=0, column=0, columnspan=2, sticky=tk.NSEW)
        self.set_init_image_label(self.VIDEO_LABEL_SHAPE, self.camera_label)

        tk.Label(frame, text="Video capture state:").grid(column=0, row=1, sticky=tk.W)
        tk.Button(frame, text="ON/OFF").grid(column=1, row=1)
        tk.Label(frame, text="Find face for prediction").grid(
            column=0, row=2, sticky=tk.W
        )
        tk.Button(frame, text="Find Face", command=self.put_face_in_label).grid(
            column=1, row=2
        )
        for widget in frame.winfo_children():
            widget.grid(padx=0, pady=3)

        return frame

    def create_prediction_frame(self, container):
        frame = tk.Frame(container)
        frame.columnconfigure(0, weight=0)

        self.face_label = tk.Label(
            frame,
            width=self.IMAGE_FACE_LABEL_SHAPE[0],
            height=self.IMAGE_FACE_LABEL_SHAPE[1],
        )
        self.face_label.grid(row=0, column=0, columnspan=2, sticky=tk.NSEW)
        self.set_init_image_label(self.IMAGE_FACE_LABEL_SHAPE, self.face_label)

        tk.Label(frame, text="Open a file").grid(column=0, row=1, sticky=tk.W)
        tk.Button(frame, text="Open a file").grid(column=1, row=1)
        tk.Label(frame, text="Predict selected face").grid(column=0, row=2, sticky=tk.W)
        tk.Button(frame, text="Predict").grid(column=1, row=2)

        for widget in frame.winfo_children():
            widget.grid(padx=0, pady=3)
        return frame

    def put_face_in_label(self):
        if self.face is not None:
            cv2image = cv.cvtColor(self.face, cv.COLOR_BGR2RGB)
            face_image = Image.fromarray(cv2image).resize(self.IMAGE_FACE_LABEL_SHAPE, Image.ANTIALIAS)
            face_image = ImageTk.PhotoImage(face_image)
            # cv.imshow("image", cv2image)
            self.face_label.imgtk = face_image
            self.face_label.configure(image=face_image)

    def load_prediction_model(self, model_path):
        """
        Function to load the prediction model with its weights
        :return: loaded model
        """
        return tf.keras.models.load_model(model_path)

    def set_up_opencv(self):
        return cv.VideoCapture(0)

    def facial_detection_and_mark(self, _image, class_cascade):
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
            face = self.crop_face(frame, (x_axis, y_axis, width, height))
            cv.rectangle(
                frame,
                (x_axis, y_axis),
                (x_axis + width, y_axis + height),
                (0, 255, 0),
                2,
            )
            return frame, face
        return frame, None

    def crop_face(self, image, face_coordinates):
        """
        Function to crop the face from an image
        :param image: image to crop the face from
        :param face_coordinates: coordinates of the face to crop
        :return: cropped face
        """
        x_axis, y_axis, width, height = face_coordinates
        face_image = image[y_axis: y_axis + height, x_axis: x_axis + width]
        return np.asarray(face_image)

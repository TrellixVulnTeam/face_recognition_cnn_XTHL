#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri June 22 16:15:10 2022
user_interface.py python file
@author: Im-Rises
"""

import tkinter as tk
import tkinter.filedialog as fd
from tkinter.scrolledtext import ScrolledText

import cv2 as cv
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from numpy import expand_dims

import person_dictionary


# pylint: disable=too-many-instance-attributes, too-few-public-methods
class UserInterface:
    """
    Class to create the user interface for the application
    """

    VIDEO_LABEL_SHAPE = (300, 250)
    IMAGE_FACE_LABEL_SHAPE = (94, 125)

    # ------------------------------CONSTRUCTOR------------------------------#
    def __init__(self, window_name, haar_cascade_weights_path, model_path):
        """
        Constructor of the class
        :param window_name: the name of the window
        :param haar_cascade_weights_path: the path of the haar cascade weights
        :param model_path: the path of the model
        :return: None
        """
        self.__scrolled_text_pred = None
        self.__camera_label = None
        self.__face_label = None
        self.__camera = cv.VideoCapture(0)
        self.__video_enabled = False
        self.__class_cascadefacial = cv.CascadeClassifier(haar_cascade_weights_path)
        self.__classification_model = self.__load_prediction_model(model_path)
        self.__window = self.__create_window(window_name)
        self.__face_buffer = None
        self.__predictable_face_buffer = None

    # ------------------------------START METHOD------------------------------#
    def start(self):
        """
        Function to start the application
        :return: None
        """
        self.__camera_label.after(20, self.__show_frames)
        self.__window.mainloop()

    # ------------------------------REFRESH VIDEO METHOD------------------------------#
    def __show_frames(self):
        if self.__video_enabled:
            b_img_ready, image_frame = self.__camera.read()
            if b_img_ready:
                camera_image, self.__face_buffer = self.__facial_detection_and_mark(
                    image_frame
                )
                cv2image = cv.cvtColor(camera_image, cv.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image).resize(self.VIDEO_LABEL_SHAPE)
                imgtk = ImageTk.PhotoImage(image=img)
                self.__camera_label.imgtk = imgtk
                self.__camera_label.configure(image=imgtk)
        self.__camera_label.after(20, self.__show_frames)

    # # ------------------------------INIT OPENCV METHOD------------------------------#
    # def __set_up_opencv(self):
    #     """
    #     Function to set up the opencv __camera
    #     :return: None
    #     """
    #     return cv.VideoCapture(0)

    # ------------------------------LOAD MODEL METHOD------------------------------#
    def __load_prediction_model(self, model_path):
        """
        Function to load the prediction model with its weights
        :return: loaded model
        """
        return tf.keras.models.load_model(model_path)

    # ------------------------------CREATE WINDOW METHOD------------------------------#
    def __create_window(self, window_name):
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
        frame_camera = self.__create_video_frame(root)
        frame_camera.grid(column=0, row=0)
        frame_face_prediction = self.__create_prediction_frame(root)
        frame_face_prediction.grid(column=1, row=0)

        return root

    # ------------------------------CREATE VIDEO FRAME FOR WINDOW------------------------------#
    def __create_video_frame(self, container):
        frame = tk.Frame(container)
        frame.columnconfigure(0, weight=0)

        self.__camera_label = tk.Label(
            frame, width=self.VIDEO_LABEL_SHAPE[0], height=self.VIDEO_LABEL_SHAPE[1]
        )
        self.__camera_label.grid(row=0, column=0, columnspan=2, sticky=tk.NSEW)
        self.__set_init_image_label(self.VIDEO_LABEL_SHAPE, self.__camera_label)

        tk.Label(frame, text="Video capture state:").grid(column=0, row=1, sticky=tk.W)
        tk.Button(frame, text="ON/OFF", command=self.__toggle_video_enabled).grid(
            column=1, row=1
        )
        tk.Label(frame, text="Find face from camera:").grid(
            column=0, row=2, sticky=tk.W
        )
        tk.Button(frame, text="Find Face", command=self.__put_face_in_label).grid(
            column=1, row=2
        )
        for widget in frame.winfo_children():
            widget.grid(padx=0, pady=3)

        return frame

    # ---------------------------CREATE PREDICTION FRAME FOR WINDOW---------------------------#
    def __create_prediction_frame(self, container):
        frame = tk.Frame(container)
        frame.columnconfigure(0, weight=0)

        self.__face_label = tk.Label(
            frame,
            width=self.IMAGE_FACE_LABEL_SHAPE[0],
            height=self.IMAGE_FACE_LABEL_SHAPE[1],
        )
        self.__face_label.grid(row=0, column=0, columnspan=2, sticky=tk.NSEW)
        self.__set_init_image_label(self.IMAGE_FACE_LABEL_SHAPE, self.__face_label)

        self.__scrolled_text_pred = ScrolledText(frame, width=35, height=7)
        self.__scrolled_text_pred.grid(column=0, row=1, columnspan=2, sticky=tk.W)
        tk.Label(frame, text="Open a file:").grid(column=0, row=2, sticky=tk.W)
        tk.Button(
            frame, text="Open a file", command=self.__open_image_with_file_browser
        ).grid(column=1, row=2)
        tk.Label(frame, text="Predict selected face:").grid(
            column=0, row=3, sticky=tk.W
        )
        tk.Button(frame, text="Predict", command=self.__predict).grid(column=1, row=3)

        for widget in frame.winfo_children():
            widget.grid(padx=0, pady=3)
        return frame

    # ----------------------GENERATE EMPTY 2D IMAGE ARRAY (BLACK IMAGE)-----------------------#
    def __set_init_image_label(self, shape, image_label):
        """
        Set the image label to a blank image. It prevents a bug where the label is
        created without image which consequently create a __window to the maximum size.
        :param shape: the shape of the image
        :param image_label: the image label to set
        :return: None
        """
        arr = np.asarray([[255] * shape[0]] * shape[1])
        face_image = Image.fromarray(arr)
        face_image = ImageTk.PhotoImage(face_image)
        image_label.imgtk = face_image
        image_label.configure(image=face_image)

    # ------------------------------SEARCH FACE IN IMAGE------------------------------#
    def __facial_detection_and_mark(self, _image):
        """
        Function to detect and mark faces in an image
        :param _image: image to detect faces in
        :param class_cascade: classifier to use for __face_buffer detection
        :return: image with marked faces
        """
        frame = _image.copy()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = self.__class_cascadefacial.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            # minSize=self.IMAGE_FACE_LABEL_SHAPE,
            flags=cv.CASCADE_SCALE_IMAGE,
        )
        for (x_axis, y_axis, width, height) in faces:
            face = self.__crop_face(_image, (x_axis, y_axis, width, height))
            cv.rectangle(
                frame,
                (x_axis, y_axis),
                (x_axis + width, y_axis + height),
                (0, 255, 0),
                2,
            )
            return frame, face
        return frame, None

    # ------------------------------CROP FACE FROM IMAGE------------------------------#
    def __crop_face(self, image, face_coordinates):
        """
        Function to crop the __face_buffer from an image
        :param image: image to crop the __face_buffer from
        :param face_coordinates: coordinates of the __face_buffer to crop
        :return: cropped __face_buffer
        """
        x_axis, y_axis, width, height = face_coordinates
        face_image = image[y_axis: y_axis + height, x_axis: x_axis + width]
        return np.asarray(face_image)

    # ------------------------------OPEN FILE BROWSER------------------------------#
    def __open_image_with_file_browser(self):
        file_name_path = fd.askopenfilename(title="Select an image")
        face_browser = cv.imread(file_name_path)
        face_browser = self.__facial_detection_and_mark(face_browser)[1]
        cv2image = cv.cvtColor(face_browser, cv.COLOR_BGR2RGB)
        face_image = Image.fromarray(cv2image).resize(
            self.IMAGE_FACE_LABEL_SHAPE, Image.ANTIALIAS
        )
        self.__predictable_face_buffer = face_image
        face_image = ImageTk.PhotoImage(face_image)
        self.__face_label.imgtk = face_image
        self.__face_label.configure(image=face_image)

    def __toggle_video_enabled(self):
        self.__video_enabled = not self.__video_enabled
        if not self.__video_enabled:
            self.__set_init_image_label(self.VIDEO_LABEL_SHAPE, self.__camera_label)

    # ------------------------------PREDICT FACE FROM MODEL------------------------------#
    def __predict(self):
        if self.__predictable_face_buffer is not None:
            prediction = self.__classification_model.predict(
                expand_dims(np.asarray(self.__predictable_face_buffer), 0))
            # print(f"{person_dictionary.id_10_person_dic[np.argmax(prediction)]}")
            ranking = np.argsort(prediction)[0][::-1]
            self.__scrolled_text_pred.delete("1.0", tk.END)
            for i in range(10):
                self.__scrolled_text_pred.insert(
                    tk.END,
                    f"{i + 1} - {person_dictionary.id_10_person_dic[ranking[i]]} : "
                    f"{int(prediction[0][ranking[i]] * 100)}%\n",
                )

    # ------------------------------PUT FACE FROM VIDEO TO LABEL------------------------------#
    def __put_face_in_label(self):
        if self.__face_buffer is not None:
            cv2image = cv.cvtColor(self.__face_buffer, cv.COLOR_BGR2RGB)
            face_image = Image.fromarray(cv2image).resize(
                self.IMAGE_FACE_LABEL_SHAPE, Image.ANTIALIAS
            )
            self.__predictable_face_buffer = face_image
            face_image = ImageTk.PhotoImage(face_image)
            self.__face_label.imgtk = face_image
            self.__face_label.configure(image=face_image)

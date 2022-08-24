import tkinter as tk

import cv2 as cv
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from numpy import expand_dims


# Put everything in private scope

class UserInterface:
    def __init__(self, window_name, haar_cascade_weights_path, model_path):
        self.camera_label = tk.Label
        self.window = self.create_window(window_name)
        self.camera = self.set_up_opencv()
        self.class_cascadefacial = cv.CascadeClassifier(haar_cascade_weights_path)
        # self.classification_model = self.load_prediction_model(model_path)

    def start(self):
        self.camera_label.after(20, self.show_frames)
        self.window.mainloop()

    def show_frames(self):
        cv2image = cv.cvtColor(self.camera.read()[1], cv.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk)
        self.camera_label.after(20, self.show_frames)

    def create_window(self, window_name):
        # Window creation
        root = tk.Tk()
        root.title(window_name)
        window_width = 800
        window_height = 340
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        root.resizable(False, False)

        # Frames creation
        root.columnconfigure(0, weight=4)
        root.columnconfigure(1, weight=4)
        frame_camera = self.create_video_frame(root)
        frame_camera.grid(column=0, row=0)
        frame_face_prediction = self.create_prediction_frame(root)
        frame_face_prediction.grid(column=1, row=0)

        return root

    def create_video_frame(self, container):
        frame = tk.Frame(container)
        frame.columnconfigure(0, weight=1)

        label = tk.Label(container)
        label.grid(row=0, column=0)
        self.camera_label = tk.Label(frame, width=350, height=250)
        self.camera_label.grid(row=0, column=0, sticky=tk.NSEW)

        tk.Label(frame, text='Video capture state:').grid(column=0, row=1, sticky=tk.W)
        tk.Button(frame, text='ON/OFF').grid(column=1, row=1)
        tk.Label(frame, text='Find face for prediction').grid(column=0, row=2, sticky=tk.W)
        tk.Button(frame, text='Find Face').grid(column=1, row=2)

        for widget in frame.winfo_children():
            widget.grid(padx=0, pady=3)

        return frame

    def create_prediction_frame(self, container):
        frame = tk.Frame(container)
        tk.Label(frame, text='Open a file').grid(column=0, row=0, sticky=tk.W)
        tk.Button(frame, text='Open a file').grid(column=1, row=0)
        tk.Label(frame, text='Predict selected face').grid(column=0, row=1, sticky=tk.W)
        tk.Button(frame, text='Predict').grid(column=1, row=1)

        for widget in frame.winfo_children():
            widget.grid(padx=0, pady=3)
        return frame

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
        face_image = Image.fromarray(
            image[y_axis: y_axis + height, x_axis: x_axis + width]
        ).resize((94, 125))
        face_array = np.asarray(face_image)
        return expand_dims(np.asarray(face_array), 0)

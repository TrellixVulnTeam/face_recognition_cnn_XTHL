#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri June 22 16:15:10 2022
ui.py python file
@author: Im-Rises
"""

import tkinter as tk


def create_window(window_name):
    # Window creation
    root = tk.Tk()
    root.title(window_name)
    window_width = 800
    window_height = 600
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    root.resizable(False, False)

    # Frames creation
    root.columnconfigure(0, weight=4)
    root.columnconfigure(1, weight=4)
    frame_camera = create_frames(root)
    frame_camera.grid(column=0, row=0)
    frame_face_prediction = create_prediction_frame(root)
    frame_face_prediction.grid(column=1, row=0)
    root.mainloop()
    # return root


def create_frames(container):
    frame = tk.Frame(container)

    frame.columnconfigure(0, weight=1)

    # Video here

    tk.Label(frame, text='Video capture state:').grid(column=0, row=0, sticky=tk.W)
    tk.Button(frame, text='ON/OFF').grid(column=1, row=0)
    tk.Label(frame, text='Find face for prediction').grid(column=0, row=1, sticky=tk.W)
    tk.Button(frame, text='Find Face').grid(column=1, row=1)

    for widget in frame.winfo_children():
        widget.grid(padx=0, pady=3)

    return frame


def create_prediction_frame(container):
    frame = tk.Frame(container)
    tk.Label(frame, text='Open a file').grid(column=0, row=0, sticky=tk.W)
    tk.Button(frame, text='Open a file').grid(column=1, row=0)
    tk.Label(frame, text='Predict selected face').grid(column=0, row=1, sticky=tk.W)
    tk.Button(frame, text='Predict').grid(column=1, row=1)

    for widget in frame.winfo_children():
        widget.grid(padx=0, pady=3)
    return frame


# def set_up_opencv():
#     return cv.VideoCapture(0)


# def update_video():
#     # cv2image = cv.cvtColor(camera.read()[1], cv.COLOR_BGR2RGB)
#     # img = Image.fromarray(cv2image)
#     # imgtk = tk.ImageTk.PhotoImage(image=img)
#     print("here")


if __name__ == "__main__":
    PROJECT_NAME = "face_recognition_cnn"
    HAAR_CASCADE_WEIGHTS = (
        "../face_detection_weights/haarcascade_frontalface_default.xml"
    )
    MODEL_PATH = "../models/resnet50_dl_lfw"

    window = create_window(PROJECT_NAME)
    # camera = set_up_opencv()

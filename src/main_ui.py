#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri June 22 16:15:10 2022
ui.py python file
@author: Im-Rises
"""

from tkinter import *

import cv2
from PIL import Image, ImageTk


def show_frames():
    cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)

    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)


if __name__ == "__main__":
    win = Tk()

    win.geometry("700x350")
    label = Label(win)
    label.grid(row=0, column=0)
    cap = cv2.VideoCapture(0)

    label.after(20, show_frames)

    show_frames()
    win.mainloop()

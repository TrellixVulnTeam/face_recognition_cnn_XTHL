#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri June 22 16:15:10 2022
main.py python file
@author: Im-Rises
"""

import cv2 as cv

print(cv.getBuildInformation())

webcam = cv.VideoCapture(0)

while webcam.isOpened():
    pass

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri June 22 16:15:10 2022
setup.py python file
@author: Im-Rises
"""

from setuptools import setup

setup(
    name="face_recognition_cnn",
    version="1.0",
    # packages=["face_recognition_cnn"],
    install_requires=[
        "pandas",
        "numpy",
        "pylint",
        "pytest",
        "black==22.6.0",
        "tensorflow==2.9.1",
        "sklearn",
        "keras",
        "matplotlib",
        "opencv-python~=4.5.5.64",
        "Pillow~=9.1.1",
        "setuptools~=41.2.0",
    ],
    url="https://github.com/Im-Rises/face_recognition_cnn",
    license="",
    author="Im-Rises",
    author_email="quentin-morel88@hotmail.com",
    description="Python Face recognition Deep Learning Script with CNN",
)

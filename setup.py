#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri June 22 16:15:10 2022
Setup.py python file
@author: Im-Rises
"""

from setuptools import setup

setup(
    name="face_recognition_cnn",
    version="1.0",
    packages=["face_recognition_cnn"],
    install_requires=["pandas", "numpy", "pylint", "pytest", "black", "tensorflow"],
    url="https://github.com/Im-Rises/face_recognition_cnn",
    license="",
    author="Im-Rises",
    author_email="quentin-morel88@hotmail.com",
    description="Python Face recognition Deep Learning Script with CNN",
)

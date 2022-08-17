#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri August 5 16:46:00 2022
main_ml_version.py python file
@author: Im-Rises
Source: https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html
description: Machine Learning script using SVM classification for the LFW dataset.
"""

from scipy.stats import loguniform
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if __name__ == "__main__":
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_samples, h, w = lfw_people.images.shape
    print(f"LFW description\n- Number of samples: {n_samples}\n- Width: {w}\n- Height: {h}\n")

    X = lfw_people.data
    n_features = X.shape[1]
    print(f"Number of features (number of pixels): {n_features}\n")

    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]
    print(f"Number of classes (number of different persons): {n_classes}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_components = 150
    pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True).fit(X_train)
    eigenfaces = pca.components_.reshape((n_components, h, w))
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    param_grid = {
        "C": loguniform(1e3, 1e5),
        "gamma": loguniform(1e-4, 1e-1),
    }
    clf = RandomizedSearchCV(
        SVC(kernel="rbf", class_weight="balanced"), param_grid, n_iter=10
    )
    clf = clf.fit(X_train_pca, y_train)

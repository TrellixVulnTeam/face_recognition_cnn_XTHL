#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri June 22 16:15:10 2022
remake_dataset.py python file
@author: Im-Rises
"""

import os
import shutil
import tarfile

from PIL import Image


def untar_dataset(tar_file):
    """
    Untar the dataset
    tar_file: path to the tar file
    :return:
    """
    print("Untar dataset...")
    try:
        shutil.rmtree("lfw_funneled")
    except:
        pass
    tar = tarfile.open(tar_file, "r:gz")
    tar.extractall()
    tar.close()


def delete_dirs_lack(dataset_directory, minimum_element_by_sample):
    """
    Delete directories with a lack of number of images
    dataset_directory: path to the dataset
    minimum_element_by_sample: minimum number of images by sample
    :return:
    """
    print("Delete directories with a lack of number of images...")
    dir_list = os.listdir(dataset_directory)
    for dir_name in dir_list:
        dir_path_name = f"{dataset_directory}/{dir_name}"
        try:
            if len(os.listdir(dir_path_name)) < minimum_element_by_sample:
                shutil.rmtree(dir_path_name)
        except:
            print("Error removing directory: " + dir_path_name)


def remove_images_excess(dataset_directory, minimum_element_by_sample):
    """
    Remove images excess of the dataset
    dataset_directory: path to the dataset
    minimum_element_by_sample: minimum number of images by sample
    :return:
    """
    print("Delete images excess of the dataset...")
    dir_list = os.listdir(dataset_directory)
    for dir_name in dir_list:
        dir_path_name = f"{dataset_directory}/{dir_name}"
        try:
            sample_elements_list = os.listdir(dir_path_name)
            n_element_to_delete = len(sample_elements_list) - minimum_element_by_sample
            if n_element_to_delete > 0:
                for i in range(n_element_to_delete):
                    os.remove(f"{dir_path_name}/{sample_elements_list[i]}")
        except:
            print("Error removing directory: " + dir_path_name)


def crop_images(dataset_directory):
    """
    Crop images from the dataset
    dataset_directory: path to the dataset
    :return:
    """
    print("Crop images from the dataset...")
    dir_list = os.listdir(dataset_directory)
    for dir_name in dir_list:
        dir_path_name = f"{dataset_directory}/{dir_name}"
        try:
            for image_name in os.listdir(dir_path_name):
                image_path = f"{dir_path_name}/{image_name}"
                croped_image = Image.open(image_path)
                croped_image = croped_image.crop((78, 70, 172, 195))
                croped_image.save(image_path)
        except:
            print("Error cropping image: " + image_path)


if __name__ == "__main__":
    tar_file = "lfw-funneled.tgz"
    dataset_directory = "lfw_funneled"
    minimum_element_by_sample = 70

    print("Processing dataset...")
    untar_dataset(tar_file)
    delete_dirs_lack(dataset_directory, minimum_element_by_sample)
    remove_images_excess(dataset_directory, minimum_element_by_sample)
    crop_images(dataset_directory)
    print("Processed dataset")

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


def delete_old_files(dataset_directory):
    """
    Delete dataset files
    dataset_directory: path to the dataset
    :return:
    """
    try:
        shutil.rmtree(dataset_directory)
        print("Delete old dataset files...")
    except FileNotFoundError:
        pass


def untar_dataset(tar_file):
    """
    Untar the dataset
    tar_file: path to the tar file
    :return:
    """
    print("Untar dataset...")
    with tarfile.open(tar_file, "r:gz") as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar)
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
        except NotADirectoryError:
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
        except NotADirectoryError:
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
        except NotADirectoryError:
            print("Error cropping image in " + dir_path_name)


if __name__ == "__main__":
    TAR_FILE_NAME = "lfw-funneled.tgz"
    DATASET_DIRECTORY = "lfw_funneled"
    MINIMUM_IMAGES_BY_CLASS = 10

    print("Processing dataset...")
    delete_old_files(DATASET_DIRECTORY)
    untar_dataset(TAR_FILE_NAME)
    delete_dirs_lack(DATASET_DIRECTORY, MINIMUM_IMAGES_BY_CLASS)
    # remove_images_excess(dataset_directory, minimum_element_by_sample)
    crop_images(DATASET_DIRECTORY)
    print("Processed dataset")

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 08:11:14 2021

@author: Chee San
"""

import glob
import os

train_dir = "C:\\Users\\Chee San\\Desktop\\random_forest\\dataset"

if os.path.exists(train_dir + "\\jpg"):
    os.rename(train_dir + "\\jpg", train_dir + "\\train")
    
# Get the class labels
num_class = 2

# Take all the images from the dataset
image_paths = glob.glob(train_dir + "\\train\\*.jpg")

# Variables to keep track
label = 0
i = 0
j = 2800

# Class names
class_names = ['non-oilpalm', 'oilpalm']

# Loop over the class labels
for x in range(1, num_class + 1):
    
    # Create a folder for each class
    os.makedirs(train_dir + "\\train\\" + class_names[label])
    
    # Get the current path
    cur_path = train_dir + "\\train\\" + class_names[label] + "\\"
    
    # Loop over the images in the dataset
    for index, image_path in enumerate(image_paths[i:j], start = 1):
        original_path = image_path
        image_path = image_path.split("\\")
        image_file_name = str(index) + ".jpg"
        os.rename(original_path, cur_path + image_file_name)
        
    i += 2800
    j += 2800
    label += 1
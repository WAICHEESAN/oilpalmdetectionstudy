# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 21:56:30 2021

@author: Chee San
"""

import os, shutil, cv2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.vgg16 import VGG16
from joblib import load

def visualize_predictions_densebase(classifier):
    path = 'C:\\Users\\Chee San\\Desktop\\to_sort\\'
    output_path = 'C:\\Users\\Chee San\\Desktop\\to_sort\\'
    
    files = []
    
    file_names_array = []
    
    for r, d, file_names in os.walk(path):
        for file in file_names:
            if file.lower().endswith(('.jpg')):
                files.append(os.path.join(r, file))
                file_names_array.append(file)

    for f in files:
        img = image.load_img(f, target_size = (224, 224))
        img_tensor = image.img_to_array(img)
        img_tensor /= 255.
        
        densebase = DenseNet121(input_shape=(224,224,3),
                          include_top=False,
                          weights='imagenet')

        densebase.trainable=False
        
        features = densebase.predict(img_tensor.reshape(1, 224, 224, 3))
        
        try:
            prediction = classifier.predict(features)
            
        except:
            prediction = classifier.predict(features.reshape(1, 7*7*1024))
        
        plt.imshow(img_tensor)
        plt.show()
        
        if np.argmax(prediction) == 0:
            print(prediction, 'non-oilpalm')
            if not os.path.exists(output_path + 'non-oilpalm'):
                os.makedirs(output_path + 'non-oilpalm')
            full_output_path = "{op}{lbl}/{fn}".format(op = output_path, 
                                                       lbl = 'non-oilpalm', 
                                                       fn = file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
        
        elif np.argmax(prediction) == 1:
            print(prediction, 'oilpalm')
            if not os.path.exists(output_path + 'oilpalm'):
                os.makedirs(output_path + 'oilpalm')
            full_output_path = "{op}{lbl}/{fn}".format(op = output_path, 
                                                       lbl = 'oilpalm', 
                                                       fn = file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
                
        else:
            print(prediction, 'not-identified')
            if not os.path.exists(output_path + 'not-identified'):
                os.makedirs(output_path + 'not-identified')
            full_output_path = "{op}{lbl}/{fn}".format(op = output_path, 
                                                       lbl = 'not-identified', 
                                                       fn = file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
                
                
model = tf.keras.models.load_model('C:\\Users\\Chee San\\Desktop\\Deep learning samples\\dense_soft_flat_finetuned.h5')
visualize_predictions_densebase(model)

def visualize_predictions_vgg16(classifier):
    path = 'C:\\Users\\Chee San\\Desktop\\to_sort\\'
    output_path = 'C:\\Users\\Chee San\\Desktop\\to_sort\\'
    
    files = []
    
    file_names_array = []
    
    for r, d, file_names in os.walk(path):
        for file in file_names:
            if file.lower().endswith(('.jpg')):
                files.append(os.path.join(r, file))
                file_names_array.append(file)

    for f in files:
        img = image.load_img(f, target_size = (224, 224))
        img_tensor = image.img_to_array(img)
        img_tensor /= 255.
        
        vgg16base = VGG16(input_shape=(224,224,3),
                          include_top=False,
                          weights='imagenet')

        vgg16base.trainable=False
        
        features = vgg16base.predict(img_tensor.reshape(1, 224, 224, 3))
        
        try:
            prediction = classifier.predict(features)
            
        except:
            prediction = classifier.predict(features.reshape(1, 7*7*512))
        
        plt.imshow(img_tensor)
        plt.show()
        
        if np.argmax(prediction) == 0:
            print(prediction, 'non-oilpalm')
            if not os.path.exists(output_path + 'non-oilpalm'):
                os.makedirs(output_path + 'non-oilpalm')
            full_output_path = "{op}{lbl}/{fn}".format(op = output_path, 
                                                       lbl = 'non-oilpalm', 
                                                       fn = file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
        
        elif np.argmax(prediction) == 1:
            print(prediction, 'oilpalm')
            if not os.path.exists(output_path + 'oilpalm'):
                os.makedirs(output_path + 'oilpalm')
            full_output_path = "{op}{lbl}/{fn}".format(op = output_path, 
                                                       lbl = 'oilpalm', 
                                                       fn = file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
                
        else:
            print(prediction, 'not-identified')
            if not os.path.exists(output_path + 'not-identified'):
                os.makedirs(output_path + 'not-identified')
            full_output_path = "{op}{lbl}/{fn}".format(op = output_path, 
                                                       lbl = 'not-identified', 
                                                       fn = file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
                
                
model = tf.keras.models.load_model('C:\\Users\\Chee San\\Desktop\\Deep learning samples\\vgg_soft_flat.h5')
visualize_predictions_vgg16(model)

def visualize_predictions_mobile(classifier):
    path = 'C:\\Users\\Chee San\\Desktop\\to_sort\\'
    output_path = 'C:\\Users\\Chee San\\Desktop\\to_sort\\'
    
    files = []
    
    file_names_array = []
    
    for r, d, file_names in os.walk(path):
        for file in file_names:
            if file.lower().endswith(('.jpg')):
                files.append(os.path.join(r, file))
                file_names_array.append(file)

    for f in files:
        img = image.load_img(f, target_size = (224, 224))
        img_tensor = image.img_to_array(img)
        img_tensor /= 255.
        
        mobilebase = MobileNet(input_shape=(224,224,3),
                          include_top=False,
                          weights='imagenet')

        mobilebase.trainable=False
        
        features = mobilebase.predict(img_tensor.reshape(1, 224, 224, 3))
        
        try:
            prediction = classifier.predict(features)
            
        except:
            prediction = classifier.predict(features.reshape(1, 7*7*512))
        
        plt.imshow(img_tensor)
        plt.show()
        
        if np.argmax(prediction) == 0:
            print(prediction, 'non-oilpalm')
            if not os.path.exists(output_path + 'non-oilpalm'):
                os.makedirs(output_path + 'non-oilpalm')
            full_output_path = "{op}{lbl}/{fn}".format(op = output_path, 
                                                       lbl = 'non-oilpalm', 
                                                       fn = file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
        
        elif np.argmax(prediction) == 1:
            print(prediction, 'oilpalm')
            if not os.path.exists(output_path + 'oilpalm'):
                os.makedirs(output_path + 'oilpalm')
            full_output_path = "{op}{lbl}/{fn}".format(op = output_path, 
                                                       lbl = 'oilpalm', 
                                                       fn = file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
                
        else:
            print(prediction, 'not-identified')
            if not os.path.exists(output_path + 'not-identified'):
                os.makedirs(output_path + 'not-identified')
            full_output_path = "{op}{lbl}/{fn}".format(op = output_path, 
                                                       lbl = 'not-identified', 
                                                       fn = file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
                
                
model = tf.keras.models.load_model('C:\\Users\\Chee San\\Desktop\\Deep learning samples\\mobile_soft_flat.h5')
visualize_predictions_mobile(model)
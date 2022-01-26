# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 22:22:39 2022

@author: Chee San
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from utilize import prep_data, plot_result, plot_confusion_matrix
import tensorflow as tf
import kerastuner as kt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet121

from tensorflow import keras
from keras.layers import Dropout
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

def features_densenet(directory, sample_count, img_width, img_height, outputdim, num_classes):
    datagen = ImageDataGenerator(rescale = 1./255,
                                   )
    features = np.zeros(shape = (sample_count, outputdim, outputdim, 1024))
    labels = np.zeros(shape = (sample_count, num_classes)) # match number of classes

    generator = datagen.flow_from_directory(directory,
                                              target_size = (img_width, img_height),
                                              batch_size = batch_size,
                                              class_mode = 'categorical')

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = densebase.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

"""
DenseNet121 - 7,7,1024
"""
densebase = DenseNet121(input_shape=(224,224,3),
                  include_top=False,
                  weights='imagenet')

densebase.trainable= True
batch_size=10
for layer in densebase.layers[:-10]:
    layer.trainable = False
densebase.summary()

for feature_class in ['oilpalm', 'non-oilpalm']:
    print(feature_class)
    prep_data(directory='C:\\Users\\Chee San\\Desktop\\Deep learning samples\\',
              species=feature_class,
              train_size=2240,
              test_size=560)
    
# Set directories
train_path = 'C:\\Users\\Chee San\\Desktop\\Deep learning samples\\train'
test_path = 'C:\\Users\\Chee San\\Desktop\\Deep learning samples\\test'

train_size = 4480
test_size = 1120

train_features, train_labels = features_densenet(train_path, train_size, 224, 224, 7, 2)
test_features, test_labels = features_densenet(test_path, test_size, 224, 224, 7, 2)

"""
# Hyperparameter tuning
"""
def model_builder(hp):
    input_shape = (7, 7, 1024)
    dense_soft_flat = keras.Sequential()
    dense_soft_flat.add(layers.Flatten(input_shape = input_shape))
    dense_soft_flat.add(Dropout(hp.Float('dropout', min_value = 0.0,
                                                max_value = 0.5, default = 0.5,
                                                step = 0.1)))
    dense_soft_flat.add(layers.Dense(2, activation = 'softmax'))
    dense_soft_flat.summary()
    # compile model
    optimizer = hp.Choice('optimizer', ['adam', 'sgd', 'RMSprop'])
    if optimizer == 'adam':
        learning_rate = hp.Float("lr", min_value = 1e-4, max_value = 1e-2, sampling = "log")
        optimizer = optimizers.Adam(learning_rate)
    elif optimizer == 'sgd':
        learning_rate = hp.Float("lr", min_value = 1e-4, max_value = 1e-2, sampling = "log")
        optimizer = optimizers.SGD(learning_rate)
    elif optimizer == 'RMSprop':
        learning_rate = hp.Float("lr", min_value = 1e-4, max_value = 1e-2, sampling = "log")
        optimizer = optimizers.RMSprop(learning_rate)    
    dense_soft_flat.compile(optimizer = optimizer,
                          loss = 'categorical_crossentropy',
                          metrics = ['accuracy', tf.keras.metrics.Precision(),
                                     tf.keras.metrics.Recall()])
    return dense_soft_flat

tuner = kt.RandomSearch(model_builder,
                     objective = "val_accuracy",
                      directory = 'C:\\Users\Chee San\\Desktop\\test',
                     max_trials = 10, executions_per_trial= 1)

tuner.search(train_features, train_labels, epochs = 10,
             validation_data = (test_features, test_labels), verbose = 0)
best_model = tuner.get_best_models()[0]
best_hyperparameters = tuner.get_best_hyperparameters()[0]
tuner.results_summary(10)










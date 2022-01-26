# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 21:25:58 2021

@author: Chee San
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from utilize import prep_data, plot_result, plot_confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet121

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
# Set the 
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
# DenseNet121 fine-tuned + K-fold + flatten + softmax
"""
epochs = 10
batch_size = 32
num_folds = 10
kfold = KFold(n_splits=num_folds, shuffle=True)

acc_per_fold = []
loss_per_fold = []
precision_per_fold = []
recall_per_fold = []

# set up loop for 10 kfold cross validation
fold_no = 1
# build layers
from datetime import datetime
start = datetime.now()

for train, test in kfold.split(train_features, train_labels):
    dense_soft_flat = models.Sequential()
    dense_soft_flat.add(layers.Flatten(input_shape = (7,7,1024)))
    dense_soft_flat.add(Dropout(0.2))
    dense_soft_flat.add(layers.Dense(2, activation = 'softmax'))
    dense_soft_flat.summary()
    # compile model
    dense_soft_flat.compile(optimizer = optimizers.RMSprop(learning_rate=0.0002),
                          loss = 'categorical_crossentropy',
                          metrics = ['accuracy', tf.keras.metrics.Precision(),
                                     tf.keras.metrics.Recall()])

    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    # run model training
    history_dense_flat = dense_soft_flat.fit(train_features[train], train_labels[train],
                                epochs = epochs,
                                batch_size = batch_size, steps_per_epoch=epochs)
    # print scores for each fold
    scores = dense_soft_flat.evaluate(train_features[test], train_labels[test], verbose = 0)
    print(f'Score for fold {fold_no}: {dense_soft_flat.metrics_names[0]} of {scores[0]}; {dense_soft_flat.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1])
    loss_per_fold.append(scores[0])
    precision_per_fold.append(scores[2])
    recall_per_fold.append(scores[3])

    plot_result(history_dense_flat)

    fold_no = fold_no + 1
duration = datetime.now() - start
print("Training completed in time: ", duration)

# print score for each fold and average with s.d.
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]} - Precision: {precision_per_fold[i]} - Recall: {recall_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} \u00B1 {np.std(acc_per_fold)}')
print(f'> Loss: {np.mean(loss_per_fold)} \u00B1 {np.std(loss_per_fold)}')
print(f'> Precision: {np.mean(precision_per_fold)} \u00B1 {np.std(precision_per_fold)}')
print(f'> Recall: {np.mean(recall_per_fold)} \u00B1 {np.std(recall_per_fold)}')
print('------------------------------------------------------------------------')

#==============================================================================
# Saving Model for Test Set Evaluation
#==============================================================================

dense_soft_flat.save('dense_soft_flat_finetuned.h5')

#==============================================================================
# Evaluate Model's Performance on Test Set
#==============================================================================
y_actual = np.argmax(test_labels, axis = 1)
labels = ['oilpalm', 'non-oilpalm']

dense_soft_flat = tf.keras.models.load_model('dense_soft_flat_finetuned.h5')
score = dense_soft_flat.evaluate(test_features, test_labels)
print("%s: %.2f%%" % (dense_soft_flat.metrics_names[1], score[1]*100))
y_pred = dense_soft_flat.predict(test_features)
y_pred_argmax = np.argmax(y_pred, axis = 1)
confusion_matrix(y_actual, y_pred_argmax)
cm_dense_flat = tf.math.confusion_matrix(labels = y_actual, predictions = y_pred_argmax).numpy()
cm_dense_flat_df = pd.DataFrame(cm_dense_flat,
                               index = labels,
                               columns = labels)

print(classification_report(y_actual, y_pred_argmax, digits = 4))

plot_confusion_matrix(cm=cm_dense_flat, classes=labels, title='DenseNet-Flatten-Softmax-Finetuned')
plt.tight_layout()
plt.savefig('dense_soft_flat_finetuned.png')
plt.clf()
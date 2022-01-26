# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 11:33:05 2021

@author: Chee San
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import mahotas
import cv2
import os
import h5py
import warnings
import glob

from sklearn.model_selection import train_test_split, cross_val_predict, KFold,\
    cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, \
    make_scorer, precision_score, recall_score,f1_score
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

#------------------------------------------------------------------------------
# Feature extraction from the sample images
#------------------------------------------------------------------------------

images_per_class = 2800
fixed_size = tuple((224, 224))

train_path = "C:\\Users\\Chee San\\Desktop\\random_forest\\dataset\\train"

h5_data = "C:\\Users\\Chee San\\Desktop\\random_forest\\output\\data.h5"

h5_labels = "C:\\Users\\Chee San\\Desktop\\random_forest\\output\\labels.h5"

bins = 8

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis = 0)
    return haralick

def fd_histogram(image, mask = None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Get training labels
train_labels = os.listdir(train_path)
train_labels.sort()
print(train_labels)
    
# Create empty lists to hold feature vectors and labels
global_features = []
labels = []

for training_name in train_labels:
    dir = os.path.join(train_path, training_name)
    current_label = training_name
    
    for x in range(1, images_per_class+1):
        file = dir + "/" + str(x) + ".jpg"
        
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)
        
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick = fd_haralick(image)
        fv_histogram = fd_histogram(image)
        
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
        
        labels.append(current_label)
        global_features.append(global_feature)
        
    print("[STATUS] processed folder: {}".format(current_label))
    
print("[STATUS] completed Global Feature Extraction...")

print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

print("[STATUS] training labels {}".format(np.array(labels).shape))

targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print("[Status] training labels encoded...")

scaler = MinMaxScaler(feature_range = (0, 1))
rescaled_features = scaler.fit_transform(global_features)
print("[STATUS] feature vector normalized")

print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))

h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_1', data = np.array(rescaled_features))

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1', data = np.array(target))

h5f_data.close()
h5f_label.close()

print("[STATUS] End of training...")

#------------------------------------------------------------------------------
# Train Random Forest model
#------------------------------------------------------------------------------
num_trees = 10
test_size = 0.20
seed = 9

test_path = "C:\\Users\\Chee San\\Desktop\\random_forest\\dataset\\test"

scoring = {'accuracy':make_scorer(accuracy_score),
           'precision':make_scorer(precision_score),
           'recall':make_scorer(recall_score),
           'f1-score':make_scorer(f1_score)}

# Get the training labels
train_labels = os.listdir(train_path)

# Sort the training labels
train_labels.sort()

if not os.path.exists(test_path):
    os.makedirs(test_path)
    
# Create the machine learning model
models = []
models.append(('RF', RandomForestClassifier(n_estimators = num_trees, random_state = seed)))

# Variables to hold the results and names
results = []
names = []

# Import the feature vector and trained labels
h5f_data = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

# Verify the shape of the feature vector and labels
print("[STATUS] features shape: {}".format(global_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))

print("[STATUS] Training started...")

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(global_features),
                                                                  np.array(global_labels),
                                                                  test_size = test_size,
                                                                  random_state = seed)
print("[STATUS] Generated train and test data...")
print("Train data: {}".format(trainData.shape))
print("Test data: {}".format(testData.shape))
print("Train labels: {}".format(trainLabels.shape))
print("Test labels: {}".format(testLabels.shape))

start = datetime.now()
for name, model in models:
    kfold = KFold(n_splits = 10)
    cv_results = cross_validate(model, trainData, trainLabels, cv = kfold, scoring=scoring)
    print("Accuracy: {} \u00B1 {}".format(cv_results['test_accuracy'].mean(), cv_results['test_accuracy'].std()))
    print("Precision: {} \u00B1 {}".format(cv_results['test_precision'].mean(), cv_results['test_precision'].std()))
    print("Recall: {} \u00B1 {}".format(cv_results['test_recall'].mean(), cv_results['test_recall'].std()))
    print("F1-score: {} \u00B1 {}".format(cv_results['test_f1-score'].mean(), cv_results['test_f1-score'].std()))
    duration = datetime.now() - start
    print("Training completed in time: {}".format(duration))
#--------------------------------------------------------------------------------------------------------------------
# Test the model
#--------------------------------------------------------------------------------------------------------------------
 
rf_classifier = RandomForestClassifier(n_estimators = num_trees, random_state = seed)

# On train set
rf_pred = cross_val_predict(rf_classifier, trainData, trainLabels, cv = 10)
print("Accuracy: {} \u00B1 {}".format(accuracy_score(trainLabels, rf_pred).mean(), accuracy_score(trainLabels, rf_pred).std()))
print("Precision: {} \u00B1 {}".format(precision_score(trainLabels, rf_pred).mean(), precision_score(trainLabels, rf_pred).std()))
print("Recall: {} \u00B1 {}".format(recall_score(trainLabels, rf_pred).mean(), recall_score(trainLabels, rf_pred).std()))
print("F1-score: {} \u00B1 {}".format(f1_score(trainLabels, rf_pred).mean(), f1_score(trainLabels, rf_pred).std()))
print(classification_report(trainLabels, rf_pred, digits = 4))
print(confusion_matrix(trainLabels, rf_pred))

# On test set
rf_test_pred = cross_val_predict(rf_classifier, testData, testLabels, cv = 10)
print("Accuracy: {} \u00B1 {}".format(accuracy_score(testLabels, rf_test_pred).mean(), accuracy_score(testLabels, rf_test_pred).std()))
print("Precision: {} \u00B1 {}".format(precision_score(testLabels, rf_test_pred).mean(), precision_score(testLabels, rf_test_pred).std()))
print("Recall: {} \u00B1 {}".format(recall_score(testLabels, rf_test_pred).mean(), recall_score(testLabels, rf_test_pred).std()))
print("F1-score: {} \u00B1 {}".format(f1_score(testLabels, rf_test_pred).mean(), f1_score(testLabels, rf_test_pred).std()))
print(classification_report(testLabels, rf_test_pred, digits = 4))
print(confusion_matrix(testLabels, rf_test_pred))

cfm = [[514, 32],
       [30, 544]]
classes = ["non-oil palm", "oil palm"]

df_cfm = pd.DataFrame(cfm, index = classes, columns = classes)
plt.figure(figsize = (10,7))
plt.title('Random forest')
plt.xlabel('Predicted label')
plt.ylabel('True label')
cfm_plot = sn.heatmap(df_cfm, annot=True, cmap = 'Purples', center = True, fmt = 'g')
sn.set(font_scale = 8.0)
cfm_plot.figure.savefig("cfm.png")

#------------------------------------------------------------------------------
# To visualize results
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

# create the model - Random Forests
clf  = RandomForestClassifier(n_estimators=num_trees, random_state=seed)

# fit the training data to the model
clf.fit(trainData, trainLabels)

output_path = 'C:\\Users\\Chee San\\Desktop\\to_sort\\'
# loop through the test images
for file in glob.glob(output_path + "/*.jpg"):
    # read the image
    image = cv2.imread(file)

    # resize the image
    image = cv2.resize(image, fixed_size)

    ####################################
    # Global Feature extraction
    ####################################
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)

    ###################################
    # Concatenate global features
    ###################################
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

    # scale features in the range (0-1)
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # rescaled_feature = scaler.fit_transform(global_feature)

    # predict label of test image
    prediction = clf.predict(global_feature.reshape(1,-1))[0]

    # show predicted label on image
    cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 3)

    # display the output image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

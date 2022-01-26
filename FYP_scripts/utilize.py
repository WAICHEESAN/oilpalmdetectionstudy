# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 22:04:49 2021

@author: Chee San
"""

'''
Contain functions to be used in different models
'''

import os, glob, random, shutil, itertools
import numpy as np
import matplotlib.pyplot as plt

def clear_used_samples(source_list, sample_list):
    for i in sample_list:
        source_list.remove(i)

def prep_data(directory, species, train_size, test_size):
    os.chdir(directory)
    if os.path.isdir(f'train/{species}') is True:
        shutil.rmtree(f'train/{species}')
    os.makedirs(f'train/{species}')
    # if os.path.isdir(f'valid/{species}') is True:
    #     shutil.rmtree(f'valid/{species}')
    # os.makedirs(f'valid/{species}')
    if os.path.isdir(f'test/{species}') is True:
        shutil.rmtree(f'test/{species}')
    os.makedirs(f'test/{species}')
    total_list = glob.glob(f'{species}/*')
    training_list = random.sample(total_list, train_size)
    clear_used_samples(total_list, training_list)
    [shutil.copy2(pic, f'train/{species}') for pic in training_list]
    # validation_list = random.sample(total_list, validate_size)
    # clear_used_samples(total_list, validation_list)
    # [shutil.copy2(pic, f'valid/{species}') for pic in validation_list]
    test_list = random.sample(total_list, test_size)
    [shutil.copy2(pic, f'test/{species}') for pic in test_list]
    
def plot_result(history):
    acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    # val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, acc, label = 'Training accuracy')
    # plt.plot(epochs, val_acc, label = 'Validation accuracy')
    plt.title('Training accuracy')
    plt.legend(loc='lower right')

    plt.subplot(2, 2, 2)
    plt.plot(epochs, loss, label = 'Training loss')
    # plt.plot(epochs, val_loss, label = 'Validation loss')
    plt.title('Training loss')
    plt.legend(loc='upper right')

    plt.show()
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Purples):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
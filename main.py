#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 02:54:13 2021

@author: buzun
"""

from preprocessing import preProcess
from model import BasicModel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

# GPU initialize
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)

# PreProcessing 
data = []
labels = []
data, labels = preProcess("Dataset")

xTrainData, xTestData, yTrainData, yTestData = train_test_split(data, labels, test_size=0.25, random_state=2)

del data
del labels

print('\nNumber of training pairs: ', len(xTrainData))
print('\nNumber of training pairs: ', len(xTestData))
print('\nNumber of training pairs: ', len(yTrainData))
print('\nNumber of training pairs: ', len(yTestData))
#print('Number of validation pairs: ', len(validData))

# Model build and fit
model = BasicModel()
history = model.fit(xTrainData, yTrainData,
          batch_size = 16,      #dec
          epochs = 35,           #inc
          verbose = 1,
          validation_data = (xTestData, yTestData)
          )

# Save weights
model.save("my_h5_model.h5") # 699 MB
model.save_weights("covid19_weights.h5") # 452 KB

# Load weights
model.load_weights("model_Adam_016_095.h5")

# Plotting #
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predictedModel = model.predict(xTestData)
model.evaluate(xTestData, yTestData)


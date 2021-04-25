#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 02:54:13 2021

@author: buzun
"""

from subprocesses import preProcess, convertBinaryResults, diff
from models import BasicModel, CNN
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

xTrainData, xTestData, yTrainData, yTestData = train_test_split(data, labels, test_size=0.33, random_state=2)

del data
del labels

print('\nNumber of xTrainData pairs: ', len(xTrainData))
print('\nNumber of xTestData pairs: ', len(xTestData))
print('\nNumber of yTrainData pairs: ', len(yTrainData))
print('\nNumber of yTestData pairs: ', len(yTestData))
#print('Number of validation pairs: ', len(validData))

# Model build and fit
model = BasicModel()
model = CNN()
model.compile(optimizer = "Adam", loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()
history = model.fit(xTrainData, yTrainData,
          batch_size = 8,      #dec
          epochs = 35,           #inc
          verbose = 2,
          validation_data = (xTestData, yTestData)
          )

# Save weights
model.save("my_h5_model.h5") # 699 MB
model.save_weights("covid19_weights.h5") # 452 KB

# Load weights
model.load_weights("my_h5_model.h5")

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

# Evaluating
testEvaluate = model.evaluate(xTestData, yTestData)
print("loss: " + str(testEvaluate[0]) + "\t accuracy: " + str(testEvaluate[1]))

# Prediction
predictedModel = model.predict(xTestData)
binaryResults = convertBinaryResults(predictedModel, 0.6)
fails, diffResults = diff(yTestData, binaryResults)
print("fails = " + str(fails))







    
    

    
    
    
    
    
    


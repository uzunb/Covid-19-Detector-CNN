#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 02:54:13 2021

@author: buzun
"""

from preprocessing import preProcess
from model import *
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
data, labels = preProcess("Dataset") # burda data ve label problemi var gibi
print(data[1])
xTrainData, xTestData, yTrainData, yTestData = train_test_split(data, labels, test_size=0.25, random_state=2)

del data
del labels

print('\nNumber of training pairs: ', len(xTrainData))
print('\nNumber of training pairs: ', len(xTestData))
print('\nNumber of training pairs: ', len(yTrainData))
print('\nNumber of training pairs: ', len(yTestData))
#print('Number of validation pairs: ', len(validData))

# Model build and fit
model = CNN()
history = model.fit(xTrainData, yTrainData,
          #batch_size = 16,      #dec
          epochs = 20,           #inc
          validation_split= 0.40                    #olmaması val_accuracy'i bulamıyor.
          #verbose = 1,
          #validation_data = (xTestData, yTestData) #olması hata çıkarıyor.
          )

#model_test
test_eval = model.evaluate(xTestData, yTestData, verbose=0)
print(test_eval)

# Save weights
model.save("my_h5_model.h5") # 699 MB
model.save_weights("covid19_weights.h5") # 452 KB

# Load weights
#model.load_weights("model_Adam_016_095.h5")

# Plotting #
print(history.history.keys())

#Accuracy and Loss
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


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


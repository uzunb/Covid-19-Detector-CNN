#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 02:54:13 2021

@author: buzun
"""

<<<<<<< HEAD
from subprocesses import preProcess, convertBinaryResults, diff
from models import BasicModel, CNN
=======
from preprocessing import preProcess
from model import *
>>>>>>> 1644ff5696f23d3aa29289a091eabce8d18d2ec0
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
<<<<<<< HEAD
data, labels = preProcess("Dataset")

xTrainData, xTestData, yTrainData, yTestData = train_test_split(data, labels, test_size=0.33, random_state=2)
=======
data, labels = preProcess("Dataset") # burda data ve label problemi var gibi
print(data[1])
xTrainData, xTestData, yTrainData, yTestData = train_test_split(data, labels, test_size=0.25, random_state=2)
>>>>>>> 1644ff5696f23d3aa29289a091eabce8d18d2ec0

del data
del labels

print('\nNumber of xTrainData pairs: ', len(xTrainData))
print('\nNumber of xTestData pairs: ', len(xTestData))
print('\nNumber of yTrainData pairs: ', len(yTrainData))
print('\nNumber of yTestData pairs: ', len(yTestData))
#print('Number of validation pairs: ', len(validData))

# Model build and fit
<<<<<<< HEAD
model = BasicModel()
model = CNN()
model.compile(optimizer = "Adam", loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()
history = model.fit(xTrainData, yTrainData,
          batch_size = 8,      #dec
          epochs = 35,           #inc
          verbose = 2,
          validation_data = (xTestData, yTestData)
=======
model = CNN()
history = model.fit(xTrainData, yTrainData,
          #batch_size = 16,      #dec
          epochs = 20,           #inc
          validation_split= 0.40                    #olmaması val_accuracy'i bulamıyor.
          #verbose = 1,
          #validation_data = (xTestData, yTestData) #olması hata çıkarıyor.
>>>>>>> 1644ff5696f23d3aa29289a091eabce8d18d2ec0
          )

#model_test
test_eval = model.evaluate(xTestData, yTestData, verbose=0)
print(test_eval)

# Save weights
model.save("my_h5_model.h5") # 699 MB
model.save_weights("covid19_weights.h5") # 452 KB

# Load weights
<<<<<<< HEAD
model.load_weights("my_h5_model.h5")
=======
#model.load_weights("model_Adam_016_095.h5")
>>>>>>> 1644ff5696f23d3aa29289a091eabce8d18d2ec0

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

# Evaluating
testEvaluate = model.evaluate(xTestData, yTestData)
print("loss: " + str(testEvaluate[0]) + "\t accuracy: " + str(testEvaluate[1]))

# Prediction
predictedModel = model.predict(xTestData)
binaryResults = convertBinaryResults(predictedModel, 0.6)
fails, diffResults = diff(yTestData, binaryResults)
print("fails = " + str(fails))







    
    

    
    
    
    
    
    


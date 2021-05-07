#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 02:54:13 2021

@author: buzun & BKaralii
"""
#%% Imports

from subprocesses import preProcess, convertBinaryResults, diff
from models import BasicModel, CNN
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import pandas as pd

#%% GPU initialize

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)

#%% PreProcessing

datasetName = "Dataset"
data, labels, dataset = preProcess(datasetName)
xTrainData, xTestData, yTrainData, yTestData = train_test_split(data, labels, test_size=0.25, random_state=2)



print('\nNumber of xTrainData pairs: ', len(xTrainData))
print('\nNumber of xTestData pairs: ', len(xTestData))
print('\nNumber of yTrainData pairs: ', len(yTrainData))
print('\nNumber of yTestData pairs: ', len(yTestData))

#%% Model build and fit

model = CNN()
history = model.fit(xTrainData, yTrainData,
          batch_size = 16,      
          epochs = 20,           
          validation_split= 0.25,
          verbose = 2,
          )

# model_test
testEvaluate = model.evaluate(xTestData, yTestData, verbose=0)
print("loss: " + str(testEvaluate[0]) + "\t accuracy: " + str(testEvaluate[1]))

#%% Save weights

model.save("my_h5_model.h5")
model.save_weights("covid19_weights.h5")

#%% Load weights

model.load_weights("my_h5_model.h5")

#%% Plotting

print(history.history.keys())

#Accuracy and Loss
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training accuracy', color="green")
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy', color="blue")
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss', color="green")
plt.plot(epochs, val_loss, 'b', label='Validation loss', color="blue")
plt.title('Training and validation loss')
plt.legend()
plt.show()


# Matrix presentation
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10),
                        subplot_kw={'xticks': [], 'yticks': []})
from PIL import Image
for i, ax in enumerate(axes.flat):
    img = Image.fromarray(dataset[i][0])
    ax.imshow(img)
    ax.set_title(dataset[i][1])
plt.tight_layout()
plt.show()

#%% Prediction

predictions = model.predict_classes(xTestData)

labelTestData = []
for i in yTestData[:,1:2]:
    labelTestData.append('Non-Covid') if (i == 1) else labelTestData.append('Covid')
        
predictionsList = []
for i in predictions:
    labelTestData.append('Non-Covid') if (i == 1) else labelTestData.append('Covid')

# Write CSV file
results=pd.DataFrame({"Label" :labelTestData, "Prediction":predictionsList})
results.to_csv("prediction_results.csv",index=False)







    
    

    
    
    
    
    
    


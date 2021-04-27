#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 01:21:45 2021

@author: buzun
"""

from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, UpSampling2D
from keras.layers import BatchNormalization, Dropout, LeakyReLU, ZeroPadding2D
from keras.layers.merge import add, concatenate
from keras.optimizers import Adam, SGD, Adagrad, Adadelta


def BasicModel(inputSize = (256,256,1)):

    inputs = Input(inputSize)
    conv = Conv2D(32, 3, activation='relu', kernel_initializer="he_normal")(inputs)
    #conv = Conv2D(64, 3, activation='relu', kernel_initializer="he_normal")(conv)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
                         
    conv = Conv2D(64, 3, activation='relu', kernel_initializer="he_normal")(pool)
    #conv = Conv2D(64, 3, activation='relu', kernel_initializer="he_normal")(conv)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    
    conv = Conv2D(128, 3, activation='relu', kernel_initializer="he_normal")(pool)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)

    conv = Conv2D(256, 3, activation='relu', kernel_initializer="he_normal")(pool)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)

    conv = Conv2D(256, 3, activation='relu', kernel_initializer="he_normal")(pool)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)  

    conv = Conv2D(256, 3, activation='relu', kernel_initializer="he_normal")(pool)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)  
                      
    flat = Flatten()(pool)
    dense = Dense(units = 128, activation="relu")(flat)
    drop = Dropout(0.4)(dense)
    dense = Dense(units = 64, activation="relu")(drop)
    drop = Dropout(0.4)(dense)
    dense = Dense(units = 32, activation="relu")(drop)
    output = Dense(units = 1, activation="sigmoid")(dense)
        
    model = Model(inputs = inputs, outputs = output)
    model.compile(optimizer = Adam(lr=0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])
        
    model.summary()
    
    return model


def CNN(inputSize = (256,256,1)):

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape = inputSize,padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))

    model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

    model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

    model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

    model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(optimizer = Adam(lr=0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

def Covid19Model():
    
    #Initializing the CNN
    #there is also a graph option but we'll use sequential ANN Model
    classifier = Sequential()
    
    #step 1 - Convolution
    #creating the feature map by using feature detector from ınput image
    
    classifier.add(Conv2D(32,3,3, input_shape=(64,64,3), activation='relu'))
    #32 Feature maps&detetctors uses 3 by 3 matrices, we can put 128 in the powerful machines

        
    #step -2 Pooling
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    
    classifier.add(Conv2D(32,3,3, input_shape=(64,64,3), activation='relu'))
    
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    
    
    #step -3 Flattening
    classifier.add(Flatten())
        
    #step-4 Full connection step
    classifier.add(Dense(output_dim = 256, activation = 'relu'))
    classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
    #binary outcome
        
    
    #compiling the cnn
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
        
        
    return classifier
    
    
    
    
    
    
    

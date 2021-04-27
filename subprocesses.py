import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Param1: Path of Image dataset
# Return: data  : NumPy array uint8
#         labels: NumPy array float64
def preProcess(imgPath):
    
    img_path = imgPath          #"Dataset"
    
    dataset = []
    width = 256
    height = 256
    
    
    for directory in os.listdir(img_path):
        path = os.path.join(img_path, directory)
    
        if not os.path.isdir(path):
            continue
        for item in os.listdir(path):
            if item.startswith("."):
                continue
            img = cv2.imread(os.path.join(path, item))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        
            img = cv2.resize(img, (width, height))
            dataset.append([img, directory])
            
    # labeling 
    data, labels = zip(*dataset)
    
    data = np.array(data, dtype="float32")
    data = data.reshape(data.shape[0], width, height, 1)  # 1 is channel
    #data = np.expand_dims(data, axis=0)
    
    # covid : 0
    # non-covid : 1
    labels = onehotLabels(labels)
    
    return data, labels

def onehotLabels(values):
    labelEncoder = LabelEncoder()
    integerEncoded = labelEncoder.fit_transform(values)
    onehotEncoder = OneHotEncoder(sparse=False)
    integerEncoded = integerEncoded.reshape(len(integerEncoded), 1)
    onehot_encoded = onehotEncoder.fit_transform(integerEncoded)
    return onehot_encoded

# for a column arrays
def convertBinaryResults(predictedResults, threshold = 0.6):
    binaryResults = []
    
    for i in range(predictedResults.shape[0]):
        if(predictedResults[i] > threshold):
            binaryResults.append(1)
        else:
            binaryResults.append(0)
    
    return np.array(binaryResults)

# for a column arrays
def diff(groundTruth, predictedResults):
    
    size = groundTruth.shape[0]
    diffResults = []
    fails = 0
    for i in range(size):
        if(groundTruth[i] != predictedResults[i]):
            fails = fails + 1
            diffResults.append(False)
        else:
            diffResults.append(True)
    
    return fails, np.array(diffResults)
    


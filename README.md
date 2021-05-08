# Covid-19-Detector-CNN

### Requirements & Dependencies
Tested this code using:
* Ubuntu 20.04.2 LTS
* Python 3.8.5
* Tensorflow 2.4.1
* Keras 2.4.3
* Opencv 4.5.2

requirements.txt consists required dependencies.  
*Easy way:*  You can install the requirements with the following commands:
```bash
cd myproject/
virtualenv venv
./venv/bin/activate
pip install -r requirements.txt
```
***

### Dataset
This dataset contain 3616 COVID-19 positive cases along with 10,192 Normal, 6012 Lung Opacity (Non-COVID lung infection), and 1345 Viral Pneumonia images.  
Used 1000s of normal and covid images.


NORMAL                   |  COVID19
:-------------------------:|:-------------------------:
![normal](https://user-images.githubusercontent.com/39219223/117543910-89614b00-b027-11eb-8592-bd735dea83ee.png)    |  ![covid](https://user-images.githubusercontent.com/39219223/117543885-646cd800-b027-11eb-99f4-f0007796dbf5.png)
    
for the full dataset : https://www.kaggle.com/tawsifurrahman/covid19-radiography-database  
------

### Model
![model_plot](https://user-images.githubusercontent.com/39219223/117543907-81a1a680-b027-11eb-800f-fb586bcf89b5.png)

---------
### Train
  
Model trained with 1500 labeled images. Model fit parameters are:
```
batch_size = 16  
epochs = 20  
validation_split= 0.25
```

ACCURACY                  |  LOSS 
:-------------------------:|:-------------------------:
![accuracy_figure](https://user-images.githubusercontent.com/39219223/117543874-574fe900-b027-11eb-81d1-8d9caf1448f3.png)    |  ![loss_figure](https://user-images.githubusercontent.com/39219223/117543900-78b0d500-b027-11eb-9313-efdc6a28b5d7.png)

--------  

### Results

![pred_figure](https://user-images.githubusercontent.com/39219223/117543917-9120ef80-b027-11eb-98d6-4b65e9531554.png)



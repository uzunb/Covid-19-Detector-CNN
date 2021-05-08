# Covid-19-Detector-CNN

### Requirements & Dependencies
I have tested this code using:
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
I used 1000s of normal and covid images.


NORMAL                   |  COVID19
:-------------------------:|:-------------------------:
![normal](https://raw.githubusercontent.com/uzunb/Covid-19-Detector-CNN/main/images/normal.png?token=AJLHAF6QD326PGY6CYFXD3TAS2PQ6)    |  ![covid](https://raw.githubusercontent.com/uzunb/Covid-19-Detector-CNN/main/images/covid.png?token=AJLHAF3G6EHNI4QEN5UXNGTAS2PPM)
    
for the full dataset : https://www.kaggle.com/tawsifurrahman/covid19-radiography-database  
------

### Model
![model_plot](https://raw.githubusercontent.com/uzunb/Covid-19-Detector-CNN/main/images/model_plot.png?token=AJLHAF732JMMWGULXPOU5ULAS2TBC) 

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
![acc](https://raw.githubusercontent.com/uzunb/Covid-19-Detector-CNN/main/images/accuracy_figure.png?token=AJLHAF5YNEHZCETQNE7QJMTAS2PTM)    |  ![loss](https://raw.githubusercontent.com/uzunb/Covid-19-Detector-CNN/main/images/loss_figure.png?token=AJLHAF7SACNHSI2GSB6SGK3AS2PXS)

--------  

### Results

![pred_fig](https://raw.githubusercontent.com/uzunb/Covid-19-Detector-CNN/main/images/pred_figure.png?token=AJLHAF37DPWK2VY2ZYPCVB3AS2QKM)




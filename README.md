# Machine Learning-Assisted Equivalent Circuit Identification for Dielectric Spectroscopy of Polymers

## About the Paper
This repository is based on our paper titled "Machine Learning-Assisted Equivalent Circuit Identification for Dielectric Spectroscopy of Polymers." The paper discusses a novel convolutional neural network (CNN) model to predict the electrical equivalent circuit (EEC) topology from broadband dielectric spectroscopy data, enhancing the characterization of polymer membranes' and achieving SOTA resutls.

## Installation
To install the required packages, please run the following command:
```bash
pip install -r requirements.txt
```
## Usage
You can easliy run the model in ```main.py``` and define global parameters. 
The file contains the modules required for generating the data ```model.generate_data()``` and  pre-processing the data ```model.preprocess_data(data=data)```.\
For training the model, the user can call ```model.Train(Training_data)```, and 
for testing the model you can use ```model.Predict(Test_data)```. 

Addtionally ```model.Predict(Test_data)``` will save the results in the folder "predictions".


 
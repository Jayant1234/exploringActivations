#!/usr/bin/env python
# coding: utf-8

# ## Improvements done as suggested by Dr.Miller and Mr.Chang in class presentation
# 1. Sigmoid added to list of activation functions and results are noted.
# 2. Random Forest ran on both MNIST and Fashion MNIST. 
# 3. Results are updated with more decisive wordings.

# In[1]:


random_seed=7
from numpy.random import seed
seed(random_seed)
from tensorflow import set_random_seed
set_random_seed(random_seed)
import numpy as np
from IPython.display import clear_output
import sys
sys.path.append('../exploringActivations/')
from data import Datasets
from activation_functions import CustomActivation
from models import CNN,DNN,CAE
import matplotlib.pyplot as plt
import scipy as sci
from keras.datasets import mnist
from keras.datasets import fashion_mnist
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import optimizers
import pandas as pd
from tensorflow.keras import backend as K
import math
import matplotlib.pyplot as plt
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from keras.activations import sigmoid, selu


# ## Overview
# We use three classes found in exploringActivation Folder. 
# They are imported using : 
# ```
# from data import Datasets
# from activation_functions import CustomActivation
# from models import CNN,DNN,CAE
# 
# ```
# 
# We use Datasets to get preprocessed data based on the choice of models. 
# 
# And we use activation functions class inside models to intitiate different activation functions
# 
# And Models class defines the 

# In[2]:


# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


# In[3]:


def plot_model(history,name,modelname,datasetname):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(name+'-'+modelname+'model accuracy on'+datasetname)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    #fig= plt.figure()
    plt.savefig('../results/'+name+'_'+modelname+'_'+datasetname)
    plt.show()


# In[8]:


def run_forrest_run(dataset_list, activation_list, modelname): 
    for dataset_name in dataset_list:
        for name in activation_list:
            for model in modelname: 
                if model=="DNN": 
                    dataset=Datasets()
                    if(dataset_name =='MNIST'): 
                        x_train,x_test,y_train,y_test=dataset.get_mnist("DNN")
                        num_classes=dataset.num_classes
                        input_shape=dataset.input_shape
                    elif(dataset_name=='Fashion-MNIST'):
                        x_train,x_test,y_train,y_test=dataset.get_fashion_mnist("DNN")
                        num_classes=dataset.num_classes
                        input_shape=dataset.input_shape
                    dnn = DNN(name)
                    score,history = dnn.run_model(input_shape, x_train, x_test, y_train, y_test,1)
            
                else:
                    dataset=Datasets()
                    if(dataset_name =='MNIST'): 
                        x_train,x_test,y_train,y_test=dataset.get_mnist("CNN")
                    elif(dataset_name=='Fashion-MNIST'):
                        x_train,x_test,y_train,y_test=dataset.get_fashion_mnist("CNN")
                    num_classes=dataset.num_classes
                    input_shape=dataset.input_shape
                    if model =="CNN":
                        cnn = CNN(name)
                        score,history = cnn.run_model(input_shape, x_train, x_test, y_train, y_test)
                    elif model =="CAE":
                        cae = CAE(name)
                        score,history = cae.run_model(input_shape, x_train, x_test, y_train, y_test)
                    score,history = cnn.run_model(input_shape, x_train, x_test, y_train, y_test)
                plot_model(history,name,model,dataset_name)


# ## Main running script , all deep models can be run for each activation using this.

# In[9]:


#running script.../seperate tests are done below
dataset_list=['MNIST', 'Fashion-MNIST']
activation_list=['swish','selu', 'gelu', 'relu','sigmoid']
modelname=["DNN","CNN","CAE"]
single_activation=['sigmoid']
run_forrest_run(dataset_list,single_activation, modelname)


# ## Random Forest on MNIST and Fashion MNIST for baseline for both

# In[11]:


# running random forest on MNIST
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
dataset = Datasets()
x_train,x_test,y_train,y_test=dataset.get_mnist("DNN")
num_classes=dataset.num_classes
input_shape=dataset.input_shape
print(input_shape)
print(x_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train.shape)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(x_train, y_train)
print("Compute predictions")
predicted = clf.predict(x_test)
print("Val Accuracy: ", accuracy_score(y_test, predicted))

print("Compute predictions")
predicted = clf.predict(x_train)
print("Train Accuracy: ", accuracy_score(y_train, predicted))


# In[12]:


# running random forest on Fashion-MNIST
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
dataset = Datasets()
x_train,x_test,y_train,y_test=dataset.get_fashion_mnist("DNN")
num_classes=dataset.num_classes
input_shape=dataset.input_shape
print(input_shape)
print(x_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train.shape)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(x_train, y_train)
print("Compute predictions")
predicted = clf.predict(x_test)
print("Val Accuracy: ", accuracy_score(y_test, predicted))

print("Compute predictions")
predicted = clf.predict(x_train)
print("Train Accuracy: ", accuracy_score(y_train, predicted))


# ## Individual testing of each Model 

# In[ ]:


#testing for DNN
dataset_list=['MNIST', 'Fashion-MNIST']
activation_list=['swish','selu', 'gelu', 'relu']
modelname="DNN"
for dataset_name in dataset_list:
    for name in activation_list:
        dataset=Datasets()
        if(dataset_name =='MNIST'): 
            x_train,x_test,y_train,y_test=dataset.get_mnist("DNN")
        elif(dataset_name=='Fashion-MNIST'):
            x_train,x_test,y_train,y_test=dataset.get_fashion_mnist("DNN")
        num_classes=dataset.num_classes
        input_shape=dataset.input_shape
        print(input_shape)
        print(x_train.shape)
        print(x_test.shape)
        print(y_test.shape)
        print(y_train.shape)
        dnn = DNN(name)
        score,history = dnn.run_model(input_shape, x_train, x_test, y_train, y_test,1)
        plot_model(history,name,modelname,dataset_name)


# In[ ]:


#testing for Convoluted Auto encoders
dataset_list=['MNIST', 'Fashion-MNIST']
activation_list=['swish','selu', 'gelu', 'relu']
modelname="CAE"
for dataset_name in dataset_list:
    for name in activation_list:
        dataset=Datasets()
        if(dataset_name =='MNIST'): 
            x_train,x_test,y_train,y_test=dataset.get_mnist("CNN")
        elif(dataset_name=='Fashion-MNIST'):
            x_train,x_test,y_train,y_test=dataset.get_fashion_mnist("CNN")
        num_classes=dataset.num_classes
        input_shape=dataset.input_shape
        print(input_shape)
        print(x_train.shape)
        print(x_test.shape)
        print(y_test.shape)
        print(y_train.shape)
        dnn = CAE(name)
        score,history = dnn.run_model(input_shape, x_train, x_test, y_train, y_test)
        plot_model(history,name,modelname,dataset_name)


# In[ ]:


#testing for CNN
dataset_list=['MNIST', 'Fashion-MNIST']
activation_list=['swish','selu', 'gelu', 'relu']
modelname="CNN"
for dataset_name in dataset_list:
    for name in activation_list: 
        dataset=Datasets()
        if(dataset_name =='MNIST'): 
            x_train,x_test,y_train,y_test=dataset.get_mnist("CNN")
        elif(dataset_name=='Fashion-MNIST'):
            x_train,x_test,y_train,y_test=dataset.get_fashion_mnist("CNN")
        num_classes=dataset.num_classes
        input_shape=dataset.input_shape
        # x_train=dataset.edge_detection(x_train)
        # x_test=dataset.edge_detection(x_test)
        print(input_shape)
        print(x_train.shape)
        print(x_test.shape)
        print(y_test.shape)
        print(y_train.shape)
        cnn = CNN(name)
        model,history=cnn.run_model(input_shape, x_train, x_test, y_train, y_test)
        plot_model(history,name,modelname,dataset_name)


# # EDA and Data preprocessing

# In[13]:


dataset=Datasets()
x_train,x_test,y_train,y_test=dataset.get_fashion_mnist("CNN")
num_classes=dataset.num_classes
input_shape=dataset.input_shape
# x_train=dataset.edge_detection(x_train)
# x_test=dataset.edge_detection(x_test)
print(input_shape)
print(x_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train.shape)
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print(unique_elements)
print(counts_elements)


# In[14]:


x_train[0].shape


# In[15]:


x_geo=dataset.filtering(x_train[1,:,:,0])
plt.imshow(x_geo)
plt.savefig('../results/filtered_images.png')


# In[16]:


plt.imshow(x_train[1,:,:,0])
plt.savefig('../results/normal_images.png')


# In[17]:


x_den=dataset.denoising(x_train[1,:,:,0])
plt.imshow(x_den)
plt.savefig('../results/denoised_images.png')


# In[18]:


x_ed=dataset.edge_detection(x_train[1,:,:,0])
plt.imshow(x_ed)
plt.savefig('../results/edge_images.png')


# In[ ]:





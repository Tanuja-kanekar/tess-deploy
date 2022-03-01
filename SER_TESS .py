#!/usr/bin/env python
# coding: utf-8

# ## TESS DATA: DATA MODELLING

# In[1]:


#Importing the  necessary libraries
import librosa 
from librosa import display
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import sys


# In[2]:


#Looking the path of current working directory using below command.
#Storing it in the path variable.
path = os.getcwd()
print(path)


# ### Loading the data

# In[3]:


#Reading the csv file and viewing it
data = pd.read_csv("Tess_data1.csv")
data.head(600)


# In[4]:


#Function to create the waveplot for the audio
def createWaveplot(data, sr, e):
    plt.figure(figsize=(10,3))
    plt.title('Waveplot for audio with {} emotion'.format(e),size=15)
    librosa.display.waveplot(data, sr=sr)
    plt.show()


# In[5]:


#Function to create the spectrogram for the audio
def createSpectrogram(data, sr, e):
    X= librosa.stft(data)
    Xdb=librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12,3))
    plt.title('Spectrogram for audio with {} emotion'.format(e),size=15)
    librosa.display.specshow(Xdb, sr=sr,x_axis='time',y_axis='hz')
    plt.colorbar(format='%+2.0f db')
    plt.show()


# In[6]:


emotion_tess = 'female_sad'
path_tess =np.array(data.Files_path2[data.labels==emotion_tess])[2]
audio_data_tess,sampling_rate = librosa.load(path_tess)

createWaveplot(audio_data_tess,sampling_rate,emotion_tess)
createSpectrogram(audio_data_tess,sampling_rate,emotion_tess)
ipd.Audio(path_tess)


# In[7]:


#Function to extract the features for all the files
Feature_data = pd.DataFrame(columns=['Features'])

counter = 0
for index, path in enumerate(data.Files_path2):
    X, sample_rate = librosa.load(path,res_type='kaiser_fast')
    mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,n_mfcc=13).T, axis=0)
    Feature_data.loc[counter]=[mfccs]
    counter=counter+1


# In[8]:


Feature_data.head()


# In[9]:


#Converting the each features to its own column format and concatenating with the Emotion field
Feature_data=pd.concat([pd.DataFrame(Feature_data['Features'].values.tolist()),data.labels],axis=1)
Feature_data[:5]


# In[10]:


#Looking for all the columns
Feature_data.columns


# In[11]:


#Intializing our input and output data i.e X and Y 
#where X contains features of our audio data and Y contains our Target part i.e Emotions
X_data=Feature_data.drop(['labels'],axis=1)
X_data.head()


# In[12]:


#Target variable
Y_data=Feature_data.labels
Y_data.head()


# In[13]:


#Spliting the data into train and test where test size is 20 percent
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_data,Y_data,test_size=0.2,random_state=40)


# In[14]:


#checking the size of train and test data
print((x_train.shape[0],x_test.shape[0]))


# In[15]:


#Loading the necessary libraries and fitting and training the model, then checking the accuracy
#Checking the accuracy for both scaled and unscaled data.
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

steps = [('scaler', StandardScaler()),
        ('SVM', SVC())]
        
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Fit the pipeline to the training set: svc_scaled
svc_scaled = pipeline.fit(x_train, y_train)

# Instantiate and fit a classifier to the unscaled data
svc_unscaled = SVC(kernel = 'linear').fit(x_train, y_train)

# Compute and printing the accuracy metrics
print('Accuracy with Scaling: {}'.format(svc_scaled.score(x_test, y_test)))
print('Accuracy without Scaling: {}'.format(svc_unscaled.score(x_test, y_test)))


# In[16]:


#Printing the accuracy of both train and test to check if model is overfitting or underfitting
train_acc = float(svc_scaled.score(x_train, y_train)*100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(svc_scaled.score(x_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)


# In[17]:


scaled_predictions = svc_scaled.predict(x_test)
scaled_predictions1 =svc_scaled.predict(x_train)


# In[18]:


#Printing the classification report
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,scaled_predictions))


# In[27]:


final_model= pipeline.fit(X_data,Y_data)
final_model


# In[23]:


import pickle


# In[28]:


pickle.dump(final_model,open('Tess_data.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





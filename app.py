from flask import Flask,render_template,request
import pickle
import pandas as pd 
import numpy as np 
import glob
import os
import sys
import IPython.display as ipd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import librosa

ALLOWED_EXTENSIONS = set(['wav'])

app = Flask(__name__)
model = pickle.load(open("Tess_data.pkl","rb"))

def allowed_file(filename):
  return '.' in filename and \
          filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def upload():
    return render_template('input.html')
	
@app.route('/output',methods=['POST'])
def output():
  if request.method=='POST':
    file = request.files['file']
    
    if file and allowed_file(file.filename):
       X, sample_rate = librosa.load(file,res_type='kaiser_fast')
       mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,n_mfcc=13).T, axis=0)
       features=mfccs.reshape(1,-1)
       x=model.predict(features)
    return render_template("output.html",result=x)
    
if __name__ == '__main__':
    app.run(debug=True)

    
from __future__ import print_function
from keras.models import load_model
from keras.utils import np_utils
import numpy as np
import cv2
from gtts import gTTS
import os

def Playme(Result):
    language = 'en'
    myobj = gTTS(text=Result, lang=language, slow = True)
    myobj.save("welcome.mp3")
    os.system("welcome.mp3")





DIR = 'D:/Study_Material/Python_3_Tutorial/PythonScripts/Machine_Learning/Project/Images/'
img_rows, img_cols = 32, 32
Result = {0:'Airplane',1:'Automobile',2:'Bird',3:'Cat',4:'Deer',5:'Dog',6:'Frog',7:'Horse',8:'Ship',9:'Truck'}
model = load_model('CIFAR10.h5')

try:
    for img in (os.listdir(DIR)):
        path = os.path.join(DIR,img)
        img = cv2.imread(path)
        img = cv2.resize(img,(img_rows,img_cols))
        Image = np.expand_dims(np.array(img), axis=0)
        ResultSet = model.predict(Image)
        print("Its a:",Result[np.argmax(ResultSet)])
except Exception as e:
    print(e)

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from os import listdir
from os.path import isfile, join
import numpy as np
import argparse
import imutils
import cv2
import os
import glob
import pylab as plt



''' ["ka","kha","ga","gha","kna","cha","chha","ja","jha","yna","t`a","t`ha","d`a","d`ha","adna","ta","tha","da","dha","na","pa","pha","ba","bha","ma","yaw","ra","la","waw","sha","shat","sa","ha","aksha","tra","gya","0","1","2","3","4","5","6","7","8","9"]


labels =['yna', 't`aa', 't`haa', 'd`aa', 'd`haa', 'a`dna', 'ta', 'tha', 'da', 'dha', 'ka', 'na', 'pa', 'pha', 'ba', 'bha', 'ma', 'yaw', 'ra', 'la', 'waw', 'kha', 'sha', 'shat', 'sa', 'ha', 'aksha', 'tra', 'gya', 'ga', 'gha', 'kna', 'cha', 'chha', 'ja', 'jha', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
'''
labels = [u'\u091E',u'\u091F',u'\u0920',u'\u0921',u'\u0922',u'\u0923',u'\u0924',u'\u0925',u'\u0926',u'\u0927',u'\u0915',u'\u0928',u'\u092A',u'\u092B',u'\u092c',u'\u092d',u'\u092e',u'\u092f',u'\u0930',u'\u0932',u'\u0935',u'\u0916',u'\u0936',u'\u0937',u'\u0938',u'\u0939','ksha','tra','gya',u'\u0917',u'\u0918',u'\u0919',u'\u091a',u'\u091b',u'\u091c',u'\u091d',u'\u0966',u'\u0967',u'\u0968',u'\u0969',u'\u096a',u'\u096b',u'\u096c',u'\u096d',u'\u096e',u'\u096f']
model = "DevaModel.h5"
text = ""
previmgname_int = 110
mypath='C:/Shubham/devanagriocr'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
#load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(model)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( join(mypath,onlyfiles[n]) )
  imgname,a = onlyfiles[n].split('.')
  imgname_int = int(imgname)
  # pre-process the image for classification
  images[n] = cv2.resize(images[n], (32,32))
  images[n] = images[n].astype("float") / 255.0
  images[n] = img_to_array(images[n])
  images[n] = cv2.cvtColor(images[n], cv2.COLOR_BGR2GRAY)
  #print image.shape
  images[n] = np.expand_dims(images[n], axis=0)
  #print image.shape
  images[n] = np.expand_dims(images[n], axis=3)
  #print image.shape
  #classify the input image
  lists = model.predict(images[n])[0]
  if((imgname_int - previmgname_int) == 1):
    text = text + (labels[np.argmax(lists)])
    previmgname_int = imgname_int
  elif(imgname_int - previmgname_int > 1 and imgname_int - previmgname_int <= 49):
    text = text +" " + (labels[np.argmax(lists)])
    previmgname_int = imgname_int
  elif((imgname_int - previmgname_int) >= 50):
    text = text + "\n" + (labels[np.argmax(lists)])
    previmgname_int = imgname_int

print("File saved successfully!")
  
with open('sample.txt', 'w') as file:
     file.write(text)

          


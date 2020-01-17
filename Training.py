import cv2
import os
from keras.applications import ResNet50
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
import random
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.utils import shuffle
from Model_ResNet50 import *

#Data Loading
X_train = []
y_train = []
X_val = []
y_val = []
path = "D:/Python Scripts/Project/Train/Modak"

for img in tqdm(os.listdir(path)):
    i = cv2.imread(os.path.join(path,img))
    i = cv2.resize(i,(224,224))
    X_train.append(i)
    y_train.append(1)

path = "D:/Python Scripts/Project/Train/Non-Modak"

for img in tqdm(os.listdir(path)):
    i = cv2.imread(os.path.join(path,img))
    i = cv2.resize(i,(224,224))
    X_train.append(i)
    y_train.append(0)

path = "D:/Python Scripts/Project/Val/Non-Modak"

for img in tqdm(os.listdir(path)):
    i = cv2.imread(os.path.join(path,img))
    i = cv2.resize(i,(224,224))
    X_val.append(i)
    y_val.append(0)
    
path = "D:/Python Scripts/Project/Val/Modak"

for img in tqdm(os.listdir(path)):
    i = cv2.imread(os.path.join(path,img))
    i = cv2.resize(i,(224,224))
    X_val.append(i)
    y_val.append(1)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

X_train = np.reshape(X_train,(722,224,224,3))
y_train = np.reshape(y_train,(722,2))
X_val = np.reshape(X_val,(206,224,224,3))
y_val = np.reshape(y_val,(206,2))

X_train, y_train = shuffle(X_train, y_train, random_state=0)
X_val, y_val = shuffle(X_val, y_val, random_state=0)


# Training
model = def_model(pretrained_weights = None) 
checkpoint =model_checkpoint = ModelCheckpoint('Intern\Weights\Resnet50.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
callback_list=  [checkpoint]
history  = model.fit(X_train, y_train, batch_size=4,epochs=100,validation_data = (X_val,y_val),callbacks = callback_list)                        

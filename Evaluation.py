import cv2
import os
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from Model_Resnet50 import *

path = "D:/Python Scripts/Project/Test/Modak"

X_test = []
y_test = []

### Loading the Test Data

for img in tqdm(os.listdir(path)):
    i = cv2.imread(os.path.join(path,img))
    i = cv2.resize(i,(224,224))
    X_test.append(i)
    y_test.append(1)

path = "D:/Python Scripts/Project/Test/Non-Modak"

for img in tqdm(os.listdir(path)):
    i = cv2.imread(os.path.join(path,img))
    i = cv2.resize(i,(224,224))
    X_test.append(i)
    y_test.append(0)

X_test  = np.array(X_test)
y_test = np.array(y_test)

X_test, y_test = shuffle(X_test, y_test, random_state=0)

### Calling the model using the weights generated after training

model = def_model(pretrained_weights='Project\Weights\Resnet50.96-0.0003-0.9873.hdf5')
y_pred = model.predict(X_test)
yp = []
for i in y_pred:
    yp.append(np.argmax(i))

### Printing the Accuracy
print('Accuracy = ',(accuracy_score(y_test, yp)*100))
           

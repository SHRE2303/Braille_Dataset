import pandas as pd
import numpy as np
import os 
from PIL import Image
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
path = 'C:/Braille Dataset'
training_data = []
for img in os.listdir(path):
    pic = cv2.imread(os.path.join(path,img))
    pic1 = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    pic2 = pic1.flatten()
    training_data.append(pic2)
df = pd.DataFrame(training_data)
target = []
for i in range(0,26):
    for j in range(0,60):
        target.append(i)
df['target'] = target
target = df['target']
inputs = df.drop('target',axis='columns')
x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.25, random_state=0      )
logisticRegr = LogisticRegression(solver='saga',max_iter=10000)
logisticRegr.fit(x_train, y_train)
score = logisticRegr.score(x_test, y_test)
print(score)
#logisticRegr.predict(inputs.iloc[0])
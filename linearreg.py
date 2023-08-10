import numpy as np
from sklearn import  linear_model
from sklearn.metrics import mean_squared_error, r2_score
import os
import cv2
import pandas as pd
from sklearn.model_selection import  train_test_split

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
x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.3, random_state=0)
regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)
y_pred = regr.predict(x_test)
print("'the coefficients are:",regr.coef_)
print("the mean squared error is:",mean_squared_error(y_test,y_pred))
print("coefficient of determination is:",r2_score(y_test,y_pred))
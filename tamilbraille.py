import numpy as np
from sklearn import  linear_model
from sklearn import tree
import os
import cv2
import pandas as pd
from sklearn.model_selection import  train_test_split
folderpath = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','z1','z2','z3','z4','z5']
path = "C:/Shrey/output/"
target = []
training_data = []
for i in range(len(folderpath)):
    path1 = path+folderpath[i]
    for img in os.listdir(path1):
        if os.path.isfile(os.path.join(path1, img)):
            target.append(i)
        pic = cv2.imread(os.path.join(path1,img))
        pic1 = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        pic2 = pic1.flatten()
        training_data.append(pic2)
df = pd.DataFrame(training_data)
df['target'] = target
target = df['target']
inputs = df.drop('target',axis='columns')
model = tree.DecisionTreeClassifier(max_depth=100,criterion='entropy')
x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.3, random_state=0)
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
 
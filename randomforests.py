import pandas as pd
import numpy as np
import os
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
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
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}
rf = RandomForestClassifier()
rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)
rand_search.fit(x_train, y_train)
best_rf = rand_search.best_estimator_
print('Best hyperparameters:',  rand_search.best_params_)
y_pred = best_rf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
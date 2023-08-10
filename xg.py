import pandas as pd
import os 
import cv2
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
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
model = XGBClassifier(min_split_loss = 0.2,grow_policy = 'lossguide',min_child_weight = 10)
model.fit(x_train,y_train)
predicted = model.predict(x_test,iteration_range=(0, model.best_iteration + 1))
print(predicted)
score = accuracy_score(y_test,predicted)
print(score)
#n_estimators=40, max_depth=100, learning_rate=1, objective='binary:logistic'
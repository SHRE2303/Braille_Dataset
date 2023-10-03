import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('D:/output.csv')
df.dropna(axis=1)
target = df['target']
inputs = df.drop('target',axis='columns')
x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.3, random_state=0)
logisticRegr = LogisticRegression(max_iter=10000)
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train)
x_test = min_max_scaler.fit_transform(x_test)
logisticRegr.fit(x_train, y_train)
score = logisticRegr.score(x_test, y_test)
print("accuracy:",score*100)
    
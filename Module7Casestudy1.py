# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 11:42:16 2021

@author: mukesh
"""

import pandas as pd
voice = pd.read_csv(r"C:\Users\mukes\Python\Case Study\voice.csv")
voice['label'] = voice['label'].map({'male':0,'female':1})

# create X data and Y data for training the model
X = voice.iloc[:,0:20]
Y = voice['label']

# Fit a logistic regression model and measure the accuracy on the test set.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

ln_model = LogisticRegression(solver='liblinear', max_iter=100)
train_x, test_x, train_y, test_y = train_test_split(X,Y, random_state= 10, test_size=0.20)
ln_model.fit(train_x, train_y)

predicted_data = ln_model.predict(test_x)
metrics.accuracy_score(predicted_data, test_y)
### 0.9274447949526814 --> 92% accuracy
metrics.mean_squared_error(predicted_data, test_y)
### 0.07255520504731862 --> 7.2%
metrics.r2_score(predicted_data,test_y)
## 0.7074971164936562 (1 is the best score) -- coefficient of determination

correlation = voice.corr()
import seaborn as sns
import matplotlib.pyplot as plt

plt.subplots(figsize=(20,15))
sns.heatmap(correlation, square=True, annot=False,cmap='YlGnBu')
plt.yticks(rotation=0,size=14)
plt.xticks(rotation=90,size=14)
plt.show()

pd.options.display.max_columns = None
pd.options.display.max_rows = None
correlation

# identify those correlation values less than -0.50 and greater than 0.50
bool_series = (correlation.lt(-0.50) | correlation.gt(0.50)) & (correlation.ne(1))
correlation[bool_series]

# for label identify those correlation values less than -0.50 and greater than 0.50
bool_label = correlation['label'].lt(-0.50) | correlation['label'].gt(0.50)
correlation['label'][bool_label]
"""
Q25        0.511455
IQR       -0.618916
meanfun    0.833921
label      1.000000
Name: label, dtype: float64
"""

# Q25 and meanfreq are highly correlated
# IQR and sd are highly correlated
# dfrange and maxdom are highly correlated
# Q25 and centroid are highlighly correlated
# sp.ent and sfm are highly correlated

X = X.drop('meanfreq',axis=1)
X = X.drop('sd',axis=1)
X = X.drop('maxdom',axis=1)
X = X.drop('centroid',axis=1)
X = X.drop('sfm',axis=1)

train_x, test_x, train_y,test_y = train_test_split(X,Y, random_state=10, test_size=0.20)
ln_model = LogisticRegression(solver='liblinear', max_iter=100)
ln_model.fit(train_x,train_y)

predicted_data = ln_model.predict(test_x)
metrics.accuracy_score(predicted_data,test_y)
### 0.9258675078864353
metrics.mean_squared_error(predicted_data,test_y)
### 0.07413249211356467
metrics.r2_score(predicted_data,test_y)
### 0.7009674052665382

## Observation is that accuracy is almost the same as the previous model.




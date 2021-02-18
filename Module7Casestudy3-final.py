# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 19:23:27 2021

@author: mukesh
"""
import pandas as pd
df = pd.read_csv(r"C:\Users\mukes\Python\Case Study\loan_borowwer_data.csv")

# to identify if any null values
df.info()
# No missing data

# to identify which are numerical features and which are categorical.
numerical_features = ['credit.policy','int.rate','installment','log.annual.inc','dti','fico','days.with.cr.line',
                      'revol.bal','revol.util','inq.last.6mths','delinq.2yrs','pub.rec']
categorical_features = ['purpose']

df['not.fully.paid'].describe()

X = df.copy()
X.drop(['not.fully.paid'],axis=1,inplace=True)
Y = df['not.fully.paid']

#Converting categorical features
X_dummied = pd.get_dummies(X)
X_dummied.info()

#Train-Test-Split
from sklearn.model_selection import train_test_split

#testing data with 30% 
X_train, X_test, Y_train, Y_test = train_test_split(X_dummied, Y, test_size = 0.30,random_state = 101)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape

# Check highly correlated features
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize=(16,14))
sns.heatmap(X_train.corr(), cmap='Reds', annot=True, linewidths=.5, ax=ax)
## No highly correlated features


# zero variance (unique values) check
from sklearn.feature_selection import VarianceThreshold
X_train_num = X_train
constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(X_train_num)
print(X_train_num.columns[constant_filter.get_support()])
x_num = X_train_num[X_train_num.columns[constant_filter.get_support()]]
print(len(X_train_num.columns),len(X_train.columns))
# no columns were dropped when variancethreshold was checked. 

# Handling Feature Importance
from sklearn.ensemble import RandomForestClassifier
import numpy as np
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
features = X_train.columns
importances = rfc.feature_importances_
indices = np.argsort(importances)
fig, ax = plt.subplots(figsize=(16,14))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
# No featues with 0 importance, we will use original features

## Prediction 
## Decision Tree Classifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,Y_train)
Y_pred = dtc.predict(X_test)
accuracy_score(Y_pred, Y_test)
# 0.72

## Report
print(classification_report(Y_test, Y_pred))

## Confusion Matrix
print(confusion_matrix(Y_test, Y_pred))

## RandomForest
rfr = RandomForestClassifier()
rfr.fit(X_train,Y_train)
Y_pred = rfr.predict(X_test)
accuracy_score(Y_pred, Y_test)
# 0.84

## Report
print(classification_report(Y_test, Y_pred))

## Confusion Matrix
print(confusion_matrix(Y_test, Y_pred))


## Using model - Support Vector Machine
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
# SVC
svc = SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
accuracy_score(Y_pred, Y_test)
# 0.84

# Report
print(classification_report(Y_test, Y_pred))

# Confusion Matrix
print(confusion_matrix(Y_test, Y_pred))


# Standardised SVC
scaler = StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)

svc = SVC()
svc.fit(X_train_transformed,Y_train)
Y_pred = svc.predict(X_test_transformed)
accuracy_score(Y_pred, Y_test)
# 0.84

# Report
print(classification_report(Y_test, Y_pred))

# confusion matrix
print(confusion_matrix(Y_test, Y_pred))

                                       
                                             







# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 18:20:55 2021

@author: mukesh
"""
# 1 importing the horse csv file into dataframe of pandas
import pandas as pd
horse = pd.read_csv(r"C:\Users\mukes\Python\Case Study\horse.csv")

# identify if there are null values and how many of them
bool = horse.isnull()
bool_count = horse.isnull().sum()
if sum(bool_count) > 0:
    print("There are %i null values"%(bool_count.sum()))
else:
    print("There are no null values")

# 2 This dataset contains many categorical features, replace them with label encoding.
Y = horse['outcome']
X = horse.drop(['outcome'],axis=1)
X = pd.get_dummies(X)
  

# 3.Replace the missing values by the most frequent value in each column.
from sklearn.impute import SimpleImputer
import numpy as np
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(X.iloc[:,0:])
X.iloc[:,0:] = imp.transform(X.iloc[:,0:])

# 4.Fit a decision tree classifier and observe the accuracy.
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier   
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
print('Shape of X_train=>',X_train.shape)
print('Shape of X_test=>',X_test.shape)
print('Shape of Y_train=>',Y_train.shape)
print('Shape of Y_test=>',Y_test.shape)
dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
dt.fit(X_train, Y_train)
dt_pred_train = dt.predict(X_train)
print('Training Set Evaluation Score=>',metrics.accuracy_score(dt_pred_train,Y_train))
dt_pred_test = dt.predict(X_test)
print('Test Set Evaluation score =>',metrics.accuracy_score(dt_pred_test,Y_test))

# 5.Fit a random forest classifier and observe the accuracy
rfc = RandomForestClassifier(criterion='entropy',random_state=42)
rfc.fit(X_train, Y_train)
rfc_pred_train = rfc.predict(X_train)
print('Training set evaluation score',metrics.accuracy_score(rfc_pred_train,Y_train))
rfc_pred_test = rfc.predict(X_test)
print('Test set evaluation score',metrics.accuracy_score(rfc_pred_test,Y_test))

## Accuracy is better with random forest classifier 0.76 vs 0.68 in case of decision tree classifier

###  Random forest leverages the power of multiple decision trees. It does not rely on the feature importance
###  given by a single decision tree. 
###  Let’s take a look at the feature importance given by different algorithms to different features


import matplotlib.pyplot as plt
feature_importance=pd.DataFrame({
    'rfc':rfc.feature_importances_,
    'dt':dt.feature_importances_
},index=X.columns)
feature_importance.sort_values(by='rfc',ascending=True,inplace=True)

index = np.arange(len(feature_importance))
fig, ax = plt.subplots(figsize=(18,20))
rfc_feature=ax.barh(index,feature_importance['rfc'],0.4,color='purple',label='Random Forest')
dt_feature=ax.barh(index+0.4,feature_importance['dt'],0.4,color='lightgreen',label='Decision Tree')
ax.set(yticks=index+0.4,yticklabels=feature_importance.index)
ax.legend()
plt.show()

"""
As you can clearly see in the above graph, the decision tree model gives high importance to a particular 
set of features. But the random forest chooses features randomly during the training process. 
Therefore, it does not depend highly on any specific set of features. 
This is a special characteristic of random forest over bagging trees. 
"""

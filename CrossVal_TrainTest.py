
# coding: utf-8

# In[41]:


# Train-Test split and Cross validation - to prevent overfitting

import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

diabetes = datasets.load_diabetes()

df = pd.DataFrame(diabetes.data, columns = diabetes.feature_names) #Data and independent variables

y = diabetes.target #dependent variable

# 1.Train-Test
X_train, x_test, Y_train, y_test = train_test_split(df, y, test_size = 0.2) #80/20 train/test

print(X_train.shape)
print(Y_train.shape)
print(x_test.shape)
print(y_test.shape)

lm = linear_model.LinearRegression()
model  = lm.fit(X_train, Y_train)

y_pred = lm.predict(x_test)

print("R2 Score: ", model.score(x_test, y_test))

plt.scatter(y_test, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.show()


# In[40]:


# 2. Cross Validation

## K-Folds, LeaveOneOut

from sklearn.model_selection import KFold
import numpy as np
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics

# X_Train = np.array([[1,5], [2,7], [3,9], [4,11]])
# Y_Train = np.array([1,2,3,4])

# kf = KFold(n_splits = 4)
# #print(kf)

# #kf.get_n_splits(X_Train)


# for train_index, test_index in kf.split(X_Train) :
#     print(train_index, test_index)
#     X_train, x_test = X_Train[train_index], X_Train[test_index]
#     Y_train, y_test = Y_Train[train_index], Y_Train[test_index]
#     print(X_train, Y_train)
#     print(x_test, y_test)

#Runs the regression 6 times, each time diving the set into different train and test, 
#fit on train and predict test, then average out regression betas   
scores = cross_val_score(model, df, y, cv=6)

print("Cross_Val_scores:", scores)

predictions = cross_val_predict(model, df, y, cv=6)

score = metrics.r2_score(y, predictions)

print("R2 Score:",score)

plt.scatter(y, predictions)
plt.show()



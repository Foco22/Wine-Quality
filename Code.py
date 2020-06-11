
# coding: utf-8

# In[5]:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:

df = pd.read_csv('wine.csv')
df.head()


# In[8]:

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),cmap='coolwarm',annot=True)
plt.show()


# In[9]:

df.describe()


# # Logistic Regretion

# In[18]:

from sklearn.linear_model import LogisticRegression
from adspy_shared_utilities import (
plot_class_regions_for_classifier_subplot)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# In[15]:

df['quality_re'] = df['quality'].apply(lambda x: 1 if x > 4 else 0)


# In[16]:

pd.pivot_table(data=df.iloc[:,-2:],index=['quality'], aggfunc=np.sum)


# In[19]:

X_ = df.iloc[:,:-2]
y_ = df.iloc[:,-1]

X_train, X_test, y_train, y_test = (train_test_split(X_,y_,random_state = 0))

clf = LogisticRegression(C=5).fit(X_train, y_train)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


# In[20]:

y_prediction = clf.predict(X_test)
confusion = confusion_matrix(y_test, y_prediction)
confusion


# # SVC Model

# In[21]:

from sklearn.svm import LinearSVC
X_train, X_test, y_train, y_test = train_test_split(X_, y_, random_state = 0)

clf = LinearSVC().fit(X_train, y_train)
print('Breast cancer dataset')
print('Accuracy of Linear SVC classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Linear SVC classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


# In[22]:

y_prediction = clf.predict(X_test)
confusion = confusion_matrix(y_test, y_prediction)
confusion


# # SVC Normalizado

# In[24]:

from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X_, y_,
                                                   random_state = 0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = SVC(C=10).fit(X_train, y_train)
print('Breast cancer dataset (normalized with MinMax scaling)')
print('RBF-kernel SVC (with MinMax scaling) training set accuracy: {:.2f}'
     .format(clf.score(X_train_scaled, y_train)))
print('RBF-kernel SVC (with MinMax scaling) test set accuracy: {:.2f}'
     .format(clf.score(X_test_scaled, y_test)))


# In[25]:

y_prediction = clf.predict(X_test)
confusion = confusion_matrix(y_test, y_prediction)
confusion


# ## Cross validation

# In[ ]:

scaler = MinMaxScaler()
X_ = scaler.fit_transform(X_)

clf = SVC(C=10)
X = X_
y = y_.as_matrix()
cv_scores = cross_val_score(clf, X, y)

print('Cross-validation scores (3-fold):', cv_scores)
print('Mean cross-validation score (3-fold): {:.3f}'
     .format(np.mean(cv_scores)))


# # Prediction Quality by LogisticRegression

# In[79]:

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

X_=df.iloc[:,:-2]
y_=df.iloc[:,-2]


mean = np.mean(X_)
std = np.std(X_)
X_t = (X_ - mean)/std
X_t

X_train, X_test, y_train, y_test = train_test_split(X_t, y_,
                                                   random_state = 0)
#scaler = MinMaxScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

#Logistic Regression 

log_reg_params = {"penalty": ['l2'], 'C': [10]}
grid_log_reg = GridSearchCV(LogisticRegression(), param_grid = log_reg_params)

log_reg= grid_log_reg.fit(X_train, y_train)
        
#We automatically get the logistic regression with the best parameters.
clf_prediction = log_reg.predict(X_test)
print("Accuracy of Logictic Regretion",":",log_reg.score(X_test,y_test))
#print confusion matrix and accuracy score before best parameters
clf1_conf_matrix = confusion_matrix(y_test, clf_prediction)
print("Confusion matrix of Logictic Regretion",":\n", clf1_conf_matrix)
print("==========================================")


# In[78]:

cv_scores = cross_val_score(log_reg, X_t, y_)

print('Cross-validation scores (3-fold):', cv_scores)
print('Mean cross-validation score (3-fold): {:.3f}'
     .format(np.mean(cv_scores)))


# In[ ]:




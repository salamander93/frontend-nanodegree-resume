
# coding: utf-8

# In[1]:

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn import cross_validation
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score
import xgboost


# In[2]:

os.chdir("/Users/ignizious/downloads")


# In[3]:

train = pd.read_csv("train_loan.csv")


# In[4]:

test = pd.read_csv("test_loan.csv")


# In[5]:

#splitting
y_train = np.array(train['default_ind'])  
y_test = np.array(test['default_ind'])
X_train = train.drop('default_ind', axis = 1)  
X_train = np.array(X_train)
X_test = test.drop('default_ind', axis = 1)
X_test = np.array(X_test)


# In[6]:

# normalizing data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)


# In[7]:

scaler=StandardScaler()
scaler.fit(X_test)
X_test=scaler.transform(X_test)


# In[8]:

scaler=StandardScaler()
scaler.fit(X_test)
X_test=scaler.transform(X_test)


# In[9]:

y_train = y_train.astype(int)
y_test = y_test.astype(int)


# In[10]:

#up_sampling 
from sklearn.utils import resample


# In[11]:

# Separate majority and minority classes for train
df_majority_train = train[train.default_ind==0]
df_minority_train = train[train.default_ind==1]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority_train, 
                                 replace=True,     # sample with replacement
                                 n_samples=199733,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled_train = pd.concat([df_majority_train, df_minority_upsampled])
 
# Display new class counts
df_upsampled_train.default_ind.value_counts()


# In[12]:

from sklearn.linear_model import LogisticRegression
# Separate input features (X) and target variable (y)
y_train1 = df_upsampled_train.default_ind
X_train1 = df_upsampled_train.drop('default_ind', axis=1)
y_train1 = np.array(y_train1)
X_train1 = np.array(X_train1)


# In[13]:

classifier=(LogisticRegression())
#fitting traing data to model
classifier.fit(X_train1,y_train1)
Y_pred=classifier.predict(X_test)


# In[14]:

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(y_test,Y_pred)
print(cfm)
print("Classification report :")
print(classification_report(y_test,Y_pred))
accuracy_score=accuracy_score(y_test,Y_pred)
print("Accuracy of the model:",accuracy_score)
#auc curve 
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, Y_pred)
auc(false_positive_rate, true_positive_rate)


# In[15]:

#%% Adjusting threshold
#store the predicted probabilities
y_pred_prob = classifier.predict_proba(X_test)
print(y_pred_prob)


# In[16]:

y_pred_class=[]
for value in y_pred_prob[:,0]:
    if value < 0.5:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)


# In[17]:

from sklearn.metrics import confusion_matrix,accuracy_score
cfm=confusion_matrix(y_test.tolist(),y_pred_class)
print(cfm)
accuracy_score=accuracy_score(y_test.tolist(),y_pred_class)
print("Accuracy of the model: ",accuracy_score)
print("Classification report :")
print(classification_report(y_test,y_pred_class))
#auc curve 
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_class)
auc(false_positive_rate, true_positive_rate)


# In[ ]:




# In[18]:

#decision tree
from sklearn.tree import DecisionTreeClassifier

model_DecisionTree=DecisionTreeClassifier()
model_DecisionTree.fit(X_train1,y_train1)


# In[19]:

Y_pred=model_DecisionTree.predict(X_test)


# In[20]:

from sklearn.metrics import confusion_matrix,accuracy_score
cfm=confusion_matrix(y_test.tolist(),Y_pred)
print(cfm)
accuracy_score=accuracy_score(y_test.tolist(),Y_pred)
print("Accuracy of the model: ",accuracy_score)
print("Classification report :")
print(classification_report(y_test,Y_pred))
#auc curve 
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, Y_pred)
auc(false_positive_rate, true_positive_rate)


# In[21]:

#using xgboost
import xgboost


# In[28]:

from xgboost import XGBRegressor
my_model = XGBRegressor(learning_rate =0.07,subsample=0.5,colsample_bytree=0.85,reg_alpha=0.005,min_child_weight = 5)


# In[29]:

my_model.fit(X_train1, y_train1, verbose=False)


# In[30]:

test_y_pred = my_model.predict(X_test)
auc = roc_auc_score(y_test, test_y_pred)
print("Performance  test : ", auc)


# In[ ]:





# coding: utf-8

# In[98]:

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


# In[99]:

os.chdir("/Users/ignizious/downloads")


# In[100]:

train = pd.read_csv("train_loan.csv")


# In[101]:

test = pd.read_csv("test_loan.csv")


# In[102]:

#splitting
y_train = np.array(train['default_ind'])  
y_test = np.array(test['default_ind'])
X_train = train.drop('default_ind', axis = 1)  
X_train = np.array(X_train)
X_test = test.drop('default_ind', axis = 1)
X_test = np.array(X_test)


# In[103]:

# normalizing data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)


# In[104]:

scaler=StandardScaler()
scaler.fit(X_test)
X_test=scaler.transform(X_test)


# In[105]:

scaler=StandardScaler()
scaler.fit(X_test)
X_test=scaler.transform(X_test)


# In[106]:

y_train = y_train.astype(int)
y_test = y_test.astype(int)


# In[107]:

#logistic regression


# In[108]:

# applying logistic model
from sklearn.linear_model import LogisticRegression

classifier=(LogisticRegression())
classifier.fit(X_train,y_train)


# In[109]:

Y_pred=classifier.predict(X_test)
Y_pred


# In[110]:

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


# In[111]:

#%% Adjusting threshold
#store the predicted probabilities
y_pred_prob = classifier.predict_proba(X_test)
print(y_pred_prob)


# In[112]:

y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value < 0.4:
        y_pred_class.append(0)
    else:
        y_pred_class.append(1)


# In[113]:

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


# In[114]:

#decision tree


# In[115]:

from sklearn.tree import DecisionTreeClassifier

model_DecisionTree=DecisionTreeClassifier()
model_DecisionTree.fit(X_train,y_train)


# In[116]:

Y_pred=model_DecisionTree.predict(X_test)


# In[117]:

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


# In[118]:

#Xgboost


# In[119]:

#using xgboost
import xgboost


# In[120]:

from xgboost import XGBRegressor
my_model = XGBRegressor(learning_rate =0.6,subsample=0.5,colsample_bytree=0.85,reg_alpha=0.005,min_child_weight = 5,n_estimators=300)


# In[121]:

my_model.fit(X_train, y_train, verbose=False)


# In[122]:

test_y_pred = my_model.predict(X_test)
auc = roc_auc_score(y_test, test_y_pred)
print("AUC  test : ", auc)


# In[123]:

train_y_pred = my_model.predict(X_train)
auc = roc_auc_score(y_train, train_y_pred)
print("AUC  train : ", auc)


# In[70]:

fpr, tpr, threshold = metrics.roc_curve(y_test, test_y_pred)
roc_auc = metrics.auc(fpr, tpr)


# In[71]:

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.5f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:




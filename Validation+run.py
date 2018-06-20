
# coding: utf-8

# In[1]:

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score


# In[2]:

os.chdir("/Users/ignizious/downloads")


# In[3]:

df2 = pd.read_csv("df_val_clean.csv")


# In[4]:

train = pd.read_csv("train_loan.csv")
test = pd.read_csv("test_loan.csv")


# In[ ]:




# In[7]:

test.info()


# In[8]:

#splitting
y_train = train['default_ind']
y_train = y_train.as_matrix()
y_test = test['default_ind']
y_test = y_test.as_matrix()
X_train = train.drop('default_ind', axis = 1)  
X_train = X_train.as_matrix()
X_test = test.drop('default_ind', axis = 1)
X_test = X_test.as_matrix()


# In[9]:

# normalizing data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)


# In[10]:

scaler=StandardScaler()
scaler.fit(X_test)
X_test=scaler.transform(X_test)


# In[11]:

y_train = y_train.astype(int)
y_test = y_test.astype(int)


# In[12]:

df2=df2.drop(["default_ind","issue_d"],axis=1)


# In[13]:

df3 = df2.as_matrix()


# In[14]:

#validation


# In[15]:

#using xgboost
import xgboost


# In[41]:

from xgboost import XGBRegressor
my_model = XGBRegressor(learning_rate =0.08,subsample=0.5,colsample_bytree=0.85,reg_alpha=0.005,min_child_weight = 5)


# In[42]:

my_model.fit(X_train, y_train, verbose=False)


# In[43]:

test_y_pred = my_model.predict(df3)


# In[44]:

df5= pd.DataFrame(test_y_pred, columns= ['loan_status'])


# In[58]:

y_pred_class=[]
for value in df5.loan_status:
    if value > 0.5:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)


# In[59]:

df86 = pd.DataFrame(y_pred_class,columns= ['loan_status'])


# In[60]:

df_c = pd.concat([df2, df86], axis=1)
df_c


# In[61]:

df_c.loan_status.value_counts()


# In[40]:

595179/128


# In[2]:

1817/593490


# In[62]:

411258/184049


# In[ ]:




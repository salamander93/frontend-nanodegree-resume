
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

df = pd.read_csv("XYZCorp_LendingData.txt",delimiter="\t",low_memory=False)


# In[5]:

#the current loans are also added in the non defaulter list, which is not a good data to get insights from
#hence we are removing those rows 
df1 = df[df.out_prncp == 0] # 2.5 lcs


# In[175]:

#removing all the working data 
df1 = df1.drop(["addr_state","last_credit_pull_d","title","il_util", "max_bal_bc","total_bal_il","collection_recovery_fee", "collections_12_mths_ex_med", "desc", "funded_amnt", "funded_amnt_inv", "earliest_cr_line", "id", "member_id", "sub_grade", "pymnt_plan", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "out_prncp", "out_prncp_inv","emp_title", "zip_code", "recoveries", "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "mths_since_rcnt_il", "inq_last_6mths","initial_list_status","total_acc"],axis=1)


# In[176]:

df1.info()


# In[177]:

#dropping any coulumns with more than 50% na values as the dataset is too big to go for 75% rate
df1 = df1.dropna(thresh=0.5*len(df1),axis=1)


# In[178]:

df1.info()


# In[179]:

#checking categorical columns with less than 1 unique value 
for col in df1.columns:
    if (len(df1[col].unique()) < 3):
        print(df1[col].value_counts())
        print()


# In[180]:

#removing policy code, for a single unique value
#removing application_type, due to the distribution being to baised
df1 = df1.drop(["policy_code","application_type"],axis=1)


# In[181]:

#outlier detection and treatment
#checking all numerical values, we decided to treat (total_rev_hi_lim,tot_coll_amt,tot_cur_bal,revol_bal and revol_util)


# In[182]:

def outliers(x):
    m = x.mean()
    s = np.std(x)
    lower_c = m-(3*s)
    upper_c = m+(3*s)
    n_upper = (x>upper_c).sum()
    n_lower = (x<lower_c).sum()
    val =  [lower_c, upper_c,n_upper,n_lower]
    return(val)


# In[183]:

outliers(df1.total_rev_hi_lim)


# In[184]:

df1.total_rev_hi_lim = np.where(df1.total_rev_hi_lim >118253.1295408675, df1.total_rev_hi_lim.quantile(0.95),df1.total_rev_hi_lim)


# In[185]:

outliers(df1.tot_coll_amt)


# In[186]:

df1.tot_coll_amt = np.where(df1.tot_coll_amt >63499.92403452018, df1.tot_coll_amt.quantile(0.95),df1.tot_coll_amt)


# In[187]:

outliers(df1.tot_cur_bal)


# In[188]:

df1.tot_cur_bal = np.where(df1.tot_cur_bal >595621.337879223, df1.tot_cur_bal.quantile(0.95),df1.tot_cur_bal)


# In[189]:

outliers(df1.revol_bal)


# In[190]:

df1.revol_bal = np.where(df1.revol_bal >71393.19766446359, df1.revol_bal.quantile(0.95),df1.revol_bal)


# In[191]:

df1 = df1[df1.annual_inc <254618.82060287986 ]


# In[192]:

df1 = df1[df1.revol_util < 101]


# In[193]:

###checking the null values


# In[194]:

#checking and imputing missing values
df1.isnull().sum()


# In[195]:

#all the missing data columns are numerical
#columns with missing data
#total_rev_hi_lim,tot_cur_bal,tot_coll_amt,revol_util


# In[196]:

df1.revol_util.fillna(df1.revol_util.mean(),inplace=True) 


# In[197]:

df1.total_rev_hi_lim.fillna(df1.total_rev_hi_lim.mean(),inplace=True)


# In[198]:

df1.tot_coll_amt.fillna(df1.tot_coll_amt.mean(),inplace=True)


# In[199]:

df1.tot_cur_bal.fillna(df1.tot_cur_bal.mean(),inplace=True) 


# In[200]:

df1.to_csv("non_active_data.csv",index=False)


# In[201]:

df1.info()


# In[ ]:




# In[202]:

#ordinal columns 
df1.emp_length.value_counts()


# In[203]:

mapped = {
    "emp_length":{
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0

    },
    "grade":{
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6,
        "G": 7
    }
}


# In[204]:

df1 = df1.replace(mapped)


# In[205]:

#ALL categorical columns
obj_cols = ["verification_status","purpose","term","home_ownership"]


# In[206]:

# for preprocessing the data converting category to numerical
from sklearn import preprocessing
le={}

for x in obj_cols:
    le[x]=preprocessing.LabelEncoder()
    
for x in obj_cols:
    df1[x]=le[x].fit_transform(df1.__getattr__(x))
df1.head()


# In[207]:

df1.to_csv("pre_split.csv",index=False)


# In[208]:

#splitting data in train and test


# In[209]:

#data is supposed to be split according to gives dates


# In[210]:

df1.issue_d.unique()


# In[211]:

#array for test data and using isin function
a = ['Dec-2015', 'Nov-2015', 'Oct-2015', 'Sep-2015',
       'Aug-2015', 'Jul-2015', 'Jun-2015']
test = df1.loc[df1['issue_d'].isin(a)]


# In[212]:

train = df1.loc[~(df1['issue_d'].isin(a))]


# In[213]:

train.info()


# In[214]:

#dropping the date column
train = train.drop(["issue_d"],axis=1)
test = test.drop(["issue_d"],axis=1)


# In[215]:

#splitting
y_train = np.array(train['default_ind'])  
y_test = np.array(test['default_ind'])
X_train = train.drop('default_ind', axis = 1)  
X_train = np.array(X_train)
X_test = test.drop('default_ind', axis = 1)
X_test = np.array(X_test)


# In[216]:

# normalizing data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)


# In[217]:

scaler=StandardScaler()
scaler.fit(X_test)
X_test=scaler.transform(X_test)


# In[218]:

y_train = y_train.astype(int)
y_test = y_test.astype(int)


# In[219]:

#trying metrics package to find the pvalues


# In[220]:

est = sm.OLS(y_train, X_train)
est2 = est.fit()
print(est2.summary())


# In[221]:

train.columns


# In[222]:

train = train.drop(["tot_coll_amt","revol_bal","acc_now_delinq"],axis=1)
test = test.drop(["tot_coll_amt","revol_bal","acc_now_delinq"],axis=1)


# In[223]:

#checking for multicollinearity 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# In[224]:

df3 = df1._get_numeric_data()
X = add_constant(df3)
pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)


# In[225]:

df1 = df1.drop(["installment","int_rate"],axis=1)


# In[110]:

#checking vif again after removing installement 
df3 = df1._get_numeric_data()
X = add_constant(df3)
pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)


# In[226]:

train = train.drop(["installment","int_rate"],axis=1)
test = test.drop(["installment","int_rate"],axis=1)


# In[227]:

train.to_csv("train_loan.csv",index=False)


# In[228]:

test.to_csv("test_loan.csv",index=False)


# In[143]:


#
#loan_status and loan_amount
plt.figure(figsize = (19,19))
plt.subplot(313)
g2 = sns.boxplot(y="revol_util", data=df1)
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)
g2.set_xlabel("Duration Distribuition", fontsize=15)
g2.set_ylabel("Count", fontsize=15)
g2.set_title("Loan Amount", fontsize=20)
plt.show()


# In[6]:

df1.default_ind.value_counts()


# In[7]:

207781/45249


# In[8]:

df.default_ind.value_counts()


# In[9]:

809502/46467


# In[ ]:




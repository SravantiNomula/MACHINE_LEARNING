#!/usr/bin/env python
# coding: utf-8

# In[88]:



# Ignoring all warnings throughout the notebook
import warnings
warnings.filterwarnings('ignore')


# In[89]:


import re
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
import xgboost as xgb

from collections import Counter
import sklearn

# Scikit-learn libraries for model building
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# Scikit-learn libraries for evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import *


sklearn.__version__


# In[90]:



data = pd.read_csv("train.csv", parse_dates=['DateTime'])
data.head()


# In[91]:


data.shape


# In[92]:


test = pd.read_csv("test.csv", parse_dates=['DateTime'])
test.head()


# In[93]:


test.shape


# # DateTime splitting

# In[94]:




test['Year'] = test['DateTime'].map(lambda x: x.year)
test['Month'] = test['DateTime'].map(lambda x: x.month)
test['Date'] = test['DateTime'].map(lambda x: x.day)
test['Hour'] = test['DateTime'].map(lambda x: x.hour)
test['Minute'] = test['DateTime'].map(lambda x: x.minute)

data['Year'] = data['DateTime'].map(lambda x: x.year)
data['Month'] = data['DateTime'].map(lambda x: x.month)
data['Date'] = data['DateTime'].map(lambda x: x.day)
data['Minute'] = data['DateTime'].map(lambda x: x.minute)
data['Hour'] = data['DateTime'].map(lambda x: x.hour)


# In[95]:


data=data.drop(labels=['DateTime'],axis=1)
test=test.drop(labels=['DateTime'],axis=1)


# In[96]:


data.head()


# # Exploratory data analysis
# 

# We start with exploratory data analysis. First, we look to see if we have outliers or missing data for each feature.

# In[97]:


# Describe the dataset.
data.head()


# In[98]:



data_counts = pd.DataFrame({'Column Count' : data.count(),
   ....:                      '% Percent of Total Data' : data.count().div(max(data.count())) * 100})

data_counts.round(1)


# Our initial look at the data reveals 26,729 rows. Name is included for 71.2% of the data. 
# A large quantity of OutcomeSubtype data is missing, which we worked to quantify below. 
# Aside from these two columns, we had at least 99.9% of the data in every other column.
# We are choosing not to use the Name variable moving forward.We feel that this level of sparsity within Name will not help us predict animal outcomes. 

# # Class Variable

# In[99]:


# Obtain headers.
headers = data.dtypes.index

# Categorical headers of classes.
cat_headers = ['OutcomeType', 'OutcomeSubtype']

# Plot paretos of top 10 values in each column.
fig, ax = plt.subplots(len(cat_headers[0:]), 1, figsize=(10, len(cat_headers)*12))
for i, column in enumerate(cat_headers[0:]):
    to_plot=data[column].value_counts().head(10)
    plot=sns.barplot(x=to_plot.index, y=to_plot, ax=ax[i])
    plot.set_xticklabels(to_plot.index, rotation=90)
    plot.set_title(column)
    plot.set_ylabel("Number of dogs/cats")
    plt.figure(figsize=(10,5))
fig.show()


# In[100]:


data.groupby(['OutcomeType', 'OutcomeSubtype']).size().reset_index(name='Count')


# In[101]:


data_sub = data[data['OutcomeSubtype'].isnull()]
data_sub.head()


# In[102]:


data_sub.shape


# In[103]:


data_sub.groupby(['OutcomeType']).size().reset_index(name='Count')


# From the data in the table above, we surmise that it is possible the majority of the subtype null data is from the Return to Owner or Adoption outcome type data. We will examine this further below.

# Of all the null entries in OutcomeSubtype,
# 
# 8803 are of OutcomeType- Adoption,
# 4786 are of OutcomeType- Return_to_owner,
# 16   are of OutcomeType- Died,
# 6    are of OutcomeType- Transfer,
# 1    is of OutcomeType - Euthanasia.
# 
# As we already have outcome subtypes for barn, foster, and offsite within adoption, we feel confident in assuming that these missing subtypes are simply the animals that were adopted. We will assume this is true and fill these null values with adopted.
# 
# As the return-to-owner outcome type also has no subtype, we will make the outcome subtype return to owner.
# 
# For the remaining 23 null outcome subtypes, we will label their subtype as other.

# In[104]:


def subtype(x):
    if x['OutcomeType'] == 'Adoption' and pd.isnull(x['OutcomeSubtype']) : return 'Adopted'
    elif x['OutcomeType'] == 'Return_to_owner' and pd.isnull(x['OutcomeSubtype']): return 'Return to Owner'
    elif pd.isnull(x['OutcomeSubtype']) : return 'Other'
    else: return x['OutcomeSubtype']

data['OutcomeSubtype'] = data.apply(subtype, axis=1)


# In[105]:


data.groupby(['OutcomeType', 'OutcomeSubtype']).size().reset_index(name='Count')


# # Feature Review

# In[106]:


# Obtain headers.
headers = data.dtypes.index

# Not all of these are categorical variables.  
# We need categorical headers and continuous headers, with two different types of plots.

# Categorical headers of features.
cat_headers = ['Name', 'AnimalType', 'SexuponOutcome', 'Breed', 'Color']

# Plot paretos of top 10 values in each column.
fig, ax = plt.subplots(len(cat_headers[0:]), 1, figsize=(10, len(cat_headers)*12))
for i, column in enumerate(cat_headers[0:]):
    to_plot=data[column].value_counts().head(10)
    plot=sns.barplot(x=to_plot.index, y=to_plot, ax=ax[i])
    plot.set_xticklabels(to_plot.index, rotation=90)
    plot.set_title(column)
    plot.set_ylabel("Number of dogs/cats")
fig.show()


# In[107]:


data_sex_dropna = data.dropna(subset=['SexuponOutcome'])
data_sex_outcome = data_sex_dropna.groupby(['OutcomeType', 'SexuponOutcome']).agg({'AnimalType':'size'}).rename(columns={'AnimalType':'# of Animals'}).reset_index()
data_sex_outcome


# In[108]:


def convert_AgeuponOutcome_to_weeks(data):
    result = {}
    for k in data['AgeuponOutcome'].unique():
        if type(k) != type(""):
            result[k] = -1
        else:
            v1, v2 = k.split()
            if v2 in ["year", "years"]:
                result[k] = int(v1) * 1
            elif v2 in ["month", "months"]:
                result[k] = int(v1) / 12
            elif v2 in ["week", "weeks"]:
                result[k] = int(v1) / 52
            elif v2 in ["day", "days"]:
                result[k] = int(v1) / 365
                
    data['_AgeuponOutcome'] = data['AgeuponOutcome'].map(result).astype(float)
    data = data.drop('AgeuponOutcome', axis = 1)
                
    return data

data = convert_AgeuponOutcome_to_weeks(data)

test = convert_AgeuponOutcome_to_weeks(test)


# Next, we look into the Age Upon Outcome feature variable. We believed that it is important to make the ages variable continuous, as opposed to categorical, as the relationship of age is continuous in nature. The most logical unit to standardize ages is weeks. 

# In[109]:


sns.distplot(data._AgeuponOutcome,bins=20, kde=False)


# From the information above, we can see that a more likely outcome for younger animals is to be transfered or adopted, while older animals are more likely to be returned to their owner or euthanized.

# In[110]:


def calc_age_category(x):
    if x  <3: return 'young'
    if x  <5 and x>=3: return 'young adult'
    if x  <10 and x>=5: return 'adult'
    return 'old'
data['_AgeuponOutcome']=data._AgeuponOutcome.apply(calc_age_category)

test['_AgeuponOutcome']=test._AgeuponOutcome.apply(calc_age_category)


# In[111]:


data1_counts = pd.DataFrame({'Column Count' : data.count(),
                     '% Percent of Total Data' : data.count().div(max(data.count())) * 100})

data1_counts.round(1)


# In[112]:


data['Name']=data['Name'].fillna('Timon')


# Naming the unnamed dogs,cats as TIMON

# In[ ]:





# # Breed

# In[113]:


print ('Total number of breeds:', len(data['Breed'].unique()))
print ('Number of cat breeds:', len(data[data['AnimalType'] == 'Cat']['Breed'].unique()))
print ('Number of dog breeds:', len(data[data['AnimalType'] == 'Dog']['Breed'].unique()))


# Given the massive number of breeds (mostly for dogs) we may overfit w/ regard to breed. Here is the distribution of the (log) number of animals by breed for both animals. We can see there are a small number of breeds containing a large number of animals, and then many breeds with a very small number of animals.

# In[114]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()


# In[115]:


Cat=data['OutcomeType']
le_res=le.fit_transform(data['OutcomeType'])
y=pd.DataFrame(le_res)
y.columns=['OutcomeType']


# In[116]:


x=data.Breed.str.contains(pat="Mix", case=True,regex=True)
Breed=pd.DataFrame(x)
le_res=le.fit_transform(Breed)
Breed=pd.DataFrame(le_res)
Breed.columns=['Breed']
data=data.drop(labels=['Breed'],axis=1)
data=pd.concat([Breed,data],axis=1)


# In[117]:


data.head


# # LabelEncoding

# In[118]:


#Train set
le_res=le.fit_transform(data['_AgeuponOutcome'])
_AgeuponOutcome=pd.DataFrame(le_res)
_AgeuponOutcome.columns=['_AgeuponOutcome']
data=data.drop(labels=['_AgeuponOutcome'],axis=1)
data=pd.concat([_AgeuponOutcome,data],axis=1)

#Test set
le_res_test=le.fit_transform(test['_AgeuponOutcome'])
_AgeuponOutcome_test=pd.DataFrame(le_res_test)
_AgeuponOutcome_test.columns=['_AgeuponOutcome']
test=test.drop(labels=['_AgeuponOutcome'],axis=1)
test=pd.concat([_AgeuponOutcome_test,test],axis=1)


# In[119]:


#Train set 
le_res=le.fit_transform(data['Breed'])
Breed=pd.DataFrame(le_res)
Breed.columns=['Breed']
data=data.drop(labels=['Breed'],axis=1)
data=pd.concat([Breed,data],axis=1)

#Test set
le_res_test=le.fit_transform(test['Breed'])
Breed_test=pd.DataFrame(le_res_test)
Breed_test.columns=['Breed']
test=test.drop(labels=['Breed'],axis=1)
test=pd.concat([Breed_test,test],axis=1)


# In[120]:


data["SexuponOutcome"]=data["SexuponOutcome"].fillna("Unknown")

test["SexuponOutcome"]=test["SexuponOutcome"].fillna("Unknown")


# In[121]:


#Train set
le_res=le.fit_transform(data['SexuponOutcome'])
SexuponOutcome=pd.DataFrame(le_res)
SexuponOutcome.columns=['SexuponOutcome']
data=data.drop(labels=['SexuponOutcome'],axis=1)
data=pd.concat([SexuponOutcome,data],axis=1)


# In[122]:


#Test set
le_res_test=le.fit_transform(test['SexuponOutcome'])
SexuponOutcome_test=pd.DataFrame(le_res_test)
SexuponOutcome_test.columns=['SexuponOutcome']
test=test.drop(labels=['SexuponOutcome'],axis=1)
test=pd.concat([SexuponOutcome_test,test],axis=1)


# In[123]:


data['AnimalType']=data['AnimalType'].apply(lambda x:1 if x in('Dog') else 0)

test['AnimalType']=test['AnimalType'].apply(lambda x:1 if x in('Dog') else 0)


# In[124]:


data['Month']=data['Month'].apply(lambda x : 0 if x <4 else (1 if x <8 else (2)))


# In[125]:


data['OutcomeType']=le.fit_transform(data['OutcomeType'])


# In[126]:


corre=data.corr()
corre


# In[127]:


sns.heatmap(corre,linewidths=2, cmap="coolwarm")


# # Dropping Unnecesary Columns

# In[128]:


data=data.drop(labels=['AnimalID'],axis=1)


# In[129]:


data=data.drop(labels=['OutcomeType'],axis=1)


# In[130]:


data=data.drop(labels=['OutcomeSubtype','Color','Name'],axis=1)

test=test.drop(labels=['Color','Name'],axis=1)


# In[131]:


data.head()


# # Splitting train and test

# In[132]:


X=data


# In[133]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)


# # Importance of Features

# # Logistic Regression 

# In[169]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
classifier = LogisticRegression(penalty='l2',random_state = 0,class_weight='balanced',multi_class='multinomial', 
                                solver='lbfgs',n_jobs=-1)
selector = RFE(classifier, 5, step=1)
selector = selector.fit(X, y)
selector.support_
selector.ranking_


# In[135]:


from sklearn.linear_model import LogisticRegression


# In[136]:


import pickle


# In[137]:


filename = 'logistic_model.sav'
pickle.dump(classifier, open(filename, 'wb'))


# In[138]:


loaded_model = pickle.load(open(filename, 'rb'))


# In[139]:


loaded_model.fit(X_train,y_train)


# In[140]:


y_pred=loaded_model.predict_proba(X_test)


# In[141]:


ll = log_loss(y_test,y_pred)
ll


# # XGBOOST

# In[170]:


from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.feature_selection import RFE
model = xgb.XGBClassifier(objective='multi:softprob')
selector = RFE(model, 5, step=1)
selector = selector.fit(X, y)
selector.support_
selector.ranking_


# In[143]:


from xgboost import XGBClassifier
import xgboost as xgb


# In[144]:


filename = 'xgboost_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[145]:


loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.fit(X_train, y_train)


# In[146]:


y_pred=loaded_model.predict_proba(X_test)


# In[147]:


ll = log_loss(y_test,y_pred)
ll


# In[148]:


xgb.plot_importance(loaded_model)


# # XGBOOST PARAMETER TUNING

# In[149]:


HYPER_PARAMS = { 
 'learning_rate': 0.20,
 'n_estimators':0,
 'max_depth': 5,
 'subsample': 0.7,
 'colsample_bytree': 0.9,
 'max_delta_step': 1,
 'objective': 'multi:softmax',
 'nthread': 1,
 }


# In[150]:


filename = 'xgboostparam_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[151]:


loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.fit(X_train,y_train)


# In[152]:


y_pred=loaded_model.predict_proba(X_test)


# In[153]:


print (log_loss(y_test,y_pred));


# # Naive Bayes

# In[174]:


from sklearn.feature_selection import RFE
from sklearn.naive_bayes import GaussianNB
nv = GaussianNB()


# In[175]:


filename = 'naive_model.sav'
pickle.dump(nv, open(filename, 'wb'))


# In[176]:


loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.fit(X_train,y_train)


# In[177]:


y_pred=loaded_model.predict_proba(X_test)


# In[178]:


ll = log_loss(y_test,y_pred)
ll


# # LightGBM

# In[171]:


from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE
lgbm = LGBMClassifier(objective='multiclass', random_state=5)
selector = RFE(lgbm, 5, step=1)
selector=selector.fit(X, y)
selector.support_
selector.ranking_


# In[160]:


filename = 'lgbm_model.sav'
pickle.dump(lgbm, open(filename, 'wb'))


# In[161]:


lgbmm = pickle.load(open(filename, 'rb'))
lgbmm.fit(X_train,y_train)


# In[162]:


y_pred=lgbmm.predict_proba(X_test)


# In[163]:


ll=log_loss(y_test,y_pred)
ll


# # Random Forest

# In[172]:


from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=4,n_estimators =20,criterion = 'entropy',random_state = 0)
selector = RFE(clf, 5, step=1)
selector = selector.fit(X, y)
selector.support_
selector.ranking_


# In[165]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=10,n_estimators =30,criterion = 'entropy',random_state = 0)
clf.fit(X_train,y_train)


# In[166]:


filename = 'random_model.sav'
pickle.dump(clf, open(filename, 'wb'))


# In[167]:


loaded_model = pickle.load(open(filename, 'rb'))
y_pred=loaded_model.predict_proba(X_test)


# In[168]:


ll=log_loss(y_test,y_pred)
ll


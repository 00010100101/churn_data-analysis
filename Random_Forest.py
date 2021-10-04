#!/usr/bin/env python
# coding: utf-8

# In[436]:


import pandas as pd


# In[437]:


import matplotlib.pyplot as plt


# In[438]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[439]:


import numpy as np


# In[440]:


train=pd.read_csv("train.csv")


# In[441]:


NAs = pd.concat([train.isnull().sum()],axis=1,keys=['Train'])
NAs[NAs.sum(axis=1)>0]                


# In[442]:


train


# In[443]:


from sklearn.compose import ColumnTransformer


# In[444]:


import matplotlib as mpl


# In[445]:


for col in train.dtypes[train.dtypes == "object"].index:
    for_dummy=train.pop(col)
    train=pd.concat([train,pd.get_dummies(for_dummy,prefix=col)],axis=1)


# In[446]:


train.head()


# In[447]:


train.drop(['subscribed_no'],axis=1)


# In[448]:


train.rename({'subscribed_yes':'subscribed'},axis=1)


# In[449]:


train.drop(['subscribed_no'],axis=1)


# In[450]:


train2=train.rename({'subscribed_yes':'subscribed'},axis=1)


# In[451]:


train2


# In[452]:


train3=train2.drop(['subscribed_no'],axis=1)


# In[453]:


train3


# In[454]:


labels=train3.pop("subscribed")


# In[455]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train3,labels,test_size=0.20)


# In[456]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=72)
rf.fit(x_train, y_train)


# In[457]:


y_pred=rf.predict(x_test)


# In[458]:


from sklearn.metrics import roc_curve,auc
false_positive_rate,true_positive_rate,thresholds=roc_curve(y_test,y_pred)
roc_auc=auc(false_positive_rate,true_positive_rate)
roc_auc


# In[459]:


n_estimators=[1,2,4,8,32,64,100,200,500,800,860]
train_results=[]
test_results=[]

for estimator in n_estimators:
    rf=RandomForestClassifier(n_estimators=estimator,n_jobs=-1)
    rf.fit(x_train,y_train)
    train_pred=rf.predict(x_train)
    false_positive_rate,true_positive_rate,threshold=roc_curve(y_train,train_pred)
    roc_auc=auc(false_positive_rate,true_positive_rate)
    train_results.append(roc_auc)
    y_pred=rf.predict(x_test)
    false_positive_rate,true_positive_rate,thresholds=roc_curve(y_test,y_pred)
    roc_auc=auc(false_positive_rate,true_positive_rate)
    test_results.append(roc_auc)
    
from matplotlib.legend_handler import HandlerLine2D
line1,=plt.plot(n_estimators,train_results,"b",label="Train AUC")
line2,=plt.plot(n_estimators,test_results,"r",label="Test AUC")
plt.legend(handler_map={line1:HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("n_estimators")
plt.show()


# In[463]:


rf.score(x_test,y_test)


# In[464]:


from sklearn.metrics import confusion_matrix


# In[465]:


confusion_matrix(y_test,y_pred)


# In[ ]:





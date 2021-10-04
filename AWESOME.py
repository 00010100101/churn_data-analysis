#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


import numpy as np


# In[7]:


train=pd.read_csv("train.csv")


# In[8]:


train


# In[9]:


train.head()


# In[10]:


test=pd.read_csv("test.csv")


# In[11]:


test


# In[12]:


test.head()


# In[13]:


train.shape

test.shape
# In[14]:


test.shape


# In[15]:


train.iloc[:,:3]


# In[16]:


train['job']


# In[17]:


train[['balance','duration']].corr()


# In[18]:


import seaborn as sns


# In[19]:


train.corr()


# In[20]:


train.isnull().sum()


# In[21]:


train['balance'].plot.box()


# In[22]:


train.plot.scatter('balance','duration')


# In[23]:


train3=train[train['duration']<4000]


# In[24]:


train2=train[train['balance']<6000]


# In[25]:


train3.plot.scatter('balance','duration')


# In[26]:


train3=train[train['balance']>-6000]


# In[27]:


train3=train[train['duration']<4000]


# In[28]:


train3.plot.scatter('balance','duration')


# In[29]:


train3=train3[train3['balance']<8000]


# In[30]:


train3.plot.scatter('balance','duration')


# In[31]:


train3=train[train['balance']>-4000]


# In[32]:


train3.plot.scatter('balance','duration')


# In[33]:


train4=train[train['balance']<8000]


# In[34]:


train4.plot.scatter('balance','duration')


# In[35]:


train3=train3[train3['duration']<4000]


# In[36]:


train3.plot.scatter('balance','duration')


# In[37]:


test.plot.scatter('balance','duration')


# In[38]:


test=test[test['duration']<2500]


# In[39]:


test=test[test['balance']<6000]


# In[40]:


test.plot.scatter('balance','duration')


# In[41]:


train4['campaign'].plot.box()


# In[42]:


train4.plot.scatter('campaign','pdays')


# In[43]:


train4=train4[train4['campaign']<40]


# In[44]:


test.plot.scatter('campaign','pdays')


# In[45]:


test=test[test['campaign']<40]


# In[46]:


train4['previous'].plot.box()


# In[47]:


train4=train4[train4['previous']<40]


# In[48]:


train4['age'].plot.box()


# In[49]:


train4.plot.scatter('age','pdays')


# In[50]:


train4['day'].plot.box()


# In[51]:


train4.plot.scatter('age','day')


# In[52]:


train4.head()


# In[53]:


train4=train4.iloc[:,:].values


# In[54]:


train4


# In[55]:


from sklearn.preprocessing import LabelEncoder


# In[56]:


country=LabelEncoder()


# In[57]:


train4[:,4]=country.fit_transform(train4[:,4])


# In[58]:


train4


# In[59]:


from sklearn.compose import ColumnTransformer


# In[60]:


from sklearn.preprocessing import OneHotEncoder


# In[61]:


test['marital']


# In[62]:


ct=ColumnTransformer(transformers=[('encode',OneHotEncoder(),[4])],remainder='passthrough')


# In[63]:


trainx=ct.fit_transform(train4)


# In[64]:


trainx


# In[65]:


trainx=pd.DataFrame(trainx)


# In[66]:


trainx


# In[67]:


trainx[14]


# In[68]:


trainx.head(80)


# In[69]:


ct=ColumnTransformer(transformers=[('encode',OneHotEncoder(),[14])],remainder='passthrough')


# In[70]:


trainy=ct.fit_transform(trainx)


# In[71]:


trainy=pd.DataFrame(trainy)


# In[72]:


trainy


# In[73]:


trainy.head(20)


# In[74]:


trainx.loc[:,0:17]


# In[75]:


trainx.loc[450:490,]


# In[76]:


trainy.loc[:,17:25]


# In[77]:


ct=ColumnTransformer(transformers=[('encode',OneHotEncoder(),[18])],remainder='passthrough')


# In[78]:


trainz=ct.fit_transform(trainy)


# In[79]:


trainz=pd.DataFrame(trainz)


# In[80]:


trainz


# In[81]:


trainz.loc[:,25:40]


# In[82]:


ct=ColumnTransformer(transformers=[('encode',OneHotEncoder(),[30])],remainder='passthrough')


# In[83]:


traina=ct.fit_transform(trainz)


# In[84]:


traina=pd.DataFrame(traina)


# In[85]:


traina


# In[86]:


ct=ColumnTransformer(transformers=[('encode',OneHotEncoder(),[33])],remainder='passthrough')


# In[87]:


trainb=ct.fit_transform(traina)


# In[88]:


trainb=pd.DataFrame(trainb)


# In[89]:


trainb


# In[90]:


ct=ColumnTransformer(transformers=[('encode',OneHotEncoder(),[36])],remainder='passthrough')


# In[91]:


trainc=ct.fit_transform(trainb)


# In[92]:


trainc


# In[93]:


trainc=pd.DataFrame(trainc)


# In[94]:


trainc


# In[95]:


ct=ColumnTransformer(transformers=[('encode',OneHotEncoder(),[38])],remainder='passthrough')


# In[96]:


traind=ct.fit_transform(trainc)


# In[97]:


traind


# In[98]:


traind=pd.DataFrame(traind)


# In[99]:


traind


# In[100]:


ct=ColumnTransformer(transformers=[('encode',OneHotEncoder(),[40])],remainder='passthrough')


# In[101]:


traine=ct.fit_transform(traind)


# In[102]:


traine=pd.DataFrame(traine)


# In[103]:


traine


# In[104]:


ct=ColumnTransformer(transformers=[('encode',OneHotEncoder(),[48])],remainder='passthrough')


# In[105]:


traine1=ct.fit_transform(traine)


# In[106]:


traine1=pd.DataFrame(traine1)


# In[107]:


traine1


# In[108]:


ct=ColumnTransformer(transformers=[('encode',OneHotEncoder(),[52])],remainder='passthrough')


# In[109]:


trainf=ct.fit_transform(traine1)


# In[110]:


trainf=pd.DataFrame(trainf)


# In[111]:


trainf


# In[112]:


trainf.loc[0:35,0:10]


# In[113]:


trainf.loc[0:35,0:30]


# In[114]:


traine.loc[0:30,45:49]


# In[115]:


trainf


# In[116]:


x=trainf.loc[:,1:53]


# In[117]:


x


# In[118]:


y=trainf.loc[:,0]


# In[119]:


y


# In[120]:


from sklearn.model_selection import train_test_split


# In[121]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size =0.2,random_state=5)


# In[122]:


x_test


# In[123]:


y


# In[124]:


from sklearn.ensemble import RandomForestClassifier


# In[125]:


rf=RandomForestClassifier()


# In[126]:


labels=trainf.loc[:,0]


# In[127]:


x_train,x_test,y_train,y_test=train_test_split(trainf,labels,test_size =0.2)


# In[128]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:





# In[ ]:





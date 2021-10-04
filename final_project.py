#!/usr/bin/env python
# coding: utf-8

# In[5]:


raw_data=pd.read_csv('C:\\Users\\Dell\\Downloads\\Python-Tutorials-master\\Python-Tutorials-master\\Decision Tree Tutorials\\churn raw data.csv',encoding='latin=1')


# In[6]:


import pandas as pd


# In[7]:


raw_data


# In[8]:


import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
np.set_printoptions(formatter={'float_kind':'{:f}'.format})
sns.set(rc={'figure.figsize':(8,6)})
from pandas import to_datetime
import itertools
import warnings
import datetime
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,r2_score


# In[9]:


for column in raw_data:
    unique_vals=np.unique(raw_data[column])
    nr_values= len(unique_vals)
    if nr_values<36:
        print('The number of values for feature {} :{} -- {}'.format(column,nr_values,unique_vals))
    else:
        print('The number of values for feature {} :{}'.format(column,nr_values))


# In[10]:


raw_data2=raw_data[['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited']]
g=sns.pairplot(raw_data2,hue='Exited',diag_kws={'bw':0.2})



# In[11]:


features = ['Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember']


for f in features:
    plt.figure()
    ax=sns.countplot(x=f,data=raw_data,hue='Exited',palette='Set1')


# In[12]:


new_raw_data=pd.get_dummies(raw_data2,columns=['Geography','Gender','HasCrCard','IsActiveMember'])
new_raw_data.head()


# In[13]:


scale_vars=['CreditScore','EstimatedSalary','Balance','Age']
scaler=MinMaxScaler()
new_raw_data[scale_vars]=scaler.fit_transform(new_raw_data[scale_vars])
new_raw_data.head()


# In[14]:


X=new_raw_data.drop('Exited',axis=1).values
y=new_raw_data['Exited'].values
print('X shape {}'.format(np.shape(X)))
print('y shape {}'.format(np.shape(y)))
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.9,test_size =0.1,random_state=0)


# In[15]:


dt=DecisionTreeClassifier(criterion='entropy',max_depth=2,random_state=1)
dt.fit(X_train,y_train)


# In[16]:


pip install graphviz


# In[17]:


import graphviz


# In[18]:


dot_data = tree.export_graphviz(dt,out_file=None,
    feature_names=new_raw_data.drop('Exited',axis=1).columns,
    class_names=new_raw_data['Exited'].unique().astype(str),
    filled=True,rounded=True,
    special_characters=True)
graph=graphviz.Source(dot_data)
graph


# In[19]:


conda install python_graphviz


# In[20]:


for i,column in enumerate(new_raw_data.drop('Exited',axis=1)):
    print ('important of feature{}:,{:.3f}'.format(column,dt.feature_importances_[i]))
    
    fi=pd.DataFrame({'Variable':[column],'Feature Importance Score':[dt.feature_importances_[i]]})
    
    try:
        final_fi =pd.concat([final_fi,fi],ignore_index=True)
    except:
        final_fi =fi
            
final_fi=final_fi.sort_values('Feature Importance Score',ascending=False).reset_index()
final_fi


# In[21]:


print("Training Accuracy is:",dt.score(X_train,y_train))

print("Testing Accuracy is:",dt.score(X_test,y_test))


# In[22]:


def plot_confusion_matrix(cm,classes=None,title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not  None:
        sns.heatmap(cm,xticklabels=classes,vmin=0,vmax=1,annot=True,annot_kws={'size':50})
    else:
        sns.heatmap(cm,vmin=0,vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[23]:


y_pred=dt.predict(X_train)

cm=confusion_matrix(y_train,y_pred)
cm_norm=cm/cm.sum(axis=1)[:,np.newaxis]
plt.figure()
plot_confusion_matrix(cm_norm,classes=dt.classes_,title='Training confusion')


# In[24]:


y_pred=dt.predict(X_train)


# In[25]:


y_pred


# In[26]:


from itertools import product
n_estimators= 100
max_features =[1,'sqrt','log2']
max_depths =[None,2,3,4,5]
for f,d in product(max_features,max_depths):
    rf=RandomForestClassifier(n_estimators=n_estimators,
                              criterion='entropy',
                              max_features=f,
                              max_depth=d,
                              n_jobs=2,
                              random_state=1337)
    rf.fit(X_train,y_train)
    prediction_test =rf.predict(X=X_test)
    print('Classification accuracy on test set with max features ={} and max_depth = {}: {:3f}'.format(f,d,accuracy_score(y_test,prediction_test)))
    cm=confusion_matrix(y_test,prediction_test)
    cm_norm=cm/cm.sum(axis=1)[:,np.newaxis]
    plt.figure()
    plot_confusion_matrix(cm_norm,classes=rf.classes_,
    title='Confusion matrix accuracy on test set with max features= {} and max_depth ={}: {:3f}'.format(f,d,accuracy_score(y_test,prediction_test)))
    


# In[88]:


unseen_data=pd.read_csv('C:\\Users\\Dell\Downloads\\Python-Tutorials-master\\Python-Tutorials-master\\Decision Tree Tutorials\\new unseen data.csv',encoding='latin=1')


# In[89]:


unseen_data


# In[90]:


unseen_data2=unseen_data[['CreditScore','Geography',
       'Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard',
       'IsActiveMember','EstimatedSalary']]


# In[91]:


unseen_data2=pd.get_dummies(unseen_data2,columns=['Geography','Gender','HasCrCard','IsActiveMember'])


# In[92]:


scale_vars=['CreditScore','EstimatedSalary','Balance','Age']
unseen_data2[scale_vars]=scaler.fit_transform(unseen_data2[scale_vars])


# In[94]:


unseen_data2.head()


# In[95]:


pred_rf=rf.predict(unseen_data2.values)


# In[96]:


pred_prob_rf=rf.predict_proba(unseen_data2.values)


# In[97]:


pred_rf


# In[98]:


pred_prob_rf


# In[ ]:





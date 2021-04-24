
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv(open('E:/ML/decision_tree/car_evaluation.csv','rb'))


# In[9]:


df.dtypes


# In[10]:


df.small.value_counts()


# In[12]:


df['2'].value_counts()


# In[13]:


df.isnull().sum()


# In[14]:


df.info()


# In[16]:


df.describe()


# In[60]:


col=df.columns
for i in col:
    print(df[i].value_counts())
col=list(col)
col.pop()

col


# In[61]:


x=df.drop(['unacc'],axis=1)
y=df['unacc']
xdf=[]
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
xdf=x.apply(LabelEncoder().fit_transform)
xd=pd.get_dummies(xdf,columns=col)
xd


# In[25]:


from sklearn.model_selection import train_test_split


# In[63]:


xtrain,xtest,ytrain,ytest=train_test_split(xd,y,test_size=0.2)
xtrain


# In[64]:


from sklearn.tree import DecisionTreeClassifier as dtc
model=dtc(criterion='gini')
model.fit(xtrain,ytrain)


# In[65]:


ypred=model.predict(xtest)


# In[67]:


from sklearn.metrics import accuracy_score
accuracy_score(ytest,ypred)


# In[68]:


xpred=model.predict(xtrain)
accuracy_score(ytrain,xpred)


# In[70]:


model.score(xtrain,ytrain)


# In[71]:


model.score(xtest,ytest)


# In[72]:


from sklearn import tree
tree.plot_tree(model.fit(xtrain,ytrain))


# In[79]:


model1=dtc(criterion='entropy',max_depth=7)
model1.fit(xtrain,ytrain)
ypred1=model1.predict(xtest)


# In[80]:


accuracy_score(ytest,ypred1)


# In[81]:


xpred1=model.predict(xtrain)


# In[82]:


accuracy_score(ytrain,xpred1)


# In[91]:


import graphviz
data=tree.export_graphviz(model1)
graph=graphviz.Source(data)
graph


# In[93]:


from sklearn.metrics import confusion_matrix, classification_report
cm=confusion_matrix(ytest,ypred)
cm


# In[94]:


cr=classification_report(ytest,ypred)
cr


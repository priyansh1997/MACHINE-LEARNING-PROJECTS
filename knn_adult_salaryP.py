# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 09:58:08 2020

@author: PRIYANSH KNN
"""
import numpy as np
import pandas as pd

file=pd.read_excel(open('adult_salary_dataset.xlsx','rb'))
file=file.dropna()

x=file.iloc[:,[1,3,5,6,7,8,9]].values
y=file.iloc[:,13].values


from sklearn.preprocessing import OneHotEncoder as ohe, LabelEncoder as le

y=le().fit_transform(y)

y=y.reshape(-1,1)

y=ohe().fit_transform(y).toarray()

x=pd.DataFrame(x)

#column can be renamed as there headings change when converting to DataFrame
#x=x.rename(columns={0:'abc'})

#uniwque() function find out the different values in a column
#x.abc.unique()

x=x.apply(le().fit_transform)

x=ohe().fit_transform(x).toarray()

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier as knn

KNN=knn(n_neighbors=3)

KNN.fit(xtrain,ytrain)

print(KNN.score(xtest,ytest))




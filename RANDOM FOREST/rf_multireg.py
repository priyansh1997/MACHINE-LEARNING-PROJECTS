
import pandas as pd 
file = pd.read_csv(open("multireg.csv",'rb'))

x=file.iloc[:,:-1]
y=file.iloc[:,-1]

file.shape
file.head()

file.describe()

import numpy as np
x=np.array(x)
y=np.array(y)

from sklearn.preprocessing import StandardScaler
x=StandardScaler().fit_transform(x)

from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest=train_test_split(x,y,test_size=0.2)

from sklearn.ensemble import RandomForestRegressor
clf=RandomForestRegressor(n_estimators=1000, random_state=7)
clf.fit(xtrain,ytrain)
ypred=clf.predict(xtest)

import matplotlib.pyplot as plt
x_coordinates=np.linspace(1,len(ytest),len(ytest))
plt.plot(x_coordinates,ypred,color='blue')
plt.plot(x_coordinates,ytest,color='pink')






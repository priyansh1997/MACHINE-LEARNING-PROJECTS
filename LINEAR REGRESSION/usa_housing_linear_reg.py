
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(open("E:/ML/linear regression/USA_Housing.csv",'rb'))
df.head()
df.info()
df.describe()
df.columns

sns.pairplot(df)

sns.distplot(df['Price'],rug=True,hist=False)

sns.heatmap(df.corr(),annot=True,fmt='0.2f')

x=df.iloc[:,0:5]
y=df.iloc[:,5]

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)


#linear regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression(normalize=True)
lr.fit(xtrain,ytrain)

from sklearn import metrics
from sklearn.model_selection import cross_val_score

def cvl(model):
    pred=cross_val_score(model,x,y,cv=10)
    return pred.mean()

def evaluate(actual,pred):
    print('MAE: ', metrics.mean_absolute_error(actual,pred))
    print('MSE: ', metrics.mean_squared_error(actual,pred))
    print('RMSE: ', np.sqrt(metrics.mean_squared_error(actual,pred)))
    print('R2 square: ',metrics.r2_score(actual,pred))
    
ypred=lr.predict(xtest)
xpred=lr.predict(xtrain)
plt.scatter(ytest,ypred)

evaluate(ytest,ypred)
evaluate(ytrain,xpred)

#robust regression is used when we want to take outliers in our data
#and we know that they will not affect much of our pred
from sklearn.linear_model import RANSACRegressor
rr=RANSACRegressor(max_trials=100)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
rr.fit(xtrain,ytrain)

ypred=rr.predict(xtest)
xpred=rr.predict(xtrain)
evaluate(ytest,ypred)
evaluate(ytrain,xpred)



#ridge resgression is used to solve the issue of festures which are quite similar(multicollinearity)
#and also reduce overfitting
from sklearn.linear_model import Ridge
rd=Ridge(alpha=100,solver='auto',tol=0.0001)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
rd.fit(xtrain,ytrain)
ypred=rd.predict(xtest)
xpred=rd.predict(xtrain)
evaluate(ytest,ypred)
evaluate(ytrain,xpred)



#Lasso is used to select the best features for the model and removel multicollineaarity 
from sklearn.linear_model import Lasso
ls=Lasso(alpha=0.1,precompute=True,positive=True,selection='random')

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
ls.fit(xtrain,ytrain)
ypred=ls.predict(xtest)
xpred=ls.predict(xtrain)
evaluate(ytest,ypred)
evaluate(ytrain,xpred)

#it possess the properties of both lasso and ridge
from sklearn.linear_model import ElasticNet
en=ElasticNet(alpha=0.1,l1_ratio=0.9,selection='random')

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
en.fit(xtrain,ytrain)
ypred=en.predict(xtest)
xpred=en.predict(xtrain)
evaluate(ytest,ypred)
evaluate(ytrain,xpred)

#this fits a polynomial line rather than a straight line and also increases the fit  
from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
xtrain1=pr.fit_transform(xtrain)
xtest1=pr.fit_transform(xtest)

lr.fit(xtrain1,ytrain)
ypred=lr.predict(xtest1)
xpred=lr.predict(xtrain1)
evaluate(ytest,ypred)
evaluate(ytrain,xpred)

































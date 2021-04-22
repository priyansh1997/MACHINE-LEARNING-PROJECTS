import pandas as pd 
import numpy as np

file = pd.read_excel(open('adult_salary_dataset.xlsx','rb'))
# =============================================================================
# file = pd.read_excel(open('adult_salary_dataset.xlsx','rb',sep=',')) if comma separated data is given
# =============================================================================

file.shape
file.head()
file.columns

file.describe()
# =============================================================================
# .describe() will show count, mean, max/min values, std deviation, interquartiles
# =============================================================================

file1=file.dropna()
# =============================================================================
# .dropna() drops the rows which have missing values
# =============================================================================

y = file.iloc[:,14]
x = file.iloc[:,[1,3,5,6,7,8,9,13]]

from sklearn.preprocessing import LabelEncoder as le,  OneHotEncoder as ohe


x = x.apply(le().fit_transform)
#applying labelencoding to multiple features at once

x1=pd.get_dummies(x,columns=['Workclass'])
x1=pd.get_dummies(x1,columns=['education'])
x1=pd.get_dummies(x1,columns=['status'])
x1=pd.get_dummies(x1,columns=['occupation'])
x1=pd.get_dummies(x1,columns=['relationship'])
x1=pd.get_dummies(x1,columns=['race'])
x1=pd.get_dummies(x1,columns=['sex'])
x1=pd.get_dummies(x1,columns=['native country'])

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x1,y,test_size=0.2,random_state=4)

from sklearn.tree import DecisionTreeClassifier as clf
classifier=clf(criterion="gini",random_state=4)

fitt=classifier.fit(xtrain, ytrain)

ypred=classifier.predict(xtest)

from sklearn.metrics import accuracy_score
accr=accuracy_score(ytest,ypred)*100






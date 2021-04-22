

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
print(diabetes.DESCR)


# =============================================================================
# diabetes.head()
# diabetes.describe()
# these both will not work because dataset is not in dataframe format it is in utils.Bunch
# =============================================================================

#diabetes_x=diabetes.data[:,np.newaxis,2]
diabetes_x=diabetes.data
diabetes_x_train=diabetes_x[:-30]
diabetes_x_test=diabetes_x[-30:]

diabetes_y_train=diabetes.target[:-30]
diabetes_y_test=diabetes.target[-30:]

model=linear_model.LinearRegression()
model.fit(diabetes_x_train,diabetes_y_train)
diabetes_y_predict=model.predict(diabetes_x_test)

print("root mean squared error is: ",np.sqrt(mean_squared_error(diabetes_y_test,diabetes_y_predict)))

print("weights: ",model.coef_)

print("intercept: ",model.intercept_)

# =============================================================================
# plt.scatter(diabetes_x_test,diabetes_y_test)
# plt.plot(diabetes_x_test,diabetes_y_predict)
# 
# plt.show
#this will work for only for one feature because no of features will increase no of dimension and graph cant show more than 3 dimension
# =============================================================================

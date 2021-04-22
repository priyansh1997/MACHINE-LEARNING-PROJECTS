
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_csv(open('cell_samples.csv','rb'))

X = file.iloc[:,[1,2,3,5,6,7,8,9,10]]

X = pd.DataFrame(X)

X.head()

X.tail()

X.columns

X.shape

X.size

X.count()

X['Class'].value_counts()

X['BareNuc'].unique()

X['BareNuc'].value_counts()

X.dtypes

X['BareNuc'] = pd.Series(X['BareNuc'])

# =============================================================================
# Parameters
"""
argscalar, list, tuple, 1-d array, or Series
errors{‘ignore’, ‘raise’, ‘coerce’}, default ‘raise’
If ‘raise’, then invalid parsing will raise an exception.

If ‘coerce’, then invalid parsing will be set as NaN.

If ‘ignore’, then invalid parsing will return the input.

downcast{‘integer’, ‘signed’, ‘unsigned’, ‘float’}, default None
If not None, and if the data has been successfully cast to a numerical dtype (or if the data was numeric to begin with), downcast that resulting data to the smallest numerical dtype possible according to the following rules:

‘integer’ or ‘signed’: smallest signed int dtype (min.: np.int8)

‘unsigned’: smallest unsigned int dtype (min.: np.uint8)

‘float’: smallest float dtype (min.: np.float32)

As this behaviour is separate from the core conversion to numeric values, any errors raised during the downcasting will be surfaced regardless of the value of the ‘errors’ input.

In addition, downcasting will only occur if the size of the resulting data’s dtype is strictly larger than the dtype it is to be cast to, so if none of the dtypes checked satisfy that specification, no downcasting will be performed on the data.

Returns
retnumeric if parsing succeeded.
Return type depends on input. Series if Series, otherwise ndarray.
"""
# =============================================================================

X['BareNuc'] = pd.to_numeric(X['BareNuc'], errors='coerce', downcast='integer').notnull()

X['BareNuc'] = X['BareNuc'].astype('int')
# use either imputer or dorpna to fill the values in place of NaN or to remove the NaN tuple 

#X= X.dropna(subset = ['BareNuc'])

#for removeing all the rows with at least one value NaN

benign_df = X[X['Class']==2][0:200]

malignant_df = X[X['Class']==4][0:200]

# matplotlib.use('TKAgg',warn=False, force=True) #sometimes when we run other programs matpltlib starts using non gui lib so we have to use tkagg libs in it to get gui based outputs 

axes = benign_df.plot(kind = 'scatter', x = 'Clump', y ='UnifSize', color = 'red', label = 'Benign' )

#to get the graph we have to run both the lines together together 
malignant_df.plot(kind = 'scatter', x = 'Clump', y ='UnifSize', color = 'blue', label = 'Malignant', ax = axes )
plt.show()

X = X.dropna()

A_df = X.iloc[:,[0,1,2,3,4,5,6,7]]

y = X.iloc[:,8]

x = np.array(A_df)

y = np.array(y) 

#for finding the best features

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# now create a classifier from svm 

from sklearn import svm
classifier = svm.SVC(kernel = 'linear', gamma = 'auto', C = 2  )
#4 types of kernel Linear, Polynomial, rbf, sigmoid
#gamma calculates the coffecients of kernel
#C is the penalty we impose on incorrectly place data points here we are giving 2 units of penalty on every incorrectly placed data points

classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)

from sklearn.metrics import classification_report

classification_report(y_test, y_predict)

#precision: out of all the predicted cancer how many are the right predictions 
#recall: out of all cancer patient how many are predicted by the system
#f1-score: harmonical mean of precision vs recall
#support: how many instances of are positive cancer patient and how many are negative or of class 2 which is benign
"""
#from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

classifier = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)


classifier = classifier.fit(X_train, y_train)  

#using print to get the details like Text(170.9,196.385,'X[1] <= 2.5\nentropy = 0.927\

print(tree.plot_tree(classifier.fit(X_train, y_train)))

"""
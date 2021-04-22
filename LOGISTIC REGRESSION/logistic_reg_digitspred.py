
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn import metrics

file = load_digits()
file.data.shape
file.target.shape
file.DESCR

plt.figure(figsize=(20,4))

for index, (image, label) in enumerate(zip(file.data[0:5],file.target[0:5])):
    plt.subplot(1, 5, 1)
    plt.imshow(np.reshape(image, (8,8)),cmap=plt.cm.magma)
    plt.title("Training: %i\n" %label,fontsize=20)
    plt.show()

xtrain,xtest,ytrain,ytest=train_test_split(file.data,file.target,test_size=0.2)

from sklearn.linear_model import LogisticRegression as lr
lr1=lr(max_iter=500)#always make an instance and then continue otherwise sometimes it will throw an error
lr1.fit(xtrain,ytrain)

print(lr1.predict(xtest[0:10]))

xtest=np.around(xtest,2)
ytest=np.around(ytest,2)
score=lr1.score(xtest,ytest)
print(score)

ypred=lr1.predict(xtest)
cm=metrics.confusion_matrix(ytest,ypred)

plt.figure(figsize=(8,8))
sns.heatmap(cm, annot=True, fmt='.2f', linewidths=0.5, square=True, cmap='copper')



index=0
clfindex=[]
for predict,actual in zip(ypred,ytest):
    if (predict==actual):
        clfindex.append(index)
    index+=1
    
plt.figure(figsize=(20,4))  
for pltindex, wrong in enumerate(clfindex[0:5]):
    plt.subplot(1, 4, 1)
    plt.imshow(np.reshape(xtest[wrong], (8,8)),cmap=plt.cm.magma)
    plt.title("Predicted: {}, Actual: {}".format(ypred[wrong],ytest[wrong]),fontsize=20)
    plt.show()
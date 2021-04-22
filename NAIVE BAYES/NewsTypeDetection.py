
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

file=fetch_20newsgroups()
file.DESCR
file.data
file.target
file.target.shape
file.target_names

types=['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

train=fetch_20newsgroups(subset='train',categories=types)
test=fetch_20newsgroups(subset='test',categories=types)

train.data[5:8]

from sklearn.feature_extraction.text import TfidfVectorizer
# =============================================================================
# Convert a collection of raw documents to a matrix of TF-IDF features.
# The sklearn.feature_extraction module can be used to extract features in a format 
# supported by machine learning algorithms from datasets consisting of formats such as text and image.
# =============================================================================

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


model=make_pipeline(TfidfVectorizer(),MultinomialNB())
model.fit(train.data,train.target)

output=model.predict(test.data)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix (test.target, output)
sns.heatmap(cm.T,cmap='copper',square=True, annot=True, fmt='d', cbar=False, xticklabels=train.target_names, yticklabels=train.target_names)

plt.xlabel('true label')
plt.ylabel('predicted label')

def predict_types(s,train=train, model=model):
    pred=model.predict([s])
    return train.target_names[pred[0]]

predict_types('electric cars')




























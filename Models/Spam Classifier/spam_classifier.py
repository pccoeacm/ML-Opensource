# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 16:26:13 2020

@author: Varun Pusarla
"""

import pandas as pd
import pickle
messages=pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t',names=["label","message"])


import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]

for i in range(len(messages)):
    review= re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review=review.lower()
    review = review.split()
    
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review= ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()


y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)

filename= 'spamclassifier.sav'
pickle.dump(spam_detect_model, open(filename,'wb'))
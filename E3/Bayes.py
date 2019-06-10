import numpy as np
import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import glob, os

dfs = []
for file_num in range(0, 26):
    url = "data"+str(file_num)+"_char.csv"
    t = pd.read_csv(url, index_col=None, header=0)
    dfs.append(t)
df  = pd.concat(dfs, axis=0, ignore_index=True)
df=shuffle(df)
df.count()

le=LabelEncoder()
le.fit(df['category'].drop_duplicates(keep='first'))

y_encoder=le.transform(y)

vectorizer = CountVectorizer(analyzer='char')
X_encoder = vectorizer.fit_transform(X)

test_ratio=0.3
test_size=int (test_ratio*df['category'].count())
X_train, X_test, y_train, y_test = X_encoder.toarray()[test_size:], X_encoder.toarray()[:test_size], y_encoder[test_size:], y_encoder[:test_size] 

clf = GaussianNB()
clf.fit(X_train, y_train)

y_predict=clf.predict(X_test)
accuracy_score(y_test, y_predict)
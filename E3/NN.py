import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.metrics import classification_report, confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, Conv2D, Flatten, MaxPooling2D   
from keras.optimizers import RMSprop
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
from matplotlib.pyplot import imshow
import glob
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

number_of_files = 26
dfs = []

for file_num in range(0,number_of_files):
    url = "/Users/b4435242/Desktop/作業/Machine_Learning/nctu_ML-master/0616018/data"+str(file_num)+"_char.csv"
    t = pd.read_csv(url, index_col=None, header=0)
    dfs.append(t)

df  = pd.concat(dfs, axis=0, ignore_index=True)
df=shuffle(df)

df.count()

y = df.category
X = df.words
words_processed = [ ]
for i in df.words:
    words_processed.append(i)
X = [str(i) for i in words_processed]

le=LabelEncoder()
le.fit(df['category'].drop_duplicates(keep='first'))

y_encoder=[]
y_encoder=le.transform(y)

X_encoder=[]
vectorizer = CountVectorizer(analyzer='char')
X_encoder = vectorizer.fit_transform(X)

test_ratio=0.3
test_size=int (test_ratio*df['category'].count())
X_train, X_test, y_train, y_test = X_encoder.toarray()[test_size:], X_encoder.toarray()[:test_size], y_encoder[test_size:], y_encoder[:test_size] 

y_train_onehot=keras.utils.to_categorical(y_train)
y_test_onehot=keras.utils.to_categorical(y_test)

print(X_train.shape)
print(X_test.shape)

from keras.layers import Dense, Activation
model = Sequential([
    Dense(500, input_shape=(8240,)),
    Activation('relu'),
     Dense(300),
    Activation('relu'),
     Dense(300),
    Activation('relu'),
    Dense(125),
    Activation('softmax'),
])

model.summary()

#Compile part
'''Your code here'''
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy']) 
#fit part
'''Your code here'''
y_train_history = model.fit(x=X_train, y=y_train_onehot, validation_split=0.2, epochs=10, batch_size=428, verbose=1)  

scores = model.evaluate(X_test, y_test_onehot)
print('acc:',scores[1])
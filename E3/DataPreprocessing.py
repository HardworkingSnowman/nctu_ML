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
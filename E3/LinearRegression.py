import csv
import pandas as pd
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import Pipeline

train_df = pd.read_csv('toy_jieba.csv')
test_df = pd.read_csv('toyword_pcs.csv')
train_df.head()

corpus1 = pd.concat([train_df.words, train_df.category]).sample(frac=1)
print(corpus1.head())
corpus2 = pd.concat([test_df.words, test_df.category]).sample(frac=1)
print(corpus2.head())

model1 = Word2Vec(corpus1)
model2 = Word2Vec(corpus2)

def most_similar(w2v_model, words, topn=10):
    similar_df = pd.DataFrame()
    for word in words:
        try:
            similar_words = pd.DataFrame(w2v_model.wv.most_similar(word, topn=topn), columns=[word, 'cos'])
            similar_df = pd.concat([similar_df, similar_words], axis=1)
        except:
            print(word, "not found in Word2Vec model!")
    return similar_df

pipeline = Pipeline([
    ('poly', PolynomialFeatures(8)),
    ('clf', LinearRegression())
])

pipeline.fit(model1, model2)

print('Prediction score:', pipeline.score(model1, model2))
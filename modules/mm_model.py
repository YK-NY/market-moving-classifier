import pandas as pd
import numpy as np
import xlrd

#import jsonlines
from pandas.io.json import json_normalize
import re

import os
import glob

import matplotlib.pyplot as plt

import spacy
import gensim

from wordcloud import WordCloud
import textwrap


from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS

from sklearn.cluster import KMeans

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV, cross_val_score
# import dill -- unable to load in heroku/flask

import modules.forex_data as fd

# Function definitions

nlp = spacy.load("en")
stopwords = spacy.lang.en.stop_words.STOP_WORDS
#stopwords.union(['-pron-'])

####################################

# print tweets from each cluster
# df = dataframe with tweets and cluster number (df_kmc)
def typical_tweets(df,num=5):
    tweets = []
    for i in range(num):
        wrap_list = textwrap.wrap(df.iloc[i,8], 76)  # 8 indexes the column with tweet text
        tweets.append('\n'.join(wrap_list))    
    return tweets
    # return the tweets

# Cluster tweets - using KMeans# 
# df - original tweet df (full or training set)
# here, load a saved model

def get_kmeans_model(df,n_clusters=5):
    
    kmc = dill.load(open('kmc_5.pkd','rb'))
    # Assign cluster number to each text
    df_kmc = pd.concat([df, pd.DataFrame(kmc.labels_, index = df.index).rename(columns={0:'Cluster'}) ], axis=1)
    
    #to print tweets from a cluster
    #for i in range(n_clusters):    
    #    print("\n From Cluster:",i,"\n")
    #    typical_tweets(df_kmc[df_kmc.Cluster == i],5)
    return df_kmc

# takes as input indices, the the df with tweets data & cluster info
# indices is a list
def get_clustered_tweets(indices, df):        
    tweets = []
    df_impact = df[df.index.isin(indices)]        
    clusters = df_impact.Cluster.unique().tolist()
    for c in clusters:
        tweets.append(typical_tweets(df_impact[df_impact.Cluster==c],3) ) # display 3 tweets from each cluster
    return tweets
    
# split into train and test data, primarily for clustering
def get_train_test(df):
    train_mask = (df['tweets_date'] < '2018-10-01')
    df_train = df.loc[train_mask]
    test_mask = (df['tweets_date'] > '2018-10-01')
    df_test = df.loc[test_mask]

    return df_train, df_test

    

####################################################

# process tweet data
def process_words(texts, stop_words=set(), allowed_pos=['NOUN', 'PROPN', 'ADJ', 'VERB', 'ADV']):
    result = []
    for t in texts:
        t = re.sub('\'', '', t)  #  replace single quotation marks (mainly to capture contractions)
        t = gensim.utils.simple_preprocess(t, deacc=True)
        #print(t)
        doc = nlp(' '.join(t))
        #print(doc)
        temp = [token.lemma_ for token in doc if token.pos_ in allowed_pos and token.lemma_ not in stop_words]
        #print(' '.join(temp))
        result.append(' '.join(temp))
    return result

    

# split into train and test set -- for the classification model
def split_test_train(df):
    train_mask = (df['tweets_date'] < '2018-10-01')
    df_train = df.loc[train_mask]
    X_train = df_train.drop('target',axis=1)
    y_train = df_train['target']
    
    test_mask = (df['tweets_date'] > '2018-10-01')
    df_test = df.loc[test_mask]
    X_test = df_test.drop('target',axis=1)
    y_test = df_test['target']
    
    return X_train, X_test, y_train, y_test
    
# tfidf - vectorize tweets
def vectorize_text(text):
    processed = process_words(text, stopwords.union(['-PRON-']))
    tfv = TfidfVectorizer(ngram_range=(1,1),
                      #tokenizer=tokenize_lemma,
                      max_features=8000,
                      sublinear_tf=True,
                      use_idf=True) 
    
    X = tfv.fit_transform(processed)
    
    return X

    
# Log Reg model to classify tweets
def classification_model(df):
    # split dataset into x,y
    X_train, X_test, y_train, y_test = split_test_train(df)

    #vectorize training and test text
    X_train_tfv = vectorize_text(X_train.text.values)
    X_test_tfv = vectorize_text(X_test.text.values)
    
    #y_train.reset_index(drop=True,inplace=True)
    #y_test.reset_index(drop=True,inplace=True)
    
    # logreg for classification
    w = {0:1, 1:99}
    clf = LogisticRegression(solver='liblinear',random_state=42, class_weight=w)
    clf.fit(X_train_tfv, y_train)
    y_pred = clf.predict(X_test_tfv)  
    
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.columns = ['target']
    
    y_pred_df.reset_index(drop=True,inplace=True)
    X_test.reset_index(drop=True,inplace=True)
    
    df_predicted = X_test.join(y_pred_df)
    #print(df_predicted.head())
    
    #print(f'Accuracy Score: {accuracy_score(y_test,y_pred)}')
    print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
    print(f'Area Under Curve: {roc_auc_score(y_test, y_pred)}')    
    
    return df_predicted 


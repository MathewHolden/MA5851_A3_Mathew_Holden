#This is the deployable code. Note must be able to find the text_clean function
#Load packages
import pandas as pd
import csv
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from time import time, sleep
import datetime
import math
import re
import nltk
from nltk.corpus import wordnet
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

#Load in saved model, vectorizer
rfclf_Filename = "Pickle_rfclf_Model.pkl"  
tfidf_Filename = "Pickle_tfidf_rf.pkl"

loaded_rfclf = pickle.load(open(rfclf_Filename, 'rb'))
loaded_tfidf = pickle.load(open(tfidf_Filename, 'rb'))

#Function to clean text of review comments ***Note still has numbers in it***
def text_clean(raw_corpus):
    corpus = []
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    for i in tqdm(raw_corpus.index):
        tokens = tokenizer.tokenize(raw_corpus[i])
        tokens = nltk.Text(tokens)
        tagged_text = nltk.pos_tag(tokens)
        stop_words = nltk.corpus.stopwords.words("english")
        text_out = [(w,t) for (w,t) in tagged_text if not w in set(stop_words)]
        text_out = [lemmatizer.lemmatize(w) for (w,t) in text_out]
        text_out = ' '.join(text_out)
        text_out = text_out.lower()
        corpus.append(text_out)
    return corpus

def best_picture_model_deploy(reviews):
    """Input a dataframe of movie reviews with column = "Comments" containing reviews and column = "Title" containing
    movie titles
    Will return a dataframe with column "Title" with movie titles and column = "Percent_Winner" containing the
    percentage of reviews for each movie classified as a best picture winner. Highest percentage is the predicted
    best picture winner"""
    results = pd.DataFrame(columns = ['Title','Percent_Winner'])
    reviews["clean_text"] = text_clean(reviews["Comments"])
    X = loaded_tfidf.transform(reviews["clean_text"])
    y_pred = loaded_rfclf.predict(X)
    movie_list = set(reviews["Title"])
    reviews["Predicted_Winner"] = y_pred
    for movie in movie_list:
        count = [reviews.loc[i]["Predicted_Winner"] for i in reviews.index if reviews.loc[i]["Title"] == movie]
        count_sum = sum(count)
        total = len(count) #use this to find percentage of winner reviews as percentage of reviews for that movie only
        percent_result = round(count_sum/total*100,2)
        results.loc[len(results)] = [movie, percent_result]
    return results
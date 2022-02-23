# This file creates the 'pipe' NLP model and saves it as model.joblib

# Import libraries
import pandas as pd
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import preprocessor

tfidf = TfidfVectorizer()
classifier = LinearSVC()

if __name__ == "__main__":
   #may need to change the following to your location of sentiments.csv
   df = pd.read_csv('/Users/z3433848/Desktop/DSI/Module 10/Lab/Lab 10/IOD_Lab-10_2/sentiments.csv')  
   pipe = make_pipeline(preprocessor(), tfidf, classifier)
   pipe.fit(df['text'],df['sentiment'])
   joblib.dump(pipe, open('model.joblib','wb'))
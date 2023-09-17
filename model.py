import numpy as np
import pandas as pd
import regex as re
import joblib
import en_core_web_sm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = en_core_web_sm.load()
classifier = LinearSVC()

def clean_text(text):
    # Reduce multiple spaces and newlines to only one
    text = re.sub(r'(\s\s+|\n\n+)', r'\1', text)
    # Remove double quotes
    text = re.sub(r'"', '', text)
    return text

def convert_text(text):
    sent = nlp(text)
    ents = {x.text: x for x in sent.ents}
    tokens = []
    for w in sent:
        if w.is_stop or w.is_punct:
            continue
        if w.text in ents:
            tokens.append(w.text)
        else:
            tokens.append(w.lemma_.lower())
    text = ' '.join(tokens)
    return text

def preprocessor(text, *args, **kwargs):
    # Your preprocessing logic here
    text = clean_text(text)
    text = convert_text(text)
    return text

class Preprocessor(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(preprocessor)

# Assuming you have uploaded 'sentiments.csv' to your Colab environment
# Modify the following line to specify the correct path if needed
df = pd.read_csv('https://raw.githubusercontent.com/venuannamdas/Data2023/master/sentiments.csv')

tfidf = TfidfVectorizer()
pipe = make_pipeline(Preprocessor(), tfidf, classifier)
pipe.fit(df['text'], df['sentiment'])

# Save the model as 'model.joblib'
joblib.dump(pipe, '/content/drive/My Drive/model.joblib')

from flask import Flask, request, render_template
from flask_ngrok import run_with_ngrok
import joblib
import numpy as np
import pandas as pd
import regex as re
import en_core_web_sm
from sklearn.base import BaseEstimator, TransformerMixin
import shutil  # Added for copying templates

app = Flask(__name__)
run_with_ngrok(app)

# Load the machine learning model from your Google Drive
model = joblib.load('/content/drive/My Drive/model.joblib')

# Define your preprocessing functions and classes here...
nlp = en_core_web_sm.load()

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

class Preprocessor(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(clean_text).apply(convert_text)

# Define the path to your templates folder in Google Drive
templates_folder = '/content/drive/My Drive/templates'


# Copy the templates from Google Drive to your local environment
shutil.copytree(templates_folder, 'templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']

    # Preprocess the input text using the preprocessor function
    preprocessed_input = preprocessor(pd.Series([input_text]))[0]

    # Use the loaded model to make predictions on the preprocessed input
    predicted_sentiment = model.predict([preprocessed_input])[0]

    if predicted_sentiment == 1:
        output = 'positive'
    else:
        output = 'negative'

    return render_template('results.html', sentiment=f'Predicted sentiment of "{input_text}" is {output}.')

if __name__ == "__main__":
    app.run()

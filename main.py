import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
import utils

app = Flask(__name__)

def stock_classifier(headlines):
    ## implement BAG OF WORDS
    countvector = CountVectorizer(ngram_range=(2, 2))
    traindataset = countvector.fit_transform(headlines)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict")
def predict():
    headlines = request.args.get('headlines')
    print(headlines)
    prediction = utils.stock_classifier(headlines)
    if prediction[0]==1:
        return render_template('predict.html',t='p')
    else:
        return render_template('predict.html',t='n')

if __name__ == '__main__':
    utils.load_saved_artifacts()
    app.run()

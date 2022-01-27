import pickle
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

__model = None

def stock_classifier(headline):
    ## implement BAG OF WORDS
    import pandas as pd
    df = pd.read_csv('Data.csv', encoding="ISO-8859-1")

    # print(df.head(5))
    train = df[df.Date < '20150101']
    test = df[df.Date > '20141231']

    # Removing punctuations
    data = train.iloc[:, 2:27]
    data.replace("[^a-zA-Z]", " ", inplace=True)

    # Renaming column names for ease of access
    list1 = [i for i in range(25)]
    new_index = [str(i) for i in list1]
    data.columns = new_index

    # Convertng headlines to lower case
    for i in new_index:
        data[i] = data[i].str.lower()

    initial_headlines = []
    for row in range(0, len(data.index)):
        initial_headlines.append(' '.join(str(i) for i in data.iloc[row, 2:25]))

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.ensemble import RandomForestClassifier

    ## implement BAG OF WORDS
    countvector = CountVectorizer(ngram_range=(2, 2))
    traindataset = countvector.fit_transform(initial_headlines)

    headlines = []
    headlines.append(headline)
    print(headlines)
    print('countvect')
    testdataset = countvector.transform(headlines)
    print(testdataset)
    return __model.predict(testdataset)

def load_saved_artifacts():
    print("loading saved artifacts...start")

    global __model
    if __model is None:
        with open('./stock_classifier.pickle', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")

if __name__ == '__main__':
    load_saved_artifacts()
    print(stock_classifier('Today The United Kingdom decides whether to remain in the European Union, or leave'))
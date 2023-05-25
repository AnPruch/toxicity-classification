import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

start_time = time.time()

data = pd.read_csv('labeled.csv')

data.loc[(data['toxic'] == 1.0), 'toxic'] = 'toxic'
data.loc[(data['toxic'] == 0.0), 'toxic'] = 'neutral'

texts = data['comment'].values
labels = data['toxic'].values
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.1,
    random_state=1,
)

NBdata = Pipeline(
    [
        ('N-grams', CountVectorizer(max_features=50_000,
                                    lowercase=False,
                                    ngram_range=(3, 4),
                                    analyzer='char')),
        ('classifier', MultinomialNB())
    ]
)
NBdata.fit(X_train, y_train)
pred_NBdata = NBdata.predict(X_test)
print(classification_report(y_test, pred_NBdata))
print("time elapsed: {:.2f}s".format(time.time() - start_time))

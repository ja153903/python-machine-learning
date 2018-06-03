# Import ---------------------------------------------------
import pandas as pd 
import numpy as np 
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
# ----------------------------------------------------------

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text

# a set with stopwords from the English language
stop = stopwords.words('english')

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield, text, label

def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

df = pd.read_csv('movie_data.csv', encoding='utf-8')

df['review'] = df['review'].apply(preprocessor)

# nltk.download('stopwords')

X_train = df.loc[:25, 'review'].values
y_train = df.loc[:25, 'sentiment'].values
X_test = df.loc[25:50, 'review'].values
y_test = df.loc[25:50, 'sentiment'].values


tfidf = TfidfVectorizer(
    strip_accents=None,
    lowercase=False,
    preprocessor=None
)

param_grid = [
    {'vect__ngram_range': [(1, 1)],
    'vect__stop_words': [stop, None],
    'vect__tokenizer': [tokenizer, tokenizer_porter],
    'clf__penalty': ['l1', 'l2'],
    'clf__C': [1.0, 10.0, 100.0]},
    {'vect__ngram_range': [(1, 1)],
    'vect__stop_words': [stop, None],
    'vect__tokenizer': [tokenizer, tokenizer_porter],
    'vect__use_idf': [False],
    'vect__norm': [None],
    'clf__penalty': ['l1', 'l2'],
    'clf__C': [1.0, 10.0, 100.0]}
]

lr_tfidf = Pipeline([
    ('vect', tfidf),
    ('clf', LogisticRegression(random_state=0))
])

# you wanna use all cores to speed up grid search
gs_lr_tfidf = GridSearchCV(
    estimator=lr_tfidf, 
    param_grid=param_grid,
    scoring='accuracy',
    cv=5, verbose=1,
    n_jobs=-1
)

gs_lr_tfidf.fit(X_train, y_train)

print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)

print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

clf = gs_lr_tfidf.best_estimator_

print('Test accuracy: %.3f' % clf.score(X_test, y_test))
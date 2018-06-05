from sklearn.feature_extraction.text import HashingVectorizer
import re 
import os 
import pickle 

cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'stopwords.pkl'), 'rb'))

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

vect = HashingVectorizer(
    decode_error='ignore',
    n_features=2**21,
    preprocessor=None,
    tokenizer=tokenizer
)

clf = pickle.load(open(os.path.join('pkl_objects', 'classifier.pkl'), 'rb'))

import numpy as np 

label = {
    0: 'negative',
    1: 'positive'
}

example = ['I love this movie']
X = vect.transform(example)

# print(clf.predict(X)[0])

# print(clf.predict(example))
# print(clf.predict_proba(example))

print("Prediction: {}\nProbability: {}".format(label[clf.predict(X)[0]], np.max(clf.predict_proba(X)) * 100))

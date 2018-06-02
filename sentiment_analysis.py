import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
# ----------------

df = pd.read_csv('movie_data.csv', encoding='utf-8')
count = CountVectorizer()
docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining, the weather is sweet, and one and one is two'
])

bag = count.fit_transform(docs)

# This gives us the index of the words in the feature vector
print(count.vocabulary_)

print(bag.toarray())

tfidf = TfidfTransformer(
    use_idf=True,
    norm='l2',
    smooth_idf=True
)

np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text

print(preprocessor(df.loc[0, 'review'][-50:]))

df['review'] = df['review'].apply(preprocessor)

def tokenizer(text):
    return text.split()

porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

nltk.downloads('stopwords')

# a set with stopwords from the English language
stop = stopwords.words('english')
'''
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]
'''




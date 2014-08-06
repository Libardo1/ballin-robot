from sklearn.feature_extraction.text import  CountVectorizer

corpus = [
    'This is the first document.',
    'This is the second second document',
    'And the third one',
    'Is this the first document'
]

vectorizer = CountVectorizer(min_df=1)

X = vectorizer.fit_transform(corpus)

print vectorizer.get_feature_names()
print X.toarray()

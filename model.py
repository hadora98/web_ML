import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

data = pd.read_excel (r'articles.xlsx')
df = pd.DataFrame(data)
pdf_train = df[df['Split']=='Train']
pdf_test  = df[df['Split']=='Dev']


text_clf1 = Pipeline([
    ('u1', FeatureUnion([
        ('word_features', Pipeline([
            ('ngramw', CountVectorizer(ngram_range=(1, 5), analyzer='word')),
            ('tfidf', TfidfTransformer()),
        ])),
        ('char_features', Pipeline([
            ('ngramc', CountVectorizer(ngram_range=(1, 5), analyzer='char')),
            ('tfidf', TfidfTransformer())
        ])),
    ])),
    ('clf',   svm.LinearSVC())]) # 0.65
model=text_clf1.fit(pdf_train['Title'], pdf_train['Source'])

pickle.dump(model, open('model.pkl','wb'))


model = pickle.load(open('model.pkl','rb'))
print(model.predict(["ترامب وإيفانكا يتعرضان لانتقادات بسبب دعم شركة مواد غذائية"]))

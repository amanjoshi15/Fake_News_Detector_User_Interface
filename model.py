import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

df=pd.read_csv('WELFake_Dataset.csv')
df=df.head(500)
df=df.dropna()

features=df[['title','text']]
label=df['label']

headlines=[]
for row in range(0,len(features.index)):
    headlines.append(' '.join(str(x) for x in features.iloc[row,0:2]))

st=PorterStemmer()
corpus=[]
for i in range(0,len(headlines)):
    r=re.sub('[^a-zA-Z]',' ',headlines[i])
    r=r.lower()
    words=word_tokenize(r)
    words=[st.stem(word) for word in words if word not in stopwords.words('english')]
    r=' '.join(words)
    corpus.append(r)
headlines=corpus

cv=CountVectorizer(ngram_range=(2,2))
features=cv.fit_transform(headlines).toarray()

X_train, X_test, Y_train, Y_test=train_test_split(features, label, test_size=0.2, random_state=1)

model=DecisionTreeClassifier()
model.fit(X_train,Y_train)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(cv, open("count_vectorizer.pkl", "wb"))
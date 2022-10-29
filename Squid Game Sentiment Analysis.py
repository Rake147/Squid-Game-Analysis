#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS,ImageColorGenerator


# In[2]:


data=pd.read_csv('C:/Users/Rakesh/Datasets/tweets_v8.csv')


# In[3]:


data.head()


# In[4]:


data.isnull().sum()


# In[5]:


data=data.drop(columns='user_description',axis=1)
dtaa=data.dropna()


# In[6]:


import nltk
import re
nltk.download('stopwords')
stemmer=nltk.SnowballStemmer('english')
from nltk.corpus import stopwords
import string
stopwords=set(stopwords.words('english'))


# In[10]:


def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopwords]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text


# In[11]:


data['text']=data['text'].apply(clean)


# In[13]:


text=' '.join(i for i in data.text)
stopwords=set(STOPWORDS)
wordcloud=WordCloud(stopwords=stopwords,background_color='white').generate(text)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[14]:


nltk.download('vader_lexicon')


# In[15]:


sentiments=SentimentIntensityAnalyzer()
data['Positive']=[sentiments.polarity_scores(i)['pos'] for i in data['text']]
data['Negative']=[sentiments.polarity_scores(i)['neg'] for i in data['text']]
data['Neutral']=[sentiments.polarity_scores(i)['neu'] for i in data['text']]


# In[16]:


data=data[['text','Positive','Negative','Neutral']]


# In[17]:


data.head()


# In[19]:


x=sum(data['Positive'])
y=sum(data['Negative'])
z=sum(data['Neutral'])


# In[20]:


def sentiment_score(a,b,c):
    if (a>b) and (a>c):
        print("Positive")
    elif (b>a) and (b>c):
        print('Negative')
    else: 
        print('Neutral')


# In[21]:


sentiment_score(x,y,z)


# In[22]:


print("Positive: ", x)
print("Negative: ", y)
print("Neutral: ", z)


# In[ ]:





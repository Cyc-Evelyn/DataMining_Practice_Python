"""

#Preprocess & split into train &test


---


1.   train & test each got 12500 negtive &12500 positive data
2.   preprocessing data
2.   concat & reindex & shuffle data
"""

import pandas as pd

data = pd.read_csv('IMDB_Dataset.csv')

"""##Preprocessing


---


1.   remove tags
2.   transfer category data into numeric (y)



"""

import re
import numpy as np

#del website tags 
def rm_tags(text): 
    re_tag = re.compile(r'<[^>]+>') 
    return re_tag.sub(' ',text) 

data['review'] = np.vectorize(rm_tags)(data['review'])

data.head()

posData = data[data['sentiment']=='positive']
negData = data[data['sentiment']=='negative']

train = pd.concat([posData[:12500],negData[:12500]],axis=0)
#reindex also shuffle the sequence
train = train.sample(frac=1).reset_index(drop=True)

test = pd.concat([posData[12500:],negData[12500:]],axis=0)
test = test.sample(frac=1).reset_index(drop=True)

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
token = Tokenizer(num_words=3800)
token.fit_on_texts(train['review'])
token.word_index

x_train_seq = token.texts_to_sequences(train['review'])
x_test_seq = token.texts_to_sequences(test['review'])

x_train = sequence.pad_sequences(x_train_seq,maxlen=380)
x_test = sequence.pad_sequences(x_test_seq,maxlen=380)

#transfrom category into numeric value
def cat2num(value):
    if value=='positive': 
        return 1
    else: 
        return 0
    
y_train = train['sentiment'].apply(cat2num)
y_test = test['sentiment'].apply(cat2num)

"""#Create RNN Model"""

from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import  Embedding
from keras.layers.recurrent import SimpleRNN

modelRNN = Sequential()
modelRNN.add(Embedding(output_dim=32,
           input_dim=3800,
           input_length=380))
modelRNN.add(Dropout(0.2))

modelRNN.add(SimpleRNN(units=16))
modelRNN.add(Dense(units=256,activation='relu'))
modelRNN.add(Dropout(0.35))
modelRNN.add(Dense(units=1,activation='sigmoid'))

modelRNN.summary()

modelRNN.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

train_history = modelRNN.fit(x_train,y_train,epochs=10,batch_size=100,verbose=2,validation_split=0.2)

scores = modelRNN.evaluate(x_test, y_test,verbose=1)
scores[1]

"""#LSTM"""

from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

modelLSTM = Sequential()
modelLSTM.add(Embedding(output_dim=32,
            input_dim=3800,
            input_length=380))
modelLSTM.add(Dropout(0.2))
modelLSTM.add(LSTM(32))
modelLSTM.add(Dense(units=2556,activation='relu'))
modelLSTM.add(Dropout(0.2))
modelLSTM.add(Dense(units=1,activation='sigmoid'))
modelLSTM.summary()

modelLSTM.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
train_history = modelLSTM.fit(x_train,y_train,epochs=10,batch_size=100,verbose=2,validation_split=0.2)

scores = modelLSTM.evaluate(x_test, y_test,verbose=1)
scores[1]

"""
#data preprocessing
---
1.  merge train & test as combi, remove the useless words
2.  remove stopwords (use nltk) + remove the word that has high frequency in both label(already take a look using wordcloud)
3. remove those words which its frequency is too low
3.  remove short word(less than  3)
4.  split into tokens =>  lowercase characters
"""

import pandas as pd
import numpy as np
train  = pd.read_csv('train.csv',delimiter='\t') 
test = pd.read_csv('test.csv',delimiter='\t')
sample = pd.read_csv('sample_submission.csv',delimiter=',')

test=test.merge(sample,how='inner',on='id')

combi =  pd.concat([train, test], axis=0, ignore_index=True)

combi = combi.drop(columns=['id'])

import re 
import warnings
#ignore the warning that reminds you using the latest version
warnings.filterwarnings("ignore", category=DeprecationWarning) 
def remove_words(input_txt,words):
    #re = regular expression
    r = re.findall(words,input_txt)
    for i in r:
        #re.sub(pattern, repl, string, count=0, flags=0) replace the match string into something
        input_txt = re.sub(i, '', input_txt)
    return input_txt
#remove website link
combi['clean_text'] = np.vectorize(remove_words)(combi['text'],'[a-zA-z]+://[^\s]*') 
combi.clean_text = combi.clean_text.str.replace("[^a-zA-Z#]", " ")
#remove punctuations
remove_list=['&','%','@','*','”',"‘","’",'“','”','$','~','(',')','!','.','?']
for x in remove_list:
    combi.clean_text = combi.clean_text.str.replace(x, " ")

combi

tokenized_text = combi.clean_text.apply(lambda x: x.split())
tokenized_text.head()

#turn into lowercase 
tokenized_text = tokenized_text.apply(lambda x: [str.lower(i) for i in x])

import nltk
from nltk.corpus import stopwords 
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
#DON'T DEL THE MOST FREQUECY WORDS , IT WOULD LOWER PRECISION(maybe there's some relation information)
# stopwords.extend(['said','show','time','think','work','because','this','people','film','will','year','first','new','two','love','even','reese','also','know','day'])

filtered_word_list = tokenized_text #make a copy of the word_list
for i in range(len(tokenized_text)):
    for word in tokenized_text[i]: # iterate over word_list
        if word in stopwords: 
            filtered_word_list[i].remove(word)# remove word from filtered_word_list if it is a stopword

filtered_word_list

for i in range(len(filtered_word_list)):
    filtered_word_list[i] = ' '.join(filtered_word_list[i])    
combi['clean_text'] = filtered_word_list
combi.head(10)

#remove short words
combi.clean_text = combi.clean_text.apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
combi.head(10)

"""#Use wordcloud find the most frequent word in fake/real news"""

import matplotlib.pyplot as plt 
all_words = ' '.join([text for text in combi['clean_text']]) 

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words) 
plt.figure(figsize=(25, 9)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis('off')
plt.show()
#most frequently words :

#not fake
normal_words =' '.join([text for text in combi['clean_text'][combi['label'] == 0]]) 

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(25, 9))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
#most frequently words : way  year first  new say two love star relationship going really

#fake
negative_words = ' '.join([text for text in combi['clean_text'][combi['label'] == 1]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(25, 9))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
#most frequently words : way may day year first know new say two love family new york back

"""#Find & clean the row includes string & reindex data
---
prevent the error that the label isn't numerical  
after that process,
1.   [:4987]are train_data
2.   [4987:]are test_data 
"""

np.unique(test['label'])
np.unique(train['label']) 
#train's label[1616] include string

combi[combi['label']=='label']

#remove that row includes string (it text=='content')
clean_data=combi[combi['label']!='label']

clean_data=clean_data.drop(columns=['text'])
# clean_data.shape #make sure we have same shape after we reindex

#there still a missing row[1616],so we need reindex
clean_data=clean_data.reset_index()
clean_data = clean_data.reset_index(drop=True) #remove old index

#clean_data[1615:] #got index[1616]
clean_data=clean_data.drop(columns=['index'])
clean_data.shape

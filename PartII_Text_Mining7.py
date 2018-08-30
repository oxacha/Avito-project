
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.metrics import confusion_matrix
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)
print('Hello')
#nrows to read first part of data
l=10000
df=pd.read_csv("train.csv",nrows=l)
print("Size of the initial dataset with NaN:", df.shape)
df=df.dropna().reset_index(drop=True)
print("Size of the initial dataset without NaN:", df.shape)
l=df.shape[0]
print("actual number of raws:",l)

###############################################################################
#creating another dataframe df_try for analysis
df_try=pd.DataFrame()

#discretization of the target probability
#seuil=np.median(df["deal_probability"])
print("seuil (to convert deal_propability to 0 or 1)=",seuil)
df_try["deal_prob"]=np.where(df["deal_probability"]>0,1,0)

#join the title and the main text of ad (adverisement)
df_try=df_try.join(df["title"])
df_try=df_try.join(df["description"])

#calculate the number of characters
df_try['char_count_title'] = df_try['title'].str.len()
df_try['char_count_main'] = df_try['description'].str.len()

#calculate the number of words
df_try['word_count'] = df_try['description'].apply(lambda x: len(x.split(" ")))

#calculate the average word length/ by=spaces
def avg_word(sentence):
  words = sentence.split(' ')#['word1','word2']
  return (sum(len(word) for word in words)/len(words))
df_try['avg_word'] = df_try['description'].apply(lambda x: avg_word(x))

#calculate the number of stopwords like "в, во, не, что, он, вот..."
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('russian')
#print(stop)

df_try['stopwords'] = df_try['description'].apply(lambda x: 
    len([x for x in x.split() if x in stop]))

#calculate the number of special characters
#instead Цве one can put @ and so on
df_try['hastags'] = df_try['description'].apply(lambda x: 
    len([x for x in x.split() if x.startswith('Цве')]))
    
#calculate the number of numerics
#15.5cm is not a digit! it is not good
df_try['numerics'] = df_try['description'].apply(lambda x: 
    len([x for x in x.split() if x.isdigit()]))
    
#calculate the number of Uppercase words
df_try['upper'] = df_try['description'].apply(lambda x: 
    len([x for x in x.split() if x.isupper()]))
    
###############################################################################
#preprocessing of the main text of ad
    
#removing lower case for the main text
df_try['description'] = df_try['description'].apply(lambda x: 
    " ".join(x.lower() for x in x.split()))#" " between words
#removing punctuation in the main text
df_try['description'] = df_try['description'].str.replace('[^\w\s]','')
#remove stop words in the main text
df_try['description'] = df_try['description'].apply(lambda x: 
    " ".join(x for x in x.split() if x not in stop))

#tokenization
#import textblob
#TextBlob(df_try['description'][1]).words

#stemming
##from nltk.stem import PorterStemmer
##st = PorterStemmer()
from nltk.stem.snowball import SnowballStemmer 
st = SnowballStemmer("russian")
df_try['description']=df_try['description'].apply(lambda x: 
    " ".join([st.stem(word) for word in x.split()]))
print(st.stem("перепрыгивающий"))
#results in перепрыгива
###############################################################################    
#lemmatization better tnan stemming
#BUT i did not find it for russian language!
#here only two examples for english
#df_try['description'] = df_try['description'].apply(lambda x: 
#    " ".join([Word(word).lemmatize() for word in x.split()]))

#one way to do it for englisg     
from textblob import Word
print(Word("dogs").lemmatize())
#results in dog

#another way to do it for english
import nltk
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
#from nltk.corpus import wordnet
wnl = WordNetLemmatizer()
print(wnl.lemmatize('dogs'))
#results in dog
###############################################################################    
#Instead of single words on can analyze duplets, triplets...
#It is not implemented in this code
import textblob
from textblob import TextBlob
#print(TextBlob(df_try['description'][3]).ngrams(2))   
print(TextBlob(df_try['description'][3])) 
print(TextBlob(df_try['description'][3]).ngrams(3))    
   

 

#Term frequency TF, for row times in row/length of row
#Inverse document frequency IDF, for document log(N/n)
#N-total number of rows, n-number of rows with word
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
 stop_words= 'english',ngram_range=(1,1))
train_vect = tfidf.fit_transform(df_try['description'])
print(train_vect)

#sentiment analysis
# we have to search this for russian language
#polarity and subjectivity
#[0] -1 negative sentiment 1 positive sentiment
#df_try['sentiments']=df_try['description'].apply(lambda x: 
#    TextBlob(x).sentiment[0])

###############################################################################
print('mashine learning for text mining')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn_pandas import DataFrameMapper



#the column with the main text df_try["description"] is the most important one
#this column needs to be transformed to TF-IDF object

#Term frequency TF, for row times in row/length of row
#Inverse document frequency IDF, for document log(N/n)
#N-total number of rows, n-number of rows with the given word

#Yhis is done in this way
#data_counts = CountVectorizer().fit_transform(df_try["description"])
#data_tfidf = TfidfTransformer(use_idf=True).fit_transform(data_counts)
#DataFrameMapper executes these two lines + it adds other column from df_try

#Now we need to add other columns to TF-IDF object
#this can be done in two ways
#1 using DataFrameMapper see below
#the problem: I do not know whether 
#(a)the usage of [CountVectorizer(),TfidfTransformer(use_idf=True)] is correct
#Do we need to perform normalization of data?

mapper = DataFrameMapper([
     ('description', [CountVectorizer(),TfidfTransformer(use_idf=True)]),
     ('char_count_title', None),
     ('stopwords', None),
     ('numerics', None),
     ('upper', None),

 ])   

#2 using pipeline with feature union
    
data=mapper.fit_transform(df_try)
target=df_try['deal_prob']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(data, target,  
                                                  test_size=0.2,random_state=1)

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
model=MultinomialNB()
#parameters = dict(alpha=np.arange(0.05,0.15,0.001))
parameters = dict(alpha=np.arange(0.001,0.5,0.01))
grid_model=GridSearchCV(estimator=model,param_grid=parameters)
grille = grid_model.fit(X_train, y_train)  
y_pred=grid_model.predict(X_test)

print("confusion matrix:")
cm=confusion_matrix(y_test,y_pred)
print(cm)
print("")
print("Metrics:")
print("Accuracy:", grid_model.score(X_test,y_test))
from sklearn.metrics import f1_score
print("F1 score:", f1_score(y_test,y_pred))
from sklearn.metrics import cohen_kappa_score
print("Cohen kappa score:", cohen_kappa_score(y_test,y_pred))
print("Best parameters: ",grid_model.best_params_)

   



























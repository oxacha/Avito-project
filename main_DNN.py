#The main difficulty is that we have to predict continious 
#variable in the interval [0,1]
#Requires: image features like whitness... (image_features_100000_0309.csv)
#generated by image_features_generator2.py
#Requires: vgg16 features (30000vgg_features.csv)
#generated by vgg16_image_features_generator.py
#Method: Dense Neural Network
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1500000)
#nrows to read first part of data
l=30000
df=pd.read_csv("train.csv",nrows=l)
print("Size of the initial dataset with NaN:", df.shape)
df=df.dropna().reset_index(drop=True)
l=df.shape[0]
print("Size of the initial dataset without NaN:", l)
print("INITIAL DATAFRAME:")
display(df.head(2))
###############################################################################
#Create df_try to run the alogorithm

#add target column
df_try=pd.DataFrame()

df_try["deal_prob"]=df["deal_probability"]

#df_try["test"]=df["deal_probability"]

#add price categories, see df["price"]
df_try=df_try.join(df["price"])

#add seq, see df['item_seq_number']
df_try=df_try.join(df["item_seq_number"])

#add user_type
df_user_type=pd.get_dummies(df["user_type"],prefix="user_type")
df_user_type=df_user_type.dropna().reset_index(drop=False)
df_user_type=df_user_type.drop("index",axis=1)
df_try=df_try.join(df_user_type)

#add type, see df["parent_category_name"]
df_parent_index = df.groupby('parent_category_name')['parent_category_name'].transform('count')
df_try=df_try.join(df_parent_index)

df_parent=pd.get_dummies(df["parent_category_name"],prefix="pcn")
df_parent=df_parent.dropna().reset_index(drop=False)
df_parent=df_parent.drop("index",axis=1)
df_try=df_try.join(df_parent)

#add type, see df["category_name"]
df_category_index = df.groupby('category_name')['category_name'].transform('count')
df_try=df_try.join(df_category_index)

df_category=pd.get_dummies(df["category_name"],prefix="cn")
df_category=df_category.dropna().reset_index(drop=False)
df_category=df_category.drop("index",axis=1)
df_try=df_try.join(df_category)

#add cities where annonce was published
df_cities_index = df.groupby('city')['city'].transform('count')
df_try=df_try.join(df_cities_index)

#df_cities=pd.get_dummies(df["city"],prefix="city")
#df_cities=df_cities.dropna().reset_index(drop=False)
#df_cities=df_cities.drop("index",axis=1)
#df_try=df_try.join(df_cities)

#add regions where annonce was published
df_region_index = df.groupby('region')['region'].transform('count')
df_try=df_try.join(df_region_index)



#add param123
df_param_1_index = df.groupby('param_1')['param_1'].transform('count')
df_try=df_try.join(df_param_1_index)

df_param_1=pd.get_dummies(df["param_1"],prefix="param_1")
df_param_1=df_param_1.dropna().reset_index(drop=False)
df_param_1=df_param_1.drop("index",axis=1)
df_try=df_try.join(df_param_1)

df_param_2_index = df.groupby('param_2')['param_2'].transform('count')
df_try=df_try.join(df_param_2_index)
df_param_3_index = df.groupby('param_3')['param_3'].transform('count')
df_try=df_try.join(df_param_3_index)

###############################################################################
###############################################################################
#TEXT FEATURES
#calculate the number of characters
df_try['char_count_title'] = df['title'].str.len()
df_try['char_count_main'] = df['description'].str.len()

#calculate the number of words
df_try['word_count'] = df['description'].apply(lambda x: len(x.split(" ")))

#calculate the number of stopwords like "в, во, не, что, он, вот..."
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('russian')
df_try['stopwords'] = df['description'].apply(lambda x: 
    len([x for x in x.split() if x in stop]))
    
#calculate the number of numerics
#15.5cm is not a digit! it is not good
#df_try['numerics'] = df['description'].apply(lambda x: 
#    len([x for x in x.split() if x.isdigit()]))
    
#calculate the number of periods
df_try['N_periods'] = df['description'].apply(lambda x: 
    x.count('.'))
###############################################################################
#IMPORT IMAGE CHARACTERISTICS
img_feat=pd.read_csv("image_features_100000_0309.csv",nrows=l)  
img_feat=img_feat.drop(["Unnamed: 0"],axis=1)
img_feat=img_feat.drop(["deal_prob_cont"],axis=1)#Control column
df_try=df_try.join(img_feat)
##
###############################################################################
#IMPORT vggg16 CHARACTERISTICS
vgg16_feat=pd.read_csv("30000vgg_features.csv",nrows=l) 
#display(img_feat["Unnamed: 0"].tail(10)) 
vgg16_feat=vgg16_feat.drop(["Unnamed: 0"],axis=1)
vgg16_feat=vgg16_feat.drop(["deal_prob_cont"],axis=1)#Control column
df_try=df_try.join(vgg16_feat)
##
###############################################################################
print('text mining')
#preprocessing of the main text of ad
    
#removing lower case for the main text
df['description'] = df['description'].apply(lambda x: 
    " ".join(x.lower() for x in x.split()))#" " between words
#removing punctuation in the main text
df['description'] = df['description'].str.replace('[^\w\s]','')
#remove stop words in the main text
df['description'] = df['description'].apply(lambda x: 
    " ".join(x for x in x.split() if x not in stop))

#stemming
##from nltk.stem import PorterStemmer
##st = PorterStemmer()
from nltk.stem.snowball import SnowballStemmer 
st = SnowballStemmer("russian")
df['description']=df['description'].apply(lambda x: 
    " ".join([st.stem(word) for word in x.split()]))
#print(st.stem("перепрыгивающий"))
#results in перепрыгива
##############################################################################
dataT=df['description']
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

#count_vect = CountVectorizer(max_features=10)
#dataT_counts = count_vect.fit_transform(dataT)
#print("countvect",dataT_counts.toarray().sum(axis=1))
#tfidf_transformer = TfidfTransformer(use_idf=True)
#dataT_tfidf = tfidf_transformer.fit_transform(dataT_counts).toarray()
#print(dataT_tfidf.sum(axis=1))


tfidf_vect=TfidfVectorizer(max_features=20)
dataT_tfidf = tfidf_vect.fit_transform(dataT).toarray()
print(dataT_tfidf.sum(axis=1))


df_try=df_try.join(pd.DataFrame(dataT_tfidf))
print("Size of our prepared dataset:", df_try.shape)
print("THE DATAFRAME TO PROCESS:")
display(df_try.head(4))
target=df_try["deal_prob"]
data=df_try.drop("deal_prob",axis=1)
###############################################################################
#PREPROCESSINGg



#from sklearn.preprocessing import MinMaxScaler
#data_scaler = MinMaxScaler(feature_range=(0, 1))
#data_scaled = ( data_scaler.fit_transform(data))
#data=pd.DataFrame(data_scaled,columns=data.columns)



#from sklearn.preprocessing import MinMaxScaler
#data_scaler = MinMaxScaler(feature_range=(0, 1))
#data_scaled = ( data_scaler.fit_transform(data))
#data=pd.DataFrame(data_scaled,columns=data.columns)

#from sklearn import preprocessing
#scaler=preprocessing.StandardScaler().fit(data)
#data_scaled=scaler.transform(data)
#data=pd.DataFrame(data_scaled,columns=data.columns)
###############################################################################





###############################################################################
#Run the algorithm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

X_train, X_test, y_train, y_test=train_test_split(data, target,  
                                                  test_size=0.3,random_state=1,
                                                  shuffle=True)

from sklearn.preprocessing import MinMaxScaler
data_scaler = MinMaxScaler(feature_range=(0, 1))
X_train =  data_scaler.fit_transform(X_train)
X_test =  data_scaler.transform(X_test)

print("Data before PCA: ",data.shape)
from sklearn.decomposition import PCA
N_before=data.shape
pca=PCA(n_components=0.9)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)
print("Data after PCA col: ",X_test.shape[1])
print("Data after PCA rows: ",X_test.shape[0]+X_train.shape[0])
import time
time.sleep(5)

from sklearn.preprocessing import MinMaxScaler
data_scaler = MinMaxScaler(feature_range=(0, 1))
X_train =  data_scaler.fit_transform(X_train)
X_test =  data_scaler.transform(X_test)

#fig=plt.figure(figsize=(8,4))
#ax1=fig.add_subplot(121)
#ax1.scatter(X_test.values[:,0],X_test.values[:,1],
#            c=y_test.values,cmap=plt.cm.Spectral,s=5)
#ax1.set_xlabel('PC1')
#ax1.set_ylabel('PC2')
#ax2=fig.add_subplot(122)
#ax2.scatter(X_test.values[:,1],y_test.values,c=y_test.values,
#            cmap=plt.cm.Spectral,s=5)
#ax2.set_xlabel('PC1')
#ax2.set_ylabel('target')
#fig.tight_layout()
#plt.show()
#print("SIZE",X_test.shape)

#print("Data after LDA: ",X_test.shape[1])
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#lda=LDA()
#X_train_lda=lda.fit_transform(X_train,y_train.toarray())
#X_test_lda=lda.transform(X_test)
#X_train=X_train_lda
#X_test=X_test_lda
#print("Data after LDA: ",X_test.shape[1])

from keras.models import Sequential
from keras.layers import Dense
from keras.constraints import max_norm
from keras.utils import np_utils
from keras.layers import Dropout

N_col=X_test.shape[1]
c=5
model=Sequential()
#The first layer
percentage1=1
delta1=int(round(N_col*percentage1,0))
model.add(Dense(output_dim=delta1,input_dim=N_col,init='normal',activation='relu'))
model.add(Dropout(.2))
#The second layer
percentage2=1
delta2=int(round(N_col*percentage2,0))
model.add(Dense(delta2,init='uniform',activation='relu'))
model.add(Dropout(.2))
#The third layer
percentage3=1
delta3=int(round(N_col*percentage3,0))
model.add(Dense(delta3,init='normal',activation='relu'))
model.add(Dropout(.2))
#,kernel_constraint=max_norm(2.)
#The forth layer
percentage4=1
delta4=int(round(N_col*percentage4,0))
model.add(Dense(delta4,init='normal',activation='relu'))
model.add(Dropout(.2))
#The fifth layer
percentage5=1
delta5=int(round(N_col*percentage5,0))
model.add(Dense(delta5,init='normal',activation='relu'))
model.add(Dropout(.2))


model.add(Dense(output_dim=1,activation='sigmoid'))

from keras import optimizers
model.compile(loss='mse',optimizer='rmsprop',metrics=["MSE"])
print("X_train: ",X_train.shape)
print("y_train: ",y_train.shape)

model.fit(X_train,y_train,nb_epoch=50,batch_size=1,verbose=2)



pred_train=model.predict(X_train)
pred_test=model.predict(X_test)

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

RMSE_train=np.sqrt( mean_squared_error(pred_train,y_train) )
MSE_train=RMSE_train*RMSE_train
print("MSE_train= ",MSE_train)
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(121)
ax.set_xlabel('y_train')
ax.set_ylabel('pred_train')
ax.plot(y_train,pred_train,color='green', marker='o', linewidth=0, markersize=2)
ax.annotate("MSE_train= %.4f" % MSE_train, xy=(0.4, 0.1))
ax.plot([y_train.min(),y_train.max()],[y_train.min(),y_train.max()])


RMSE_test=np.sqrt( mean_squared_error(pred_test,y_test) )
print("RMSE_test= ",RMSE_test)
ax = fig.add_subplot(122)
ax.set_xlabel('y_test')
ax.set_ylabel('pred_test')
ax.plot(y_test,pred_test,color='red', marker='o', linewidth=0, markersize=2)
ax.annotate("RMSE_test= %.4f" % RMSE_test, xy=(0.4, 0.1))
ax.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()])




#creates vgg16 features using images.jpg 
#and store them to vgg_features.csv
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.metrics import confusion_matrix

import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

pd.set_option('display.max_columns', 5000)
pd.set_option('display.max_rows', 1000)
#nrows to read first part of data
l=30000
df=pd.read_csv("train.csv",nrows=l)
print("Size of the initial dataset with NaN:", df.shape)
df=df.dropna().reset_index(drop=True)
print("Size of the initial dataset without NaN:", df.shape)
l=df.shape[0]
print("actual number of raws:",l)
#display(df.head(5))

###############################################################################
#creating dataframe df_try for analysis
df_try=pd.DataFrame()

#discretization of the target probability
#seuil=np.median(df["deal_probability"])
#print("seuil (to convert deal_propability to 0 or 1)=",seuil)
#df_try["deal_prob_disc"]=np.where(df["deal_probability"]>0,1,0)
df_try["deal_prob_cont"]=df["deal_probability"]

import matplotlib.pyplot as plt
import collections as cl
import operator
from PIL import Image

dfl=pd.DataFrame()
##############################################################################
print("Calculating vgg function:")
model=VGG16(include_top=True, weights='imagenet') 
def ox(name):
    impath="C:/Users/Павел/.spyder-py3/projects/avito_test/images/train_jpg/"
    ext=".jpg"
    filename=impath+name+ext
    
    img=image.load_img(filename,target_size=(224,224))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    x=preprocess_input(x)
    probabilites = model.predict(x)
    #print(probabilites.shape)
    #display(pd.DataFrame(probabilites).head())
    #vfvd
    #img_matrix=np.asarray(img)#kartinka v forme 3h mernoy matrici
    return probabilites[0]
dfl=df['image'].apply(ox).to_frame(name="vgg")
#display(dfl.head)

print("PostProcessing_raw:")
dfl.to_csv('Raw_vgg_features.csv',sep=',')#zapisivaem v fayl
print("PostProcessing_final:")
for i in range(0, 1000):
    df_try[str(i)] = dfl["vgg"].apply(lambda x: x[i])
df_try.to_csv('vgg_features.csv',sep=',')#zapisivaem v fayl



###############################################################################
#print("output")
#dg=pd.read_csv('vgg_features.csv',sep=',')
#display(dg.head(20))















#creates trivial image features like whiteness, uniformity,... using image.jpg
#and store them to image_features.csv
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.metrics import confusion_matrix
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

#BACKGROUND
#line_number=0
#print("line number: ",line_number)
#print("the name of the image: ",df["image"][line_number])
#name=df["image"][line_number]
#
#im=plt.imread(impath+name+ext)
##plt.imshow(im)
#print("dimmention de l'image:",im.shape)
#X=im.reshape(im.shape[0]*im.shape[1],im.shape[2])
#print(X)

#imgpil = Image.open(impath+name+ext)
#img=np.asarray(imgpil)
#imgpil = Image.fromarray(img)
#img0= np.zeros((100, 200, 3), dtype=np.uint8)
#plt.imshow(img)
#print("get_data:",imgpil.getdata()[1])#line of 3 colors





###############################################################################
print("Size and Dimension:")
import os
def getSize(name):
    impath="C:/Users/Павел/.spyder-py3/projects/avito_test/images/train_jpg/"
    ext=".jpg"
    #filename = images_path + filename
    st = os.stat(impath+name+ext)
    return st.st_size

def getDimensions(name):
    impath="C:/Users/Павел/.spyder-py3/projects/avito_test/images/train_jpg/"
    ext=".jpg"
    filename = impath+name+ext
    img = Image.open(filename)
    #img=np.asarray(img)
    img_size = img.size
    return img_size 

features=pd.DataFrame()
features['gD'] = df['image'].apply(getDimensions)

df_try['Dimension_0'] = features['gD'].apply(lambda x: x[0])
df_try['Dimension_1'] = features['gD'].apply(lambda x: x[1])
features=features.drop("gD",axis=1)

df_try['size'] = df['image'].apply(getSize)
df_try.to_csv('image_features.csv',sep=',')
###############################################################################
def color_analysis(img):
    # obtain the color palatte of the image 
    palatte = cl.defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1
    
    # sort the colors present in the image 
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]
        
    light_percent = round((float(light_shade)/shade_count)*100, 2)
    dark_percent = round((float(dark_shade)/shade_count)*100, 2)
    return light_percent, dark_percent
#------------------------------------------------------------------------------
print("Dullness, Whiteness:")
def perform_color_analysis(name):
    #path = images_path + img 
    impath="C:/Users/Павел/.spyder-py3/projects/avito_test/images/train_jpg/"
    ext=".jpg"
    filename = impath+name+ext
    #img = Image.open(filename)
    imgpil = Image.open(filename) #.convert("RGB")
    im=imgpil
    # cut the images into two halves as complete average may give bias results
    size = im.size
    halves = (size[0]/2, size[1]/2)
    im1 = im.crop((0, 0, size[0], halves[1]))
    im2 = im.crop((0, halves[1], size[0], size[1]))

    try:
        light_percent1, dark_percent1 = color_analysis(im1)
        light_percent2, dark_percent2 = color_analysis(im2)
    except Exception as e:
        return None

    light_percent = (light_percent1 + light_percent2)/2 
    dark_percent = (dark_percent1 + dark_percent2)/2 

    return [round(dark_percent,2), round(light_percent,2)]

features['ca'] = df['image'].apply(perform_color_analysis)
df_try['dullness'] = features['ca'].apply(lambda x: x[0])
df_try['whiteness'] = features['ca'].apply(lambda x: x[1])
features=features.drop("ca",axis=1)
    
df_try.to_csv('image_features.csv',sep=',')
###############################################################################
print("Uniformity of the image:")
from skimage import feature
#for low number the image is uniform
def average_pixel_width(path):
    #path = images_path + img 
    im = Image.open(path)    
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    #print(edges_sigma1)
    #imgpil = Image.fromarray(edges_sigma1)
    #plt.imshow(edges_sigma1)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw*100

df_try['pixel_width'] = df['image'].apply(lambda x : 
    average_pixel_width(impath+x+ext))
    
df_try.to_csv('image_features.csv',sep=',')

############################################################################
#super long
#from sklearn.cluster import KMeans
#from collections import Counter
#def get_dominant_color(path, k=4):
#    
#    image = Image.open(path)
#    image=np.asarray(image)
#    #reshape the image to be a list of pixels
#    image = image.reshape((image.shape[0] * image.shape[1], 3))
#
#    #cluster and assign labels to the pixels 
#    clt = KMeans(n_clusters = k)
#    labels = clt.fit_predict(image)
#
#    #count labels to find most popular
#    label_counts = Counter(labels)
#    print("lc ",label_counts)
#    print("lc ",label_counts.most_common(1)[0][0])
#    #subset out most popular centroid
#    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
#
#    return list(dominant_color)
#
#df_try['dominant_color'] = df['image'].apply(lambda x : 
#    get_dominant_color(impath+x+ext)[0])
#ffgffg
#df_try.to_csv('image_features.csv',sep=',')
#dg=pd.read_csv('image_features.csv',sep=',')
#display(dg.head(10))

##############################################################################
print("Average color:")
def get_average_color(name):
    #path = images_path + img 
    impath="C:/Users/Павел/.spyder-py3/projects/avito_test/images/train_jpg/"
    ext=".jpg"
    filename = impath+name+ext
    img = Image.open(filename)
    img=np.asarray(img)
    average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
    return average_color


features['gac'] = df['image'].apply(get_average_color)
df_try['average_red'] = features['gac'].apply(lambda x: x[0])
df_try['average_green'] = features['gac'].apply(lambda x: x[1])
df_try['average_blue'] = features['gac'].apply(lambda x: x[2])
features=features.drop("gac",axis=1)

df_try.to_csv('image_features.csv',sep=',')
###############################################################################

#dg=pd.read_csv('image_features.csv',sep=',')
display(df_try.head(20))















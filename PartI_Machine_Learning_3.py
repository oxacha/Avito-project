#Like Exo N2 "la regression logistique" and N3 "Support Vector Machine"
#from Python Machine Learning modul!
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.metrics import confusion_matrix
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1500000)
print("Hello")
#nrows to read first part of data
l=10000
df=pd.read_csv("train.csv",nrows=l)
print("Size of the initial dataset with NaN:", df.shape)
df=df.dropna()
print("Size of the initial dataset without NaN:", df.shape)
#df=df[df["parent_category_name"]!='NaN']

###############################################################################
#make all columns consisted of either 0 or 1
df_cities=pd.get_dummies(df["city"],prefix="city")
df_cities=df_cities.dropna().reset_index(drop=False)
df_cities=df_cities.drop("index",axis=1)

df_parent=pd.get_dummies(df["parent_category_name"],prefix="pcn")
df_parent=df_parent.dropna().reset_index(drop=False)
df_parent=df_parent.drop("index",axis=1)

lbs_seq=[0, df.item_seq_number.quantile(0.25), 
         df.item_seq_number.quantile(0.5),
         df.item_seq_number.quantile(0.75),np.max(df.item_seq_number)]
v_seq=pd.cut(df['item_seq_number'], lbs_seq)
df_seq=pd.get_dummies(v_seq,prefix="seq")
df_seq=df_seq.dropna().reset_index(drop=False)
df_seq=df_seq.drop("index",axis=1)

lbs_price=[0, df.price.quantile(0.25), df.price.quantile(0.5),
       df.price.quantile(0.65),np.max(df.price)]
v_price=pd.cut(df['price'], lbs_price)
df_price=pd.get_dummies(v_price,prefix="price")
df_price=df_price.dropna().reset_index(drop=False)
df_price=df_price.drop("index",axis=1)

###############################################################################
#Create df_try to run the alogorithm

#add target column
df_try=pd.DataFrame()
seuil=np.median(df["deal_probability"])
print("seuil (to convert deal_propability to 0 or 1)=",seuil)
df_try["deal_prob"]=np.where(df["deal_probability"]>0,1,0)

#add price categories, see df["price"]
df_try=df_try.join(df_price)
#add seq, see df['item_seq_number']
df_try=df_try.join(df_seq)
#add type, see df["parent_category_name"]
df_try=df_try.join(df_parent)
#add cities where annonce was published
#df_try=df_try.join(df_cities)
print("")
print("number of 1 (edinichek) in each column:")
print(np.sum(df_try))
print("")

target=df_try["deal_prob"]
data=df_try.drop("deal_prob",axis=1)
#data=df_try.iloc[:,-1]
#display(data.head(10))
print("Size of our prepared dataset:", df_try.shape)

#!!!uncomment to run the perfect model 
#only use to check the perfomance of the algorithm
#df_try['repeat']=df_try.deal_prob
###############################################################################
#Run the algorithm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
X_train, X_test, y_train, y_test=train_test_split(data, target,  
                                                  test_size=0.2,random_state=1)

#Chose the algorithm
#Logistic Regression (Exo 2)                                                 
clf=linear_model.LogisticRegression()
parameters={'C':[0.001, 0.01, 0.1, 1, 10, 100]}
#k-means (Exo 4)
#clf=neighbors.KNeighborsClassifier(n_neighbors=5,metric='minkowski')
#parameters={'n_neighbors':[2, 4, 6, 8, 10, 15, 20],
#            'metric':['minkowski','manhattan']}
#Arbres de decision
#clf=DecisionTreeClassifier(criterion="entropy",max_depth=4)
#parameters={'criterion':['entropy','gini']}
#SVM
#clf=svm.SVC()
#parameters={'C':[0.1,1,10],'kernel':['rbf','linear','poly'],
#            'gamma':[0.001, 0.1, 0.5, 1, 10]}



grid_clf=model_selection.GridSearchCV(estimator=clf, param_grid=parameters)
grille=grid_clf.fit(X_train,y_train)

y_pred=grid_clf.predict(X_test)
print("confusion matrix:")
cm=confusion_matrix(y_test,y_pred)
print(cm)
print("")
print("Metrics:")
print("Accuracy:", grid_clf.score(X_test,y_test))
from sklearn.metrics import f1_score
print("F1 score:", f1_score(y_test,y_pred))
from sklearn.metrics import cohen_kappa_score
print("Cohen kappa score:", cohen_kappa_score(y_test,y_pred))
print("Best fitting parametera of the model:",grid_clf.best_params_)


## Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split  #if error in model_selection use sklearn.crossvalidation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import pydotplus as pdp
from IPython.display import Image
from sklearn import svm,datasets
from sklearn.datasets import load_iris
import seaborn as sns
from sklearn.datasets import make_blobs


#### Making a Classifier

url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv'
#col_names = ['sepal_length','sepal_width','petal_length','petal_width','species']
dataset = pd.read_csv(url)
print(dataset.shape)   # shape of the DataFrame
print("---------------------------------------------------------------------------")
print(dataset.head(20)) 
print("---------------------------------------------------------------------------")
print(dataset.describe())  # statistics pertaining to the DataFrame (Column-wise)
print("---------------------------------------------------------------------------")
print(dataset.groupby('species').size())  # split the data into groups based on some criteria
# then tells the size of each group
print("---------------------------------------------------------------------------")


## Step1: Preprocessing Data

array = dataset.values   # creating a ndarray of the DataFrame
X = array[:,0:4]         # spliting the data of first 4 columns
Y = array[:,4]           # spliting the data in the species column

## Step2: Creating test and training data

validation_size = 0.20   # setting a percent size to allot to test data
seed = 7                 # controls the shuffling applied to the data before applying the split
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y,test_size = validation_size,random_state=seed)

print("---------------------------------------------------------------------------")

## Step3: Creating a model/classifier

##models = []
##models.append(('LR', LogisticRegression()))
##models.append(('LDA', LinearDiscriminantAnalysis()))
##models.append(('KNN', KNeighborsClassifier()))
##models.append(('CART', DecisionTreeClassifier()))
##models.append(('NB', GaussianNB()))
##models.append(('SVM', SVC()))
##
### Evaulate each model in turn
##
##results = []
##names = []
##scoring = 'accuracy'
##
### KFold provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).
### Each fold is then used once as a validation while the k - 1 remaining folds form the training set.
##
### cross_val_score evaluate a score by cross-validation
##
##for name, model in models:
##    kfold = model_selection.KFold(n_splits=10, random_state = seed)
##    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
##    results.append(cv_results)
##    names.append(name)
##    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
##    print(msg)
##
### Compare Algos and choose the model with best scores
##
##fig = plt.figure()
##fig.suptitle("Algo Comparison")
##x = fig.add_subplot(111)
##plt.boxplot(results)
##
##plt.show()
##
##
### Creating the clf
##
##clf = KNeighborsClassifier()
##
#### Step4: Apply fitting the data on the model
##
##clf.fit(X_train, Y_train)
##
#### Step5: Apply predict on your model
##
##predictions = clf.predict(X_validation)
##print(predictions)
##print(accuracy_score(Y_validation, predictions))
##print(confusion_matrix(Y_validation, predictions))
##print(classification_report(Y_validation, predictions))
##
##
##
#############Decision Tree Classifier##############
##
#### 1. Preprocessing Data
##cols=['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']
##pima=pd.read_csv('./PimaIndiansDiabetes.csv',header=None,names=cols)
##print(pima.head())
##feature_cols=['pregnant','age','insulin','bmi','glucose','pedigree','bp']
##x=pima[feature_cols]
##y=pima.label
##
#### 2. Creating test and training data
##x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
##
#### 3. Create model
##clf=DecisionTreeClassifier()
##
####clf=RandomForestClassifier(n_estimators=50)
##
#### 4. Apply fitting
##clf=clf.fit(x_train,y_train)
##
#### 5. Apply prediction
##y_pred = clf.predict(x_test)
##result = confusion_matrix(y_test,y_pred)
##print('Confusion Matrix')
##print(result)
##result1 = classification_report(y_test,y_pred)
##print('Classification Report: ',)
##print(result1)
##result2 = accuracy_score(y_test,y_pred)
##print('Accuracy:', result2)
##
### Export a decision tree in DOT format
##dot_data=export_graphviz(clf,out_file=None,filled=True,rounded=True,special_characters=True,feature_names=feature_cols,class_names=['0','1'])
##
### Load graph as defined by data in DOT format
##graph=pdp.graph_from_dot_data(dot_data)
##
### Draw graph
##graph.write_png('Tree.png')
##
### Show graph
##Image(graph.create_png())
##
##
##
####################Support vector Machine###########
##
#### 1. Preprocessing Data
##
##iris = datasets.load_iris()
##X = iris.data[:,:2]
##y = iris.target
##
### Create SVM Boundary
##
##x_min = X[:,0].min()-1 ##3.6
##x_max = X[:,0].max()+1 ##8.8
##y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
##
##h = (x_max/x_min)/100   #threshhold
##
### Create meshgrid from arange(x_min,x_max,h)
##xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
##
### Flatten using ravel and make contour
##x_plot = np.c_[xx.ravel(),yy.ravel()]
##
### Set regularization parameter as 1
##c = 1.0
##
#### 2. Create a support verctor classifier model using linear kernel and apply fitting
##svc_classifier = svm.SVC(kernel='linear',C=1.0).fit(X,y)
##
##
#### 3. Predict x_plot to get y parameter 
##z = svc_classifier.predict(x_plot)
##
### Reshape according to input variable so that plotting can occur
##z = z.reshape(xx.shape)
##
### Create a window of size 15x5
##plt.figure(figsize=(8,5))
##
### Create a subplot
##plt.subplot(131)
##
### Create contour(hyperplane) on the figure window
##plt.contour(xx,yy,z,cmap=plt.cm.tab10, alpha=0.1)
##
### Customize graph
##plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.Set1)
##plt.xlabel('Sepal-length')
##plt.ylabel('Petal-length')
##plt.xlim(xx.min(),xx.max())
##plt.title('Support Vector Classifier Linear Kernel')
##
#### svc with rbf kernel 
##svc_classifier=svm.SVC(kernel='rbf', gamma='auto',C=1.0).fit(X,y)
##z=svc_classifier.predict(x_plot)
##z=z.reshape(xx.shape)
##plt.subplot(132)
##plt.contour(xx,yy,z,cmap=plt.cm.tab10,alpha=0.3)
##plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.Set1)
##plt.xlabel('Sepal-length')
##plt.ylabel('Petal-length')
##plt.xlim(xx.min(),xx.max())
##plt.title('Support Vector Classifier RBF Kernel')
##
#### svc with poly kernel
##svc_classifier=svm.SVC(kernel='poly', gamma='auto',C=1.0).fit(X,y)
##z=svc_classifier.predict(x_plot)
##z=z.reshape(xx.shape)
##plt.subplot(133)
##plt.contour(xx,yy,z,cmap=plt.cm.tab10,alpha=0.3)
##plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.Set1)
##plt.xlabel('Sepal-length')
##plt.ylabel('Petal-length')
##plt.xlim(xx.min(),xx.max())
##plt.title('Support Vector Classifier Poly Kernel')
##plt.show()
##
##
###############Regression#################
##
#### 1. Preprocessing Data
##X, y = load_iris(return_X_y=True)
##
#### 2. Create classifier model and apply fitting
##clf = LogisticRegression(random_state=0).fit(X, y)
##
#### 3. Predict
##print(clf.predict(X[:2, :]))
##print(clf.predict_proba(X[:2, :]))
##print(clf.score(X, y))
##
##
##
##################K-means Clustering######
##
#### 1. Preprocessing Data
##X,y=iris.data,iris.target
##
#### 2. Create classifier model and apply fitting
##k_means = KMeans(n_clusters=3,random_state=1)
##k_means.fit(X)
###model.fit(X,y);for supervised learning model.fit(X); Unsupervised learning
##
#### 3. Predict
##y_pred = k_means.predict(X) #model.predict(X)
##plt.scatter(X[:,0],X[:,1],c=y_pred)
##plt.show()
##
##
################Naive Bayes' Algorithm###########
##
#### 1. Preprocessing Data
##X,y=make_blobs(300,2,centers=2,random_state=2,cluster_std=1.5)
##plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='jet')
##
#### 2. Create classifier model and apply fitting
##model_GBN=GaussianNB()
##model_GBN.fit(X,y)
##
##rng=np.random.RandomState(0)
##
#### 3. Predict
##Xnew=[-6,-14]+[14,18]*rng.rand(2000,2)
##ynew=model_GBN.predict(Xnew)
##plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='jet')
##lim=plt.axis()
##plt.scatter(Xnew[:,0],Xnew[:,1],c=ynew,cmap='summer',alpha=0.1)
##plt.axis(lim)
##yprob=model_GBN.predict_proba(Xnew)
##yprob[-10:].round(3)
##plt.show()
##
##
##
##
##
##
##
##
##
##

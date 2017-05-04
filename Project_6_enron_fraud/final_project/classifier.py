# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:39:53 2017

@author: Badrinath
"""
import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.decomposition import RandomizedPCA
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
import numpy as np
from numpy.random import random_integers
import pandas as pd
from scipy.stats import pointbiserialr, spearmanr
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus','deferral_payments','deferred_income',
                 'exercised_stock_options',
                 'loan_advances',
                 'long_term_incentive',
                 'restricted_stock',
                 'total_payments',
                 'total_stock_value',
                 'from_messages',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
def findoutlier(data_dict, x, y):
    # yellow if poi and black if not poi
    data = featureFormat(data_dict, [x, y, 'poi'])
    for d in data:
        if d[2]:
            color = 'yellow'
        else:
            color = 'black'
        plt.scatter(d[0], d[1], c=color)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

#findoutlier(data_dict, 'total_payments', 'total_stock_value')
#findoutlier(data_dict, 'salary', 'bonus')

outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
for key in outliers:
    data_dict.pop(key, 0)

#double check the plots and see if you can remove more
#findoutlier(data_dict, 'total_payments', 'total_stock_value')
#findoutlier(data_dict, 'salary', 'bonus')

#looking at the data again we find another outlier
data_dict.pop("LOCKHART EUGENE E",0) # All NaNs

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#lets clean data a bit for feature creation and create new features
              
for name in my_dataset.keys():
    if my_dataset[name]['salary']=='NaN':
        my_dataset[name]['salary']=0.
for name in my_dataset.keys():
    if my_dataset[name]['bonus']=='NaN':
        my_dataset[name]['bonus']=0.
for name in my_dataset.keys():
    data = my_dataset[name]
    data['net_salary'] = data['salary']+data['bonus']
for name in my_dataset.keys():
    data = my_dataset[name]
    if data["from_poi_to_this_person"] == 'NaN' or data["to_messages"] == 'NaN':
        data['from_fraction_messages'] = 0.
    else:
        data['from_fraction_messages'] = float(data['from_poi_to_this_person']) / data['to_messages']
for name in my_dataset.keys():
    data = my_dataset[name]
    if data["from_this_person_to_poi"] == 'NaN' or data["to_messages"] == 'NaN':
        data['to_fraction_messages'] = 0.
    else:
        data['to_fraction_messages'] = float(data['from_this_person_to_poi']) / data['to_messages']
for name in my_dataset.keys():
    data = my_dataset[name]
    if data["shared_receipt_with_poi"] == 'NaN' or data["to_messages"] == 'NaN':
        data['shared_fraction_messages'] = 0.
    else:
        data['shared_fraction_messages'] = float(data['shared_receipt_with_poi']) / data['to_messages']

for name in my_dataset.keys():
    data = my_dataset[name]
    if data['total_payments'] == 'NaN':
        data['total_payments'] = 0 

for name in my_dataset.keys():
    data = my_dataset[name]
    if data['total_payments']==0:
        data['bonus_total_ratio']=0
    else:
        data['bonus_total_ratio'] = float(data['bonus']) / data['total_payments']
        
for name in my_dataset.keys():
    data = my_dataset[name]
    if data['total_payments']==0:
        data['salary_total_ratio']=0
    else:
        data['salary_total_ratio'] = float(data['salary']) / data['total_payments']

for name in my_dataset.keys():
    data = my_dataset[name]
    if data['bonus']==0:
        data['salary_bonus_ratio']=0
    else:
        data['salary_bonus_ratio'] = float(data['salary']) / data['bonus']


for name in my_dataset.keys():
    data = my_dataset[name]
    if data['total_stock_value'] == 'NaN':
        data['total_stock_value'] = 0

for name in my_dataset.keys():
    data = my_dataset[name]
    if data['restricted_stock'] == 'NaN':
        data['restricted_stock'] = 0
       
for name in my_dataset.keys():
    data = my_dataset[name]
    if data['total_stock_value']==0:
        data['res_total_stock_ratio']=0
    else:
        data['res_total_stock_ratio'] = float(data['restricted_stock']) / data['total_stock_value']
del data



### Extract features and labels from dataset for local testing
features_list = features_list + ['res_total_stock_ratio','salary_total_ratio','bonus_total_ratio','from_fraction_messages','to_fraction_messages','salary_bonus_ratio']
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.pipeline import Pipeline




scaler = MinMaxScaler()
features = scaler.fit_transform(features)

#scaler = StandardScaler()
#features = scaler.fit_transform(features)


features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#Classifier Selection
classifiers = [
    SGDClassifier(random_state=42),
    DecisionTreeClassifier(random_state=42),
    ExtraTreeClassifier(random_state=42),
    ExtraTreesClassifier(random_state=42),
    RandomForestClassifier(random_state=42),
    AdaBoostClassifier(random_state=42),
    GradientBoostingClassifier(random_state=42),
    GaussianNB(),
    LogisticRegression(random_state=42)]

#pipeline classifiers
classifiers1 = [
    Pipeline(steps=[('scaler', scaler), ('SGD', SGDClassifier(random_state=42))]),
    Pipeline(steps=[('scaler', scaler), ('Dtc', DecisionTreeClassifier(random_state=42))]),
    Pipeline(steps=[('scaler', scaler), ('Etct', ExtraTreeClassifier(random_state=42))]),
    Pipeline(steps=[('scaler', scaler), ('Etce', ExtraTreesClassifier(random_state=42))]),
    Pipeline(steps=[('scaler', scaler), ('Rfc', RandomForestClassifier(random_state=42))]),
    Pipeline(steps=[('scaler', scaler), ('Adaboost', AdaBoostClassifier(random_state=42))]),
    Pipeline(steps=[('scaler', scaler), ('gradboost', GradientBoostingClassifier(random_state=42))]),
    Pipeline(steps=[('scaler', scaler), ('Gaussiannb', GaussianNB())]),
    Pipeline(steps=[('scaler', scaler), ('LRC', LogisticRegression(random_state=42))])]






log_cols = ["Classifier", "f1_score"]
#log_cols = ["Classifier", "precision_score"]
#log_cols = ["Classifier", "recall_score"]
log 	 = pd.DataFrame(columns=log_cols)

sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.3, random_state=42)

df = pd.DataFrame(features)
df_test = pd.DataFrame(labels)
X = df.values
y = df_test.values

acc_dict = {}

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #for clf1,clf in zip(classifiers,classifiers1):
    for clf in classifiers:
        #name = clf1.__class__.__name__
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_test)
        acc = f1_score(y_test, train_predictions)
        #acc = precision_score(y_test, train_predictions)
         #acc = recall_score(y_test, train_predictions)
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc

for clf in acc_dict:
	acc_dict[clf] = acc_dict[clf] / 1000.0
	log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
	log = log.append(log_entry)

plt.xlabel('f1_score',fontsize=18)
#plt.xlabel('precision_score')
#plt.xlabel('recall_score')
#plt.title('Classifier Accuracy')
sns.set(font_scale=2)
sns.set_color_codes("muted")
b=sns.barplot(x='f1_score', y='Classifier', data=log, color="b")
#sns.barplot(x='precision_score', y='Classifier', data=log, color="b")
#sns.barplot(x='recall_score', y='Classifier', data=log, color="b")
plt.show()

"""#performance on test set of each parameter to see how they perform
X = features
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
plt.figure(figsize=(15,10))

#N Estimators
plt.subplot(2,3,1)
feature_param = range(1,100)
scores=[];f1=[]
for feature in feature_param:
    clf = GradientBoostingClassifier(n_estimators=feature,random_state=42)
    c = clf.fit(X_train,y_train)
    f1.append(f1_score(c.predict(X_test),y_test))
    scores.append(clf.score(X_test,y_test))
#plt.plot(scores, '.-')
plt.plot(f1, '.-')
plt.axis('tight')
plt.xlabel('parameter')
plt.ylabel('f1 score')
# plt.ylabel('score')
plt.title('N Estimators')
plt.grid()



#Max Features
plt.subplot(2,3,2)
feature_param = ['auto','sqrt','log2',None,1,2,3,4,5,6,7,8,9,10]
scores=[];f1=[]
for feature in feature_param:
    clf = GradientBoostingClassifier(max_features=feature,random_state = 42)
    c = clf.fit(X_train,y_train)
    f1.append(f1_score(c.predict(X_test),y_test))
    scores.append(clf.score(X_test,y_test))
#plt.plot(scores, '.-')
plt.plot(f1, '.-')
plt.axis('tight')
plt.xlabel('parameter')
plt.ylabel('f1 score')
# plt.xlabel('parameter')
# plt.ylabel('score')
plt.title('Max Features')
plt.grid()

#Max Depth
plt.subplot(2,3,3)
feature_param = range(1,20)
scores=[];f1=[]
for feature in feature_param:
    clf = GradientBoostingClassifier(max_depth=feature,random_state=42)
    c = clf.fit(X_train,y_train)
    f1.append(f1_score(c.predict(X_test),y_test))
    scores.append(clf.score(X_test,y_test))
#plt.plot(feature_param, scores, '.-')
plt.plot(feature_param, f1, '.-')
plt.axis('tight')
plt.xlabel('parameter')
plt.ylabel('f1 score')
# plt.xlabel('parameter')
# plt.ylabel('score')
plt.title('Max Depth')
plt.grid()

#Min Samples Split
plt.subplot(2,3,4)
feature_param = range(2,50)
scores=[];f1=[]
for feature in feature_param:
    clf = GradientBoostingClassifier(min_samples_split =feature,random_state=42)
    c = clf.fit(X_train,y_train)
    f1.append(f1_score(c.predict(X_test),y_test))
    scores.append(clf.score(X_test,y_test))
#plt.plot(feature_param, scores, '.-')
plt.plot(feature_param, f1, '.-')
plt.axis('tight')
plt.xlabel('parameter')
plt.ylabel('f1 score')
# plt.xlabel('parameter')
# plt.ylabel('score')
plt.title('Min Samples Split')
plt.grid()

#Min Weight Fraction Leaf
plt.subplot(2,3,5)
feature_param = np.linspace(0,.5,20)
scores=[];f1=[]
for feature in feature_param:
    clf = GradientBoostingClassifier(min_weight_fraction_leaf =feature,random_state=42)
    c = clf.fit(X_train,y_train)
    f1.append(f1_score(c.predict(X_test),y_test))
    scores.append(clf.score(X_test,y_test))
#plt.plot(feature_param, scores, '.-')
plt.plot(feature_param, f1, '.-')
plt.axis('tight')
plt.xlabel('parameter')
plt.ylabel('f1 score')
# plt.xlabel('parameter')
# plt.ylabel('score')
plt.title('Min Weight Fraction Leaf')
plt.grid()

#Max Leaf Nodes
plt.subplot(2,3,6)
feature_param = range(2,30)
scores=[];f1=[]
for feature in feature_param:
    clf = GradientBoostingClassifier(max_leaf_nodes=feature,random_state=42)
    c = clf.fit(X_train,y_train)
    f1.append(f1_score(c.predict(X_test),y_test))
    scores.append(clf.score(X_test,y_test))
#plt.plot(feature_param, scores, '.-')
plt.plot(feature_param, f1, '.-')
plt.axis('tight')
plt.xlabel('parameter')
plt.ylabel('f1 score')
# plt.xlabel('parameter')
# plt.ylabel('score')
plt.title('Max Leaf Nodes')
plt.grid()"""



#grid search CV to hypertune parameters


"""parameter_candidates = [
  {'n_estimators': [3,5,7,10,50,100], 'max_depth': [1,2,3,4,5]},
  {'min_samples_split': [2, 3, 4, 5,10, 20], 'min_weight_fraction_leaf': [0.05,0.1,0.5], 'max_leaf_nodes': [2,3,4,5,10,20],'max_features': [2,3,4,5,6,7,8,9,10],'loss':['deviance','exponential'],'learning_rate':[.1,.3,.7,1]}]
gr = GradientBoostingClassifier(random_state=42)
clf = GridSearchCV(gr, parameter_candidates, n_jobs=-1, scoring = 'f1')
clf.fit(features_train,labels_train)
clf.best_estimator_"""



# this method is computationally very intensive
"""#sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.3, random_state=42)
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
df = pd.DataFrame(features)
df_test = pd.DataFrame(labels)
X = df.values
y = df_test.values

parameter_candidates = [
  {'n_estimators': [3,5,7,10,35,50,100], 'max_depth': [1,2,3,4,5,6,7,8,9,10]},
  {'min_samples_split': [2, 3, 4, 5,6,7,8,9,10], 'min_weight_fraction_leaf': [0.05,0.1,0.3,0.5], 'max_leaf_nodes': [2,3,4,5,6,7,8,9,10],'max_features': [2,3,4,5,6,7]}]

performance_df = pd.DataFrame()
best_est=[];f1=[];rec=[];pre=[]
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = GridSearchCV(estimator=GradientBoostingClassifier(random_state=42), param_grid=parameter_candidates, n_jobs=-1,scoring='f1')
    yy=[float(i[0]) for i in y_train]
    c1 = clf.fit(X_train, yy)
    c2 = c1.best_estimator_
    train_predictions = c2.predict(X_test)
    acc = f1_score(y_test, train_predictions)
    best_est.append(c1.best_estimator_)
    f1.append(acc)
    rec.append(recall_score(y_test, train_predictions))
    pre.append(precision_score(y_test, train_predictions))
performance_df["best_estimator"] = best_est
performance_df["f1_score"] = f1
performance_df["recall_score"] = rec
performance_df["precision_score"] = pre
performance_df = performance_df.sort('f1_score',ascending=False)"""


    
    
    
    
    
    
    
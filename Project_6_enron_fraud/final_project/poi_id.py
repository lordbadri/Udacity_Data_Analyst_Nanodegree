#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot as plt
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.decomposition import RandomizedPCA
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
import numpy as np
from numpy.random import random_integers
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from scipy.stats import pointbiserialr, spearmanr
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
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
    data = featureFormat(data_dict, [x, y, 'poi'])
    for d in data:
        if d[2] == True:
            color = 'yellow'
        else:
            color = 'black'
        plt.scatter(d[0], d[1], c=color)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

plt.figure()
findoutlier(data_dict, 'total_payments', 'total_stock_value')
plt.figure()
findoutlier(data_dict, 'salary', 'bonus')

outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
for key in outliers:
    data_dict.pop(key, 0)

#double check the plots and see if you can remove more
plt.figure()
findoutlier(data_dict, 'total_payments', 'total_stock_value')
plt.figure()
findoutlier(data_dict, 'salary', 'bonus')

#looking at the data again we find another outlier
data_dict.pop("LOCKHART EUGENE E",0) # All NaNs

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#lets clean data a bit for feature creation and create new features
# Cleaning here means setting the NaN values to zero and also while creating freatures we must take care not to divide over 0.
              
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

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

from sklearn.linear_model import LogisticRegression
#clf = LogisticRegression(C=1000,class_weight ='balanced',random_state = 42,tol = 1e-4,penalty ='l1')
#clf=LogisticRegression()

from sklearn.svm import SVC
#clf = SVC(kernel='linear',random_state = 42)

from sklearn.tree import DecisionTreeClassifier
#clf=DecisionTreeClassifier(max_features=4,min_samples_split=16, max_depth=2,min_samples_leaf=9,min_weight_fraction_leaf=0.057, max_leaf_nodes=3, random_state=42, class_weight='balanced')
#clf=DecisionTreeClassifier(min_samples_split=100,random_state=42)
#clf=DecisionTreeClassifier(random_state=42)


from sklearn.linear_model import SGDClassifier
#clf = SGDClassifier(random_state=42)

from sklearn.ensemble import ExtraTreesClassifier
#clf=ExtraTreesClassifier(random_state=42)
"""clf = ExtraTreesClassifier(bootstrap=False, class_weight='balanced',
           criterion='gini', max_depth=2, max_features='auto',
           max_leaf_nodes=None, min_impurity_split=1e-07,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=5, n_jobs=1,
           oob_score=False, random_state=42, verbose=0, warm_start=False)"""

from sklearn.tree import ExtraTreeClassifier
#clf = ExtraTreeClassifier(random_state=42)

from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(random_state=42)

from sklearn.ensemble import GradientBoostingClassifier

#clf = GradientBoostingClassifier(random_state=42)

clf = GradientBoostingClassifier(criterion='friedman_mse', init=None,
                                 learning_rate=1, loss='deviance', max_depth=2,
                                 max_features=3, max_leaf_nodes=2, min_impurity_split=1e-7,
                                 min_samples_leaf=1, min_samples_split=3,
                                 min_weight_fraction_leaf=0.05, n_estimators=4,
                                 presort='auto', random_state=42, subsample=1.0, verbose=0,
                                 warm_start=False)

"""clf =GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=1, loss='deviance', max_depth=3,
              max_features=4, max_leaf_nodes=5, min_impurity_split=1e-07,
              min_samples_leaf=1, min_samples_split=5,
              min_weight_fraction_leaf=0.05, n_estimators=10,
              presort='auto', random_state=42, subsample=1.0, verbose=0,
              warm_start=False)"""




from sklearn.ensemble import AdaBoostClassifier
#clf=AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1, n_estimators=10, random_state=42)

#clf=MLPClassifier(hidden_layer_sizes=(512, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=2000, shuffle=True, random_state=42, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#clf = MLPClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

selector = SelectKBest(score_func=f_classif, k=4).fit(features,labels)
features = selector.transform(features)


features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


#scaler = MinMaxScaler()
#features = scaler.fit_transform(features)

gradboost = clf
scaler = MinMaxScaler()

pipe = Pipeline(steps=[('scaler', scaler), ('gradboost', gradboost)])         
pipe.fit(features_train,labels_train)
pred = pipe.predict(features_test)
#pipe.score(features_test,labels_test)


"""features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)"""
                                
"""n_components = 4
pca = RandomizedPCA(n_components=n_components).fit(features_train)
features_train_pca = pca.transform(features_train)
features_test_pca = pca.transform(features_test)"""


"""clf = clf.fit(features_train,labels_train)
model = SelectFromModel(clf, prefit=True)
features_train = model.transform(features_train)
features_test = model.transform(features_test)"""

#clf = clf.fit(features_train,labels_train)
#pred = clf.predict(features_test)
print(accuracy_score(pipe.predict(features_train),labels_train))
print(accuracy_score(pred,labels_test))
print(precision_score(pred,labels_test))
print(recall_score(pred,labels_test))
print(f1_score(pred,labels_test))
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(pipe, my_dataset, features_list)



        

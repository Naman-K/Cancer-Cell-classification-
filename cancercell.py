# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 01:29:37 2019

@author: NamanK
"""

import sklearn 
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()

label_names = data['target_names']
label = data['target']
feature_names = data['feature_names']
features = data['data']

from sklearn.model_selection import train_test_split  #Model Selection not cross validation
x_train, x_test, label_train, label_test = train_test_split(features,label, test_size=0.20)

from sklearn import svm
model = svm.SVC(kernel='linear',degree=3,gamma=1)

model.fit(x_train, label_train)
model.score(x_test, label_test)

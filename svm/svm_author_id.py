#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

from sklearn import svm

clf = svm.SVC(C=10000.0, kernel='rbf')

print "doing stuff..."

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)
print "accuracy:", round(accuracy, 3)

'''
a1 = pred[10]
a2 = pred[26]
a3 = pred[50]

print "a1: ", a1, " a2: ", a2, " a3: ", a3
'''

import numpy as np
count = np.count_nonzero(pred == 1)
print "instances: ", count


#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
from sklearn.naive_bayes import GaussianNB
import sys
from email_preprocess import preprocess
from time import time
sys.path.append("../tools/")



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
clf = GaussianNB()
tfit_start = time()
clf.fit(features_train, labels_train)
print "Fitting Time: ", str(round(time() - tfit_start, 3))

tpred_start = time()
clf.predict(features_test[0])
print "Prediction Time: ", str(round(time() - tpred_start, 3))

print "Fitted score is:", clf.score(features_test, labels_test)


#########################################################



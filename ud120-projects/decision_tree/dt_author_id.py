#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.tree import DecisionTreeClassifier

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
elems_num = len(features_train)
#########################################################
### your code goes here ###
clf = DecisionTreeClassifier(random_state=0, min_samples_split=40)
tfit_start = time()
clf.fit(features_train[:elems_num], labels_train[:elems_num])
print "Fitting Time: ", str(round(time() - tfit_start, 3))

#X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired

tpred_start = time()
#print "10 ", clf.predict(features_test[10])
#print "26 ", clf.predict(features_test[26])
#print "50 ", clf.predict(features_test[50])
p = clf.predict(features_test)
print "Chris ", sum(p)
print "1's: ", len([a for a in p if a == 1])
print "0's: ", len([a for a in p if a == 0])
print "Prediction Time: ", str(round(time() - tpred_start, 3))

print "Fitted score is:", clf.score(features_test, labels_test)


#########################################################



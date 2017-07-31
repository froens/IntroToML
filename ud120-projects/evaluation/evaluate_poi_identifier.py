#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys

from sklearn.metrics import recall_score, precision_score

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=0)
print(sum(y_test))
### your code goes here
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

print "precision: ", precision_score(predictions, true_labels) # .75, not accepted
print "recall: ", recall_score(predictions, true_labels) #.66, not accepted

print "precision: ", precision_score(true_labels, predictions) #.66, accepted
print "recall: ", recall_score(true_labels, predictions) #.75, accepted

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

a = classifier.predict(X_test)
print a

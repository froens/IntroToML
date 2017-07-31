#!/usr/bin/python

import sys
import pickle

from sklearn.preprocessing import MinMaxScaler

sys.path.append("../tools/")
import matplotlib.pyplot as plt
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from mpl_toolkits.mplot3d import Axes3D
import math

def plot3d(f1, f2, f3, data_dict, labels):
    fig3d = plt.figure()
    ax = fig3d.add_subplot(111, projection='3d')

    f1_list = [v[f1] if v[f1] != 'NaN' else 0 for (k, v) in data_dict.iteritems()]
    f2_list = [v[f2] if v[f2] != 'NaN' else 0 for (k, v) in data_dict.iteritems()]
    f3_list = [v[f3] if v[f3] != 'NaN' else 0 for (k, v) in data_dict.iteritems()]

    for (i, (x, y, z)) in enumerate(zip(f1_list, f2_list, f3_list)):
        c = 'r' if labels[i] > 0 else 'b'
        ax.scatter(x, y, z, color=c)

    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_zlabel(f3)
    plt.show()

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 #'deferral_payments',
                 'total_payments',
                 'loan_advances',
                 'bonus',

                 #'deferred_income',
                 #'expenses',

                 'total_stock_value',
                 #'restricted_stock_deferred',
                 'exercised_stock_options',
                 'restricted_stock',

                 #'other',
                 'long_term_incentive',

                 #'director_fees',
                 #'to_messages',
                 #'email_address',
                 #'from_poi_to_this_person',
                 #'from_this_person_to_poi',
                 #'shared_receipt_with_poi'
                 #'from_messages',

                 ### Self-made features
                 'from_poi_freq',
                 'to_poi_freq',
                 ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = dict(pickle.load(data_file))

labels = [data_dict[k]['poi'] for k in data_dict.keys()]

### Task 2: Remove outliers

#Visualize the outliers
visual_inspection = False

#Remove outliers
outliers = ['TOTAL']
for o in outliers:
    data_dict.pop(o)

feature_names = [f for f in features_list if f != 'poi']

#Count number of non-empty data fields in entries
#for (k,v) in data_dict.iteritems():
#    values = dict(v).values()
#    s = sum([1 if v != 'NaN' else 0 for v in values])
#    print k, s

if visual_inspection:
    for f in feature_names:
        values_with_names =[(k, data_dict[k][f]) for k in data_dict.keys()]

        print f, min([(n,v) for (n,v) in values_with_names if v != 'NaN'], key=lambda (name, value): value), max([(n,v) for (n,v) in values_with_names if v != 'NaN'], key=lambda (name, value): value)

        #values_with_names = sorted([(k, data_dict[k][f]) for k in data_dict.keys()], key=lambda (name, value): value)
        #labels = sorted([(k, data_dict[k][f]) for k in data_dict.keys()], key=lambda (name, value): value)
        values = [v for (n,v) in values_with_names]

        fig = plt.figure()
        plt.title(f)
        ax = fig.add_subplot(111)
        for i, v in enumerate(values):
            c = 'r' if labels[i] > 0 else 'b'
            col = ax.scatter(x=v, y=0, color=c)
        plt.show()




### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.


my_dataset = data_dict

for (k,v) in my_dataset.iteritems():
    ### New feature:
    ### from_poi_freq
    to_messages = float(v["to_messages"])
    from_poi_to_this_person = float(v["from_poi_to_this_person"])
    from_poi_freq = from_poi_to_this_person / to_messages if to_messages != 0 else 1
    v["from_poi_freq"] = from_poi_freq if not math.isnan(from_poi_freq) else 0
    # print v["from_poi_freq"]

    ### New feature:
    ### to_poi_freq
    from_this_person_to_poi = float(v["from_this_person_to_poi"])
    from_messages = float(v["from_messages"])
    to_poi_freq = from_this_person_to_poi / from_messages if from_messages != 0 else 1
    v["to_poi_freq"] = to_poi_freq if not math.isnan(to_poi_freq) else 0
    # print v["to_poi_freq"]



plot3d('exercised_stock_options', 'bonus', 'to_poi_freq', data_dict, labels)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

# from sklearn.svm import SVC
# clf = SVC(kernel='linear')

from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()

# classifiers = [
#     #KNeighborsClassifier(3),
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma=2, C=1),
#     GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     MLPClassifier(alpha=1),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis()]



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
features_train = scaler.fit_transform(features_train, labels_train)

from sklearn.feature_selection import chi2, f_classif
selection = SelectKBest(chi2, k='all')
selection.fit(features_train, labels_train)
scores = selection.scores_
for (f,s) in zip(feature_names, scores):
   print f , s


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
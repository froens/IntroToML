#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

features = ["salary", "bonus", "director_fees", "restricted_stock"]
### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_with_sal = dict([(k, data_dict[k]) for k in data_dict if data_dict[k]["salary"] != "NaN"])

for i in range(0,4):
    d_max = max(data_with_sal, key=lambda x: data_with_sal[x]["salary"])
    print d_max
    data_with_sal.pop(d_max, 0)


data = featureFormat(data_dict, features)




### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()



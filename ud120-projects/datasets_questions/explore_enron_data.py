#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle, math

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

total_payment_nan = [k for k in enron_data if enron_data[k]['total_payments'] == 'NaN']
poi_total_payment_nan = [k for k in total_payment_nan if enron_data[k]['poi']]

total_payment_pct = float(len(total_payment_nan)) / float(len(enron_data))
total_payment_pct = float(len(poi_total_payment_nan)) / float(len(enron_data))
#salaries = [enron_data[k]['salary'] for k in enron_data if isinstance(enron_data[k]['salary'], int )]
emails = [enron_data[k]['email_address'] for k in enron_data if enron_data[k]['email_address'] != 'NaN']
i = 1

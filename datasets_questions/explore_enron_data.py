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

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

'''
print len(enron_data.keys())
print len(enron_data[enron_data.keys()[0]])

'''

#print enron_data[enron_data.keys()[0]]

#print enron_data["PRENTICE JAMES"]["total_stock_value"]
#print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
#print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

'''
print enron_data["SKILLING JEFFREY K"]["total_payments"]
print enron_data["LAY KENNETH L"]["total_payments"]
print enron_data["FASTOW ANDREW S"]["total_payments"]
'''

'''
import math
count = 0
for k in enron_data:
	if (math.isnan(float(enron_data[k]["salary"])) == False):
		count += 1
print count

count = 0
for k in enron_data:
	if (enron_data[k]["email_address"] != "NaN"):
		count += 1
print count
'''

import math
count = 0
poi = 0
for k in enron_data:
	if (enron_data[k]["poi"] == True):
		poi += 1
		if (math.isnan(float(enron_data[k]["total_payments"])) == False):
			count += 1
print poi
print count
print count / poi

#print enron_data[enron_data.keys()[0]]
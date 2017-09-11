#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
import numpy as np
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop("TOTAL", 0)
data = featureFormat(data_dict, features)
	
### your code below
count = 0
for point in data:
	count += 1
	salary = point[0]
	bonus = point[1]
	if (bonus > 5000000) and (salary > 1000000):
		print bonus, ", ", salary
	matplotlib.pyplot.scatter(salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()



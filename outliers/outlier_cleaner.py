#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
	cleaned_data = []
	error_values = []
	
	### your code goes here		
	for pred, actual in zip(predictions, net_worths):
		error_values.append(abs(pred - actual))
		
	cleaned_data = zip(ages, net_worths, error_values)
	cleaned_data.sort(key=lambda x: x[2])
	cleaned_data = cleaned_data[0:81]
	
	return cleaned_data



'''

y = np.exp(b0 + (b1 * x) ) / ( 1 + np.exp(b0 + b1 * x ) )
	
'''
from random import randrange
from math import sqrt
from csv import reader
from random import seed
import numpy as np



# method for reading values from csv file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		readCSV = reader(file)
		for row in readCSV:
			if not row:
				continue
			dataset.append(row)
	return dataset		

# method for converting string to float
def str_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# methods for implementing Logistic regression

def variance(val, mean):
	return sum([(x - mean)**2 for x in val ])

def covariance(x, x_mean, y, y_mean):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - x_mean ) * (y[i] - y_mean )
	return covar
		
def coeff(train_dataset):
	x = [row[0] for row in train_dataset]
	y = [row[1] for row in train_dataset]
	x_mean, y_mean = np.mean(x), np.mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]

def root_mean_square(actual, predictions):
	sum_error = 0.0
	for i in range(len(actual)):
		predicted_error = predictions[i] - actual[i]
		sum_error += (predicted_error**2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)	

def simple_linear_regression(train_dataset, test_dataset):
	predictions = list()
	b0, b1 = coeff(train_dataset)
	for row in test_dataset:
		y = np.exp(b0 + (b1 * row[0]) ) / ( 1 + np.exp(b0 + b1 * row[0] ) )
		predictions.append(y)
	return predictions	


#main
seed(1)
train_dataset_filename = 'train.csv'
test_dataset_filename = 'test.csv'
train_dataset = load_csv(train_dataset_filename)
for i in range(len(train_dataset[0])):
	str_to_float(train_dataset, i)	
test_dataset = load_csv(test_dataset_filename)
for i in range(len(test_dataset[0])):
	str_to_float(test_dataset, i)
predictions = simple_linear_regression(train_dataset, test_dataset)
val = np.mean(predictions)

if val > 0.5 :
	print(1)
	
if 	val < 0.5:
	print(0)


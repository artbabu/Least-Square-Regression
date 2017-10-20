#!/usr/bin/env python
import numpy as np
import math
import matplotlib.pyplot as plt

#Code fragments provided by Raj Patel

#Load your dataset here



#Create your Features "x" and Target Values "targets" here from dataset
#preprocessing steps, standardize features


## Cut off value for the train set
#use first 100 datapoints for training, the rest for testing.  you need to change this.
N_TRAIN = 100;


## Training Features (Change accordingly)
## Testing Features (Change accordingly)
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]

## Training values (Change accordingly)
## Testing values (Change accordingly)
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]


lambdas =[0,0.01,0.1,1,10,100,1000]
#Use different values of lambdas from the above mentioned array

#To store Validation errors (Square error) for different values of lambda
validationErrorSet = list()

#Here write a function that takes as input a lambda value and outputs the optimal weight vector
#def calc_weights()...

#Here write a function that takes as input a weight vector and outputs the squared-error loss (Validation error)
#def calc_val_err()...



#Note: You will have to calculate cross validation on your own, libraries are not allowed
#For each subset of k_fold crosss validation create your training and testing set
	#For each value of lambda calculate the optimal weights and square error using the functions create above, 
	#add the squared error to ValidationErrorSet, you can write this whole calculation with whichever way suitable to you 




#Find the best value of Error from Validation error set

#lmb = "Best Lambda: "+best_lambda
#error = "Error at Best Lambda: %.4f"%best_val_error

# Produce a plot of results.
#Change the details below as per your need
plt.semilogx(lambdas, validationError,label='Validation error')
plt.semilogx(best_lambda,best_val_error,marker='o',color='r',label="Best Labmda")
plt.ylabel('Sum Squared Error')
plt.text(5, 116, lmb, fontsize=15)
plt.text(5, 109, error, fontsize=15)
plt.legend()
plt.xlabel('Lambda')
plt.show()

#!/usr/bin/env python

# Run logistic regression training for different learning rates with stochastic gradient descent.


import scipy.special as sps
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from itertools import combinations, permutations
from numpy.linalg import inv
from collections import defaultdict
import math

np.seterr(all='print')

dummy_feature_List = ['country_group_CAN', 'country_group_EURO', 'country_group_USA', 'Position_C',
                      'Position_D', 'Position_L', 'Position_R']

# Preprocessing functions
def standardize(s_df):
    for col in s_df.columns.values:
        col_mean = s_df[col].mean()
        col_std = s_df[col].std()

        #         mean of interaction terms of two discrete variable are zero. those columns are
        #         filtered while standizing as it cause singular matrix
        if (col_mean != 0 or col_std != 0) or col not in dummy_feature_List:
            s_df[col] = s_df[col].apply(lambda x: (x - col_mean) / float(col_std))

    return s_df


def preprocessing(x_df):
    x_df = pd.get_dummies(x_df, prefix=['country_group', 'Position'], columns=['country_group', 'Position'])
    x_df = x_df.apply(pd.to_numeric, args=('coerce',))

    # # # adding interaction terms
    for col_1, col_2 in combinations(x_df.columns, 2):
        cond1 = col_1 not in dummy_feature_List
        cond2 = col_2 not in dummy_feature_List
        if cond1 or cond2:
            x_df['{}*{}'.format(col_1, col_2)] = np.multiply(x_df[col_1], x_df[col_2])
            # x_df.to_csv('ex.csv', sep='\t')

    #         #     standardization
    x_df = standardize(x_df)
    #     y_df = standardize(y_df)

    #     deleteing column = 0
    x_df = x_df.loc[:, (x_df != 0).any(axis=0)]
    x_df.insert(loc=0, column='x0', value=1)

    return x_df


def calculate_negative_log_likeli(target, weights, features):
    z = np.dot(features, weights)
    y = sps.expit(z) + 0.1E-8

    likelihood = np.mean(np.sum((1 - target) * z + np.log(y)))

    return likelihood

def calculate_accuracy(target, weights, features):
    z = np.dot(features, weights)
    y = sps.expit(z)

    total_count = target.size
    total_match = 0
    print "total Dataset Count => {}".format(total_count)

    for n in range(0, total_count):
        predicted = 1 if y[n] >= 0.5 else 0
        if target[n] == predicted:
            total_match += 1

    print "Correctly predicted Count => {}".format(total_match)
    return (total_match / float(total_count)) * 100


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_dataset():
    input = os.path.join("..", 'Model_Trees_Full_Dataset',
                         'preprocessed_datasets.csv')
    data = pd.read_csv(input)

    # random shuffle
    data = data.iloc[np.random.permutation(len(data))]
    data[u'GP_greater_than_0'] = data[u'GP_greater_than_0'].map({'yes': 1, 'no': 0})
    # data[u'GP_greater_than_0'] = data[u'GP_greater_than_0'].map({'yes': 1, 'no': 0})
    print data.head(5)
    training_df = data[data[u'DraftYear'].isin([2004, 2005, 2006])]
    testing_df = data[data[u'DraftYear'] == 2007]

    drop_class = [u'id', u'Country', u'Overall', u'PlayerName', u'sum_7yr_TOI', u'DraftYear', u'sum_7yr_GP']
    training_df.drop(drop_class, inplace=True, axis=1)
    testing_df.drop(drop_class, inplace=True, axis=1)

    y_train_df = training_df.filter(["GP_greater_than_0"])
    x_train_df = training_df.drop(["GP_greater_than_0"], axis=1)

    y_test_df = testing_df.filter(["GP_greater_than_0"])
    x_test_df = testing_df.drop(["GP_greater_than_0"], axis=1)

    x_train_df_processed = preprocessing(x_train_df)
    x_test_df_processed = preprocessing(x_test_df)

    # print x_train_df_processed.head()
    # print y_train_df.head()

    y_train = y_train_df.values
    x_train = x_train_df_processed.values
    y_test = y_test_df.values
    x_test = x_test_df_processed.values

    return (x_train, y_train), (x_test, y_test)

max_iter = 4
tol = 0.00001

# Step size for gradient descent.
etas = [0.5, 0.3, 0.1, 0.05, 0.01, 0.001]
# etas = [0.1, 0.05, 0.01]

(X, t), (X_test, t_test) = load_dataset()

# # # For plotting data
# X1 = df[df[u'GP_greater_than_0'] == 0]
# X2 = df[df[u'GP_greater_than_0'] == 1]
# X1 = X1.drop([u'GP_greater_than_0'], axis=1)
# X2 = X2.drop([u'GP_greater_than_0'], axis=1)

n_train = t.size
feature_size = X.shape[1] -1
# Error values over all iterations.
all_errors = dict()

for eta in etas:
    # Initialize w.
    w = np.array([0.1] + feature_size * [0.0], dtype=np.float32)
    e_all = []

    for iter in range(0, max_iter):
        for n in range(0, n_train):
            # Compute output using current w on sample x_n.
            y = sps.expit(np.dot(X[n, :], w))
            actual = t[n]
            # Gradient of the error, using Assignment result
            grad_e = np.multiply((y - t[n]), X[n, :])

            # Update w, *subtracting* a step in the error derivative since we're minimizing
            w = np.subtract(w, (eta * grad_e))
            # print grad_e
        # Compute error over all examples, add this error to the end of error vector.
        # Compute output using current w on all data X.
        y = sps.expit(np.dot(X, w))

        # e is the error, negative log-likelihood (Eqn 4.90)
        # e = calculate_negative_log_likelihood(t,w,X)
        # e = - np.mean(np.dot(np.log(y + 1E-08),t) + np.dot(np.log(1 - y + 1E-08),(1 - t)))
        e = -np.mean(np.multiply(t, np.log(y + 1E-08)) + np.multiply((1 - t), np.log(1 - y + 1E-08)))
        e_all.append(e)

        # Print some information.
        # print 'eta={0}, epoch {1:d}, negative log-likelihood {2:.4f}, w={3}'.format(eta, iter, e, w.T)
        # print 'eta={0}, epoch {1:d}, negative log-likelihood {2:.4f}'.format(eta, iter, e)

        # Stop iterating if error doesn't change more than tol.
        if iter > 0:
            if np.absolute(e - e_all[iter - 1]) < tol:
                break

    all_errors[eta] = e_all
    print "Final Accuracy ==> ",calculate_accuracy(t_test, w, X_test)
# Plot error over iterations for all etas
plt.figure(10)
plt.rcParams.update({'font.size': 15})
for eta in sorted(all_errors):
    plt.plot(all_errors[eta], label='sgd eta={}'.format(eta))

plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression with SGD')
plt.xlabel('Epoch')


plt.legend()
plt.show()
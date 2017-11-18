import scipy.special as sps
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from itertools import combinations, permutations
from numpy.linalg import inv
from collections import defaultdict
import math
import time

dummy_feature_List = ['country_group_CAN','country_group_EURO', 'country_group_USA', 'Position_C',
       'Position_D', 'Position_L', 'Position_R']
def standardize(s_df):
    for col in s_df.columns.values:
        col_mean = s_df[col].mean()
        col_std = s_df[col].std()

        #         mean of interaction terms of two discrete variable are zero. those columns are
        #         filtered while standizing as it cause singular matrix
        if (col_mean != 0 or col_std != 0) or col not in dummy_feature_List :
            s_df[col] = s_df[col].apply(lambda x: (x - col_mean) / float(col_std))

    return s_df

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

def load_dataset():
    df = pd.read_csv("../Model_Trees_Full_Dataset/normalized_datasets.csv")
    # df = pd.read_csv("../Model_Trees_Full_Dataset/preprocessed_datasets.csv")
    df = df.iloc[np.random.permutation(len(df))]
    # drop_columns = [u'id', u'PlayerName', u'sum_7yr_TOI', u'DraftYear',u'Country', u'Overall', u'sum_7yr_GP']
    drop_columns = [u'id', u'PlayerName', u'sum_7yr_TOI', u'DraftYear', u'Overall', u'sum_7yr_GP']
    x_df = pd.get_dummies(df, prefix=['country_group', 'Position'], columns=['country_group', 'Position'])
    training_df = x_df[x_df[u'DraftYear'].isin([2004, 2005, 2006])]
    testing_df = x_df[x_df[u'DraftYear'] == 2007]

    dropped_train = training_df.drop(drop_columns, axis=1, )
    dropped_test = testing_df.drop(drop_columns, axis=1, )

    y_train_df = dropped_train.filter([u'GP_greater_than_0'])
    y_train_df = y_train_df.replace(['yes', 'no'], [1, 0])
    x_train_df = dropped_train.drop([u'GP_greater_than_0'], axis=1)
    y_train_df = y_train_df.apply(pd.to_numeric, args=('coerce',))

    y_test_df = dropped_test.filter([u'GP_greater_than_0'])
    y_test_df = y_test_df.replace(['yes', 'no'], [1, 0])
    x_test_df = dropped_test.drop([u'GP_greater_than_0'], axis=1)

    # x_train_df = standardize(x_train_df)
    # x_test_df = standardize(x_test_df)

    y_train = y_train_df.values
    x_train = x_train_df.values
    y_test = y_test_df.values
    x_test = x_test_df.values

    return (x_train, y_train), (x_test, y_test)

def log_likelihood(target, weights, features, predicted):
    xw = np.dot(features, weights)
    ll = np.mean(np.dot(np.transpose(1-target), xw) - np.log(predicted + 1E-08))
    return ll

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


max_iter = 1000
tol = 0.00001

# Step size for gradient descent.
etas = [0.5, 0.3, 0.1, 0.05, 0.01]
# etas = [0.1, 0.001, 0.01,0.0001,0.05]
#etas = [0.1, 0.05, 0.01]

(x_train,y_train), (x_test, y_test) = load_dataset()
n_train = y_train.shape[0]
feature_size = x_train.shape[1] -1
print x_train.shape
print y_train.shape
print x_test.shape
print y_test.shape

# Error values over all iterations.
all_errors = dict()

for eta in etas:
    # Initialize w.
    w = np.array([0.1] + feature_size * [0.0], dtype=np.float32)
    e_all = []

    for iter in range(0, max_iter):
        # x_train, y_train = unison_shuffled_copies(x_train, y_train)
        for n in range(0, n_train):
            # Compute output using current w on sample x_n.
            y = sps.expit(np.dot(x_train[n, :], w))

            # Gradient of the error, using Assignment result
            grad_e = (y - y_train[n]) * x_train[n, :]

            # Update w, *subtracting* a step in the error derivative since we're minimizing
            # w = fill this in
            w = w - (eta * grad_e)

        # Compute error over all examples, add this error to the end of error vector.
        # Compute output using current w on all data X.
        y = sps.expit(np.dot(x_train, w))

        # e is the error, negative log-likelihood (Eqn 4.90)
        #         e = log_likelihood(y_train, w, x_train)
        e = log_likelihood(y_train, w, x_train,y)
        # e = - np.mean(np.multiply(y_train, np.log(y + 1E-08)) + np.multiply((1 - y_train), np.log(1 - y + 1E-08)))
        #         e = -np.mean(np.multiply(y_train,np.log(y)) + np.multiply((1-y_train),np.log(1-y)))
        e_all.append(e)

        # Print some information.
        #         print 'eta={0}, epoch {1:d}, negative log-likelihood {2:.4f}'.format(eta, iter, e, w.T)
        print 'eta={0}, epoch {1:d}, negative log-likelihood {2:.4f}'.format(eta, iter, e)

        # Stop iterating if error doesn't change more than tol.
        if iter > 0:
            if np.absolute(e - e_all[iter - 1]) < tol:
                break

    all_errors[eta] = e_all
    print "Final Accuracy ==> ", calculate_accuracy(y_test, w, x_test)

# Plot error over iterations for all etas
plt.figure(10)
plt.rcParams.update({'font.size': 15})
for eta in sorted(all_errors):
    plt.plot(all_errors[eta], label='sgd eta={}'.format(eta))

plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression with SGD')
plt.xlabel('Epoch')
# plt.axis([0, max_iter, 0.2, 0.7])
plt.legend()
plt.show()

import math
from collections import OrderedDict, defaultdict
import sys, codecs, optparse, os
import pandas as pd
import numpy as np
from itertools import combinations, permutations
from numpy.linalg import inv


# finding regularized weights for given X, Y and lambda(reqularised parameter)
def get_weights(lam_da, x, y):
    i = np.identity(feature_count, dtype=None)
    i[0][0] = 0
    i_lambda = np.multiply(lam_da, i)
    x_trans = np.transpose(x)
    w_first_exp = np.add(i_lambda, np.matmul(x_trans, x))
    w_sec_exp = np.matmul(x_trans, y)

    w = np.matmul(inv(w_first_exp), w_sec_exp)
    return w


def get_LSE(lam_da, x, y, test_x, test_y):
    w = get_weights(lam_da, x, y)
    w_trans = np.transpose(w)
    w_trans_x = np.multiply(w_trans, test_x)
    w_trans_x.shape

    pred_y = np.sum(w_trans_x, axis=1)

    square_error = 0
    weight_sum = 0
    for j in range(w.shape[0]):
        weight_sum += np.power(w[j], 2)
    weight_sum = np.multiply(lam_da, weight_sum)
    for i in range(pred_y.shape[0]):
        square_error += np.power((test_y[i] - pred_y[i]), 2)

    average_square_error = np.divide(square_error, pred_y.shape[0]) + weight_sum
    print "lambda = ", lam_da, "---> Error = ", average_square_error


def standardize(s_df):
    for col in s_df.columns:
        if col not in ["country_group", "Position"]:
            col_mean = s_df[col].mean()
            col_std = s_df[col].std()

            s_df[col] = s_df[col].apply(lambda x: (x - col_mean) / float(col_std))

    return s_df


input = os.path.join('Model_Trees_Full_Dataset', 'preprocessed_datasets.csv')
input_df = pd.read_csv(input)

# lambdas = [0, 0.01, 0.1, 1, 10, 100, 1000]
lambdas = [0.01, 0.1]

df = input_df[input_df[u'DraftYear'].isin([2004, 2005, 2006])]
testing_df = input_df[input_df[u'DraftYear'] == 2007]

drop_class = [u'id', u'Country', u'Overall', u'PlayerName', u'sum_7yr_TOI', u'DraftYear', u'GP_greater_than_0']
df.drop(drop_class, inplace=True, axis=1)

df = pd.get_dummies(df, prefix=['country_group', 'Position'], columns=['country_group', 'Position'])

y_df = df.filter(["sum_7yr_GP"], axis=1)
x_df = df.drop(["sum_7yr_GP"], axis=1)
y_test = testing_df.filter(["sum_7yr_GP"], axis=1)
x_test = testing_df.drop(["sum_7yr_GP"], axis=1)

# # adding interaction terms
# for col_1, col_2 in combinations(x_df.columns,2):

# # #     cond1 = col_1 not in ["country_group","Position"]
# # #     cond2 = col_2 not in ["country_group","Position"]
# # #     if cond1 and cond2:
#     x_df['{}*{}'.format(col_1, col_2)] = x_df[col_1] * x_df[col_2]

# standardization
print x_df.columns
x_df = standardize(x_df)
y_df = standardize(y_df)

x_df.insert(loc=0, column='x0', value=1)

feature_count = len(x_df.columns.tolist())
train_set_count = x_df.shape[0]
test_size = int(train_set_count / 10)

x = x_df.as_matrix()
y = y_df.as_matrix()

k_fold = 1

while k_fold <= 10:
    print "Fold ------------->", k_fold
    test_st = (k_fold - 1) * test_size
    test_en = k_fold * test_size

    train_x = x[: test_st]
    test_x = x[test_st:test_en]
    train_x = np.concatenate((train_x, x[test_en:]), axis=0)

    train_y = y[: test_st]
    test_y = y[test_st:test_en]
    train_y = np.concatenate((train_y, y[test_en:]), axis=0)
    k_fold += 1

    for i in lambdas:
        get_LSE(i, train_x, train_y, test_x, test_y)
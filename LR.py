import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from itertools import combinations, permutations
from numpy.linalg import inv


# finding regularized weights for given X, Y and lambda(reqularised parameter)
def calc_weights(lam_da, x, y):
    i = np.identity(feature_count, dtype=None)
    i_lambda = np.multiply(lam_da, i)
    x_trans = np.transpose(x)
    w_first_exp = np.add(i_lambda, np.matmul(x_trans, x))
    w_sec_exp = np.matmul(x_trans, y)

    w = np.matmul(inv(w_first_exp), w_sec_exp)
    return w


def train(lam_da, x, y, t_x, t_y):
    w = calc_weights(lam_da, x, y)
    average_square_error = calc_val_err(w, lam_da, t_x, t_y)

    #     print "lambda = ", lam_da, "---> Error = ", average_square_error
    return average_square_error, w


def calc_val_err(w, lam_da, test_x, test_y):
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

    average_square_error = (square_error[0] / float(pred_y.shape[0])) + weight_sum

    return average_square_error


def standardize(s_df):
    for col in s_df.columns.values:
        col_mean = s_df[col].mean()
        col_std = s_df[col].std()

        #         print col,"---",col_mean,"----",col_std
        #         print col_mean > 0 or col_std > 0
        if (col_mean > 0 or col_std > 0):
            s_df[col] = s_df[col].apply(lambda x: (x - col_mean) / float(col_std))

    return s_df


def preprocessing(x_df, y_df):
    x_df = pd.get_dummies(x_df, prefix=['country_group', 'Position'], columns=['country_group', 'Position'])
    x_df = x_df.apply(pd.to_numeric, args=('coerce',))

    # # adding interaction terms
    for col_1, col_2 in combinations(x_df.columns, 2):
        x_df['{}*{}'.format(col_1, col_2)] = np.multiply(x_df[col_1], x_df[col_2])
    # x_df.to_csv('ex.csv', sep='\t')

    # standardization
    x_df = standardize(x_df)
    # y_df = standardize(y_df)

    #     deleteing column = 0
    x_df = x_df.loc[:, (x_df != 0).any(axis=0)]
    x_df.insert(loc=0, column='x0', value=1)

    return x_df, y_df


# Load your dataset here
input = os.path.join('Model_Trees_Full_Dataset', 'preprocessed_datasets.csv')


input_df = pd.read_csv(input)

lambdas = [0, 0.01, 0.1, 1, 10, 100, 1000]
# lambdas = [0.01, 0.1]

df = input_df[input_df[u'DraftYear'].isin([2004, 2005, 2006])]
testing_df = input_df[input_df[u'DraftYear'] == 2007]

df.to_csv("training")
testing_df.to_csv("test")

drop_class = [u'id', u'Country', u'Overall', u'PlayerName', u'sum_7yr_TOI', u'DraftYear', u'GP_greater_than_0']
df.drop(drop_class, inplace=True, axis=1)
testing_df.drop(drop_class, inplace=True, axis=1)

y_df = df.filter(["sum_7yr_GP"], axis=1)
x_df = df.drop(["sum_7yr_GP"], axis=1)
y_test = testing_df.filter(["sum_7yr_GP"], axis=1)
x_test = testing_df.drop(["sum_7yr_GP"], axis=1)

x_df, y_df = preprocessing(x_df, y_df)

feature_count = len(x_df.columns.tolist())
train_set_count = x_df.shape[0]
test_size = int(train_set_count / 10)

x = x_df.as_matrix()
y = y_df.as_matrix()

k_fold = 1

best_weight_vector = {}
for i in lambdas:
    best_weight_vector[i] = {}

while k_fold <= 10:
    test_st = (k_fold - 1) * test_size
    test_en = k_fold * test_size

    train_x = x[: test_st]
    test_x = x[test_st:test_en]
    train_x = np.concatenate((train_x, x[test_en:]), axis=0)

    train_y = y[: test_st]
    test_y = y[test_st:test_en]
    train_y = np.concatenate((train_y, y[test_en:]), axis=0)
    k_fold += 1

    validationErrorSet = {}
    validationErrorSet[k_fold] = {}
    # best weight vector with respect to lambda comparing LSE from 10-fold validation

    for i in lambdas:
        validationErrorSet[k_fold][i], w = train(i, train_x, train_y, test_x, test_y)
        if 'LSE' not in best_weight_vector[i]:
            best_weight_vector[i]['LSE'] = validationErrorSet[k_fold][i]
            best_weight_vector[i]['K_FOLD'] = k_fold
            best_weight_vector[i]['w'] = w

        elif best_weight_vector[i]['LSE'] > validationErrorSet[k_fold][i]:
            best_weight_vector[i]['LSE'] = validationErrorSet[k_fold][i]
            best_weight_vector[i]['K_FOLD'] = k_fold
            best_weight_vector[i]['w'] = w

print best_weight_vector

x_test_df, y_test_df = preprocessing(x_test, y_test)

x_test = x_test_df.as_matrix()
y_test = y_test_df.as_matrix()

test_result = {}
# Find the best value of Error from Validation error set
for k, v in best_weight_vector.iteritems():
    print "Lambda =", k, " Fold =", v['K_FOLD']
    print "Error while training ==> ------------------------->", v['LSE']
    test_result[k] = calc_val_err(v['w'], k, x_test, y_test)
    print "Error while Testing ==> ------------------------->", test_result[k]

best_val_error = min(test_result.itervalues())
best_lambda = [k for k, v in test_result.iteritems() if v == best_val_error]

lmb = '{}:{}'.format("Best Lambda", best_lambda)
error = '{}:{}'.format("Error at Best Lambda ", best_val_error)

# Produce a plot of results.


# plt.semilogx(lambdas, test_result.itervalues(), label='Validation error')
# plt.semilogx(best_lambda, best_val_error, marker='o', color='r', label="Best Labmda")
# plt.ylabel('Sum Squared Error')
# plt.text(5, 116, lmb, fontsize=15)
# plt.text(5, 109, error, fontsize=15)
# plt.legend()
# plt.xlabel('Lambda')
# plt.show()

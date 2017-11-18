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
import keras

from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import losses

np.seterr(all='print')

def load_dataset():
    df = pd.read_csv("./Least-Square-Regression/Model_Trees_Full_Dataset/preprocessed_datasets.csv")
    drop_columns = [u'id', u'PlayerName',u'sum_7yr_TOI',u'DraftYear',u'Country',u'Overall', u'GP_greater_than_0']
    x_df = pd.get_dummies(df, prefix=['country_group', 'Position'], columns=['country_group', 'Position'])
    
    training_df = x_df[x_df[u'DraftYear'].isin([2004, 2005, 2006])]
    testing_df = x_df[x_df[u'DraftYear'] == 2007]
    
    dropped_train = training_df.drop(drop_columns, axis =1, )
    dropped_test = testing_df.drop(drop_columns, axis =1, )
    
    y_train_df = dropped_train.filter([u'sum_7yr_GP'])
    x_train_df = dropped_train.drop([u'sum_7yr_GP'], axis=1)

    y_test_df = dropped_test.filter([u'sum_7yr_GP'])
    x_test_df = dropped_test.drop([u'sum_7yr_GP'], axis=1)
    
    # print x_train_df.columns
    y_train = y_train_df.values
    x_train = x_train_df.values
    y_test = y_test_df.values
    x_test = x_test_df.values
    
    return (x_train,y_train), (x_test, y_test)


(x_train,y_train), (x_test, y_test) = load_dataset()
# print x_train.shape
# print y_train.shape
# print x_test.shape
# print y_test.shape



model = Sequential()
model.add(Dense(units=100, input_dim=22, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=1, activation='linear'))
model.compile(loss=losses.MSE, optimizer=keras.optimizers.Adagrad(lr=0.1))
history = model.fit(x_train, y_train,validation_data=(x_test,y_test), epochs=50, batch_size=10)

loss = model.evaluate(x_test, y_test,batch_size=1, verbose=1)
print("\n[INFO] loss={:.4f}".format(loss))
# list all data in history
# print(history.history.keys())
plt.plot(history.history['loss'][1:])
plt.plot(history.history['val_loss'][1:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

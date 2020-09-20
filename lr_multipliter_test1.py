# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 10:35:32 2020

@author: a
"""


 
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import keras.backend as K
from keras.optimizers import Adam
 
# create some data   创建散点图数据
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
# plot data
plt.scatter(X, Y)
plt.show()
 
X_train, Y_train = X[:160], Y[:160]     # first 160 data points
X_test, Y_test = X[160:], Y[160:]       # last 40 data points
 


def our_get_gradients(loss, params):
    for p in params:
        print(params[0].name)
    return [K.zeros_like(p) for p in params]



if __name__ == '__main__':
    # build a neural network from the 1st layer to the last layer
    model = Sequential()
     
    model.add(Dense(units=1, input_dim=1,name = 'myhahaha1')) 
    model.add(Dense(units=1, input_dim=1,name = 'myhahaha2')) 
    # choose loss function and optimizing method
    adam_opt = Adam(1e-3)
    adam_opt.get_gradients = our_get_gradients
    
    model.compile(loss='mse',
                  optimizer=adam_opt)
    # training
    print('Training -----------')
    for step in range(301):
        cost = model.train_on_batch(X_train, Y_train)
        if step % 100 == 0:
            print('train cost: ', cost)
     
    # test
    print('\nTesting ------------')
    cost = model.evaluate(X_test, Y_test, batch_size=40)
    print('test cost:', cost)
    W, b = model.layers[0].get_weights()
    print('Weights=', W, '\nbiases=', b)
     
    # plotting the prediction
    Y_pred = model.predict(X_test)
    plt.scatter(X_test, Y_test)
    plt.plot(X_test, Y_pred)
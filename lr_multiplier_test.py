# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 10:07:25 2020

@author: a
"""

from keras.models import Sequential
from keras.layers import Dense
from keras_lr_multiplier import LRMultiplier

model = Sequential()
model.add(Dense(
    units=5,
    input_shape=(5,),
    activation='tanh',
    name='Dense',
))
model.add(Dense(
    units=2,
    activation='softmax',
    name='Output',
))
model.compile(
    optimizer=LRMultiplier('adam', {'Dense': 0.5, 'Output': 1.5}),
    loss='sparse_categorical_crossentropy',
)

if __name__ == '__main__':
    pass
    
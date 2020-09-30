# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 07:31:03 2020

@author: tunan
"""
from bert4keras.models import build_transformer_model
from keras.layers import Dense,TimeDistributed,Bidirectional,LSTM,Dropout
from keras.models import Model
from bert4keras.optimizers import Adam
from util import logfile,_init,set_value,get_value

def build_model():
    '''加载预训练模型，重新定义模型结构。
        从bert预训练模型中截取最后一层，接上dense+softmax。
        返回值：None。model是全局变量，直接修改覆盖。
    '''
    config_path,checkpoint_path,bert_layers,num_labels,learning_rate = get_value('config_path'),get_value('checkpoint_path'),\
        get_value('bert_layers'),get_value('num_labels'),get_value('learning_rate')
    #加载模型
    model = build_transformer_model(
        config_path,
        checkpoint_path,
    )
    #模型最后一个transformer输出层的名称。
    output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
    output = model.get_layer(output_layer).output #得到bert最后一个transformer输出的向量，大小为768.
    output = Dense(num_labels, activation="softmax")(output)
    model = Model(model.input, output) #根据输入输出生成模型。
    # model.summary()
    
    #模型损失使用CRF.sparse_loss,adam学习器，CRF的离散准确率。
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate),
        metrics=['accuracy']
    )
    return model

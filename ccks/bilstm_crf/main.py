# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 20:59:04 2020

@author: tunan
"""

import json,os
import numpy as np
from keras import backend as K
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy, crf_viterbi_accuracy
from keras.models import Model, Input
from keras.layers import Dense, Bidirectional, Dropout, LSTM, TimeDistributed, Masking,Embedding
from keras.utils import to_categorical, plot_model
import matplotlib.pyplot as plt
from tqdm import tqdm

from bert4keras.snippets import sequence_padding, DataGenerator
from keras.utils.np_utils import to_categorical
from bert4keras.tokenizers import Tokenizer
from sklearn.model_selection import KFold
from keras.optimizers import Adam

from mypredict_model import Evaluator,evaluate
from mypreprocess_data import data_generator,load_data,my_initialize
from myutil import _init,set_value,get_value
from sgdr_implementation import LR_Cycle

# Build model
def build_model(vocab_size, n_tags):
    learning_rate = get_value('learning_rate')
    # Bert Embeddings
    inputs = Input(shape=(None, ), name="bert_output")
    x = Embedding(vocab_size+1, 256, embeddings_initializer='uniform')(inputs)
    # LSTM model
    lstm = Bidirectional(LSTM(units=256, return_sequences=True), name="bi_lstm")(x)
    drop = Dropout(0.1, name="dropout")(lstm)
    dense = TimeDistributed(Dense(n_tags, activation="softmax"), name="time_distributed")(drop)
    crf = CRF(n_tags)
    out = crf(dense)
    model = Model(inputs=inputs, outputs=out)
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(loss=crf.loss_function, optimizer=Adam(learning_rate), metrics=[crf.accuracy])

    # 模型结构总结
    model.summary()
    return model

def train_single_model():
    '''给定模型参数后，生成并训练模型。
    返回：该模型在test_data验证集上的f1,precision,recall.
    '''
    K.clear_session()  #训练开始前清除显存
    epochs,batch_size,modeldata_path = get_value('epochs'),get_value('batch_size'),get_value('modeldata_path')
    test_data = get_value('test_data')
    train_data = get_value('train_data')
    token2id,num_labels = get_value('token2id'),get_value('num_labels')
    global model
    model = build_model(len(token2id),num_labels)    #定义模型结构
    evaluator = Evaluator() 
    train_generator = data_generator(train_data, batch_size)
    
    sched = LR_Cycle(iterations = len(train_generator),cycle_mult = 2)
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[evaluator]
    )
        #学习率退火，训练10个epoch
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator,sched]
    )
    model.load_weights(os.path.join(modeldata_path,r'model/origin_model.weights'))   #加载模型预测，返回预测值。
    f1, precision, recall = evaluate(test_data,model)
    return f1,precision, recall

def model_train():
    '''
    加载数据集，使用5折交叉验证，返回每个模型在测试集上的f1。
    all_f1(list(float)):5个数据集上的f1值。
    '''    
    data = []           #读取数据集
    data1_path =  r'..\datasets\yidu-s4k\subtask1_training_part1.txt'
    data2_path =  r'..\datasets\yidu-s4k\subtask1_training_part1.txt'
    test_data_path = r'..\datasets\yidu-s4k\subtask1_test_set_with_answer.json'
    with open(data1_path,'r',encoding='gbk') as f:
        data.extend(f.readlines())
    with open(data2_path,'r',encoding='gbk') as f:
        data.extend(f.readlines())
    test_data = []
    with open(test_data_path,'r',encoding='gbk') as f:
        test_data.extend(f.readlines())        

    X = load_data(data)     #分割数据集
    test_data = load_data(test_data) 
    set_value('test_data',test_data)
    kf = KFold(n_splits=5,shuffle = True)
    all_f1 = []
    
    for train_index, val_index in kf.split(X):
        train_data = list(np.array(X)[train_index])
        valid_data = list(np.array(X)[val_index])   
        set_value('train_data', train_data)
        set_value('valid_data',valid_data)
        f1,_,_ = train_single_model()   #训练模型
        all_f1.append(f1)
    return all_f1

_init()
tokenizer = Tokenizer(get_value('dict_path'), do_lower_case=True)
labels = ['疾病和诊断', '检查', '检验','手术','药物','解剖部位'] 
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 2 + 1

set_value('labels',labels) 
set_value('id2label',id2label)
set_value('label2id',label2id)
set_value('num_labels',num_labels)
set_value('tokenizer',tokenizer)
set_value('learning_rate', 0.01)  #设置为最优的学习率
set_value('epochs',100)    
my_initialize()  #设置token2id变量
    
if __name__ == '__main__':  
    f1 = model_train()
    f1 = np.array(f1)
    line = "bilstm-crf,{:.6f},{:.6f},{:.6f}\n".format(f1.mean(),f1.min(),f1.max())
    print(line) 
    open("result_log.txt",'a').write(line)
    #应该是在训练之前把embedding装入内存中出错，装入的taken2id。
    

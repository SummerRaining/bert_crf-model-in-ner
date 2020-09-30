# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:21:24 2020

@author: a
"""
import warnings
warnings.filterwarnings("ignore")  #忽略python的warning。
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import json,os,random
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer

from bert4keras.layers import ConditionalRandomField
from tqdm import tqdm
from keras_lr_multiplier import LRMultiplier
from sklearn.model_selection import KFold
from util import Logger,logfile
logfile('./log.txt')            #log文件

def train_single_model():
    '''给定模型参数后，生成并训练模型。
    返回：该模型在test_data验证集上的f1,precision,recall.
    '''
    from keras import backend as K
    K.clear_session()  #训练开始前清除显存
    global model
    build_model()    #定义模型结构
    #定义评价器，训练模型
    evaluator = Evaluator() 
    train_generator = data_generator(train_data, batch_size)
    
    from sgdr_implementation import LR_Cycle
    sched = LR_Cycle(iterations = len(train_generator),cycle_mult = 2)
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
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
    f1, precision, recall = evaluate(test_data)
    return f1,precision, recall
    
def model_train():
    '''
    加载数据集，使用5折交叉验证，返回每个模型在测试集上的f1。
    all_f1(list(float)):5个数据集上的f1值。
    '''
    global valid_data,test_data,train_data,id2label,tokenizer,label2id,num_labels
    
    data = []           #读取数据集
    data1_path =  r'.\datasets\yidu-s4k\subtask1_training_part1.txt'
    data2_path =  r'.\datasets\yidu-s4k\subtask1_training_part1.txt'
    test_data_path = r'.\datasets\yidu-s4k\subtask1_test_set_with_answer.json'
    with open(data1_path,'r',encoding='gbk') as f:
        data.extend(f.readlines())
    with open(data2_path,'r',encoding='gbk') as f:
        data.extend(f.readlines())
    test_data = []
    with open(test_data_path,'r',encoding='gbk') as f:
        test_data.extend(f.readlines())        

    X = load_data(data)     #分割数据集
    test_data = load_data(test_data)
    kf = KFold(n_splits=5,shuffle = True)
    all_f1 = []
    for train_index, val_index in kf.split(X):
        train_data = list(np.array(X)[train_index])
        valid_data = list(np.array(X)[val_index])    
        f1,_,_ = train_single_model()   #训练模型
        all_f1.append(f1)
    return all_f1

if __name__ == '__main__':   #if不改变变量的作用域
    #定义
    maxlen = 512
    epochs = 10
    batch_size = 8
    bert_layers = 12
    learing_rate = 1e-5  # bert_layers越小，学习率应该要越大
    crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率
    
    modeldata_path = r'C:\Users\tunan\model_data\bert4keras'
    # bert配置
    config_path = os.path.join(modeldata_path,r'chinese_L-12_H-768_A-12\bert_config.json')
    checkpoint_path = os.path.join(modeldata_path,r'chinese_L-12_H-768_A-12\bert_model.ckpt')
    dict_path = os.path.join(modeldata_path,r'chinese_L-12_H-768_A-12\vocab.txt')
    
    f1 = model_train()
    print(np.array(f1).mean(),np.array(f1).std())


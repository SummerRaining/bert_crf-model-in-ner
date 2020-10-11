# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:30:26 2020
@author: tunan
"""
import warnings
warnings.filterwarnings("ignore")  #忽略python的warning。
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from keras import backend as K
import json,os,random
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.layers import ConditionalRandomField
from tqdm import tqdm
from keras_lr_multiplier import LRMultiplier
from sklearn.model_selection import KFold
from sgdr_implementation import LR_Cycle
#加载自定义文件
from util import logfile,_init,set_value,get_value
_init()
from preprocess_data import load_data,data_generator
from build_model import build_model
from predict_model import Evaluator,evaluate


logfile('./log.txt')            #log文件

def train_single_model():
    '''给定模型参数后，生成并训练模型。
    返回：该模型在test_data验证集上的f1,precision,recall.
    '''
    K.clear_session()  #训练开始前清除显存
    epochs,batch_size,modeldata_path = get_value('epochs'),get_value('batch_size'),get_value('modeldata_path')
    test_data = get_value('test_data')
    train_data = get_value('train_data')
    global model
    model = build_model()    #定义模型结构
    evaluator = Evaluator() 
    train_generator = data_generator(train_data, batch_size)

    sched = LR_Cycle(iterations = len(train_generator),cycle_mult = 2)
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=5,
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
  
def hyperparameter_tune(model_name,path):
    modeldata_path = get_value('modeldata_path')
    set_value('config_path',os.path.join(modeldata_path,r'{}\bert_config.json'.format(path)))
    set_value('checkpoint_path',os.path.join(modeldata_path,r'{}\bert_model.ckpt'.format(path)))
    set_value('dict_path',os.path.join(modeldata_path,r'{}\vocab.txt'.format(path)))
    f1 = model_train()
    f1 = np.array(f1)
    line = "{} model,{:.6f},{:.6f},{:.6f}\n".format(model_name,f1.mean(),f1.min(),f1.max())
    print(line)
    open("result_log.txt",'a').write(line)
    
def hyperparameter_tune_lr(lr):
    set_value('learning_rate', lr)
    f1 = model_train()
    f1 = np.array(f1)
    line = "{} learning rate ,{:.6f},{:.6f},{:.6f}\n".format(lr,f1.mean(),f1.min(),f1.max())
    print(line)
    open("result_log.txt",'a').write(line)
    
    

if __name__ == '__main__':   #if不改变变量的作用域
    # 建立分词器,do_lower_case:只包含小写字母，大写字母作为unk token处理。
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
    set_value('learning_rate', 9e-05)  #设置为最优的学习率
    modeldata_path = get_value('modeldata_path')  #设置最优的预训练模型
    path = 'chinese_roberta_wwm_ext_L-12_H-768_A-12'
    set_value('config_path',os.path.join(modeldata_path,r'{}\bert_config.json'.format(path)))
    set_value('checkpoint_path',os.path.join(modeldata_path,r'{}\bert_model.ckpt'.format(path)))
    set_value('dict_path',os.path.join(modeldata_path,r'{}\vocab.txt'.format(path)))
    
    set_value('epochs',100)  #
    f1 = model_train()
    f1 = np.array(f1)
    line = "roberta_wwm model with lr 9e-5,{:.6f},{:.6f},{:.6f}\n".format(f1.mean(),f1.min(),f1.max())
    print(line) 
    open("result_log.txt",'a').write(line)
    
# =============================================================================
#     #尝试不同的学习率
#     for lr in [3e-5,3e-6,9e-5,2e-4]:
#         hyperparameter_tune_lr(lr)
# 
#     #尝试不同的模型
#     model_path = {'base_bert':'chinese_L-12_H-768_A-12'
#      ,'roberta_wwm':'chinese_roberta_wwm_ext_L-12_H-768_A-12'
#      ,'bert_wwm':'chinese_roberta_wwm_ext_L-12_H-768_A-12'
#      ,'roberta':'brightmart_roberta_zh_l12'
#      ,'electra':'chinese_electra_base_L-12_H-768_A-12'}
#     for model_name in model_path:
#         path = model_path[model_name]
#         hyperparameter_tune(model_name,path) 
# =============================================================================

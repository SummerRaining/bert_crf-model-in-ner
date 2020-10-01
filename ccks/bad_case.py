# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 12:55:25 2020
输入:model（训练好的模型）,data（原始样本列表）
输出：
1、错误率最高的10条样本。
2、错误数量最多的10条样本。

@author: tunan
"""
from util import logfile,_init,set_value,get_value
import os,random
from preprocess_data import load_data,data_generator
_init()
from main import train_single_model,build_model
from bert4keras.tokenizers import Tokenizer
#设置模型配置
set_value('learning_rate', 3e-5)
path = 'chinese_roberta_wwm_ext_L-12_H-768_A-12'
modeldata_path = get_value('modeldata_path')
set_value('config_path',os.path.join(modeldata_path,r'{}\bert_config.json'.format(path)))
set_value('checkpoint_path',os.path.join(modeldata_path,r'{}\bert_model.ckpt'.format(path)))
set_value('dict_path',os.path.join(modeldata_path,r'{}\vocab.txt'.format(path)))

#分割数据集,生成数据集配置。
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
X = load_data(data)     
test_data = load_data(test_data)
random.shuffle(X)
train_data = X[:int(len(X)*0.8)]
valid_data = X[int(len(X)*0.8):]
set_value('test_data',test_data)
set_value('train_data', train_data)
set_value('valid_data',valid_data)
import pickle
# =============================================================================
# with open('bad_case_data.pkl','wb') as f:
#     pickle.dump([train_data,valid_data,test_data], f)
# =============================================================================

#标签，分词器等
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
set_value('epochs',10)

train_single_model()
model = build_model()    #定义模型结构
model.load_weights(os.path.join(modeldata_path,r'model/origin_model.weights'))   #加载模型预测，返回预测值。

def add_pos(x):
    '''给输入样本添加文本位置
    '''
    line = [i[0] for i in x]
    d = []
    for i in range(len(line)):
        s,e = len(''.join(line[:i])),len(''.join(line[:i+1]))
        d.append(x[i]+[s,e] )
    return d

from predict_model import NamedEntityRecognizer
NER = NamedEntityRecognizer() 
def predict1(sample):
    text = ''.join([i[0] for i in sample]) 
    R = set(NER.recognize1(text, model))
    line = add_pos(sample)
    T = set([tuple(i) for i in line if i[1] != 'O'] ) 
    X, Y, Z = 1e-10, 1e-10, 1e-10
    X += len(R & T)                                  
    Y += len(R)                                      
    Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    
    case1 = R-T
    case2 = T-R
    wrong_num = len(R-T)+len(T-R)
    # print(wrong_num)
    return f1,wrong_num,case1,case2,line,R
result = []
for sample in test_data:
    f1,wrong_num,case1,case2,line,R = predict1(sample)
    result.append([f1,wrong_num,case1,case2,line,R])
sample = sorted(result,key=lambda x:x[0])[:10]
print(sample[0][3])
sample[0][2]
#找到错误率最高的样本

#后面在算概率，不要搞的那么复杂，bad case应该要按照需求一步一步来做。
# =============================================================================
# #一条记录返回两个度量，f1和wrong_num.和出错信息,case1(不是实体预测出错),case2(是实体但没有预测出来)
# #需要case的位置index，和预测的概率。
# from bert4keras.snippets import ViterbiDecoder, to_array  
# def predict_prob(text,model):
#     '''输入一条样本，返回实体和标签的列表。
#     mapping(list(list)):是一个列表，处理英文和数字，将数字作为一个token进行编码
#     '''                   
#     tokenizer = get_value('tokenizer')
#     tokens = tokenizer.tokenize(text)                  # 对其token化,转换成列表，且加入头部和尾部。输出的依然是字。
#     while len(tokens) > 512:                           #tokens截断到最大512.   
#         tokens.pop(-2)
#     mapping = tokenizer.rematch(text, tokens)       #重新匹配，句子和token序列。
#     token_ids = tokenizer.tokens_to_ids(tokens)     #转换成id序列。
#     segment_ids = [0] * len(token_ids)              #生成分区id。
#     token_ids, segment_ids = to_array([token_ids], [segment_ids])       
#     nodes = model.predict([token_ids, segment_ids])[0]      #预测该样本，得到的是crf的输出
#     return nodes,mapping
# 
# f1,wrong_num,case1,case2  = predict1()
# id2label = get_value('id2label')
# #把mapping文件的顺序对齐。
# text = ''.join([i[0] for i in test_data[0]]) 
# prob,mapping =  predict_prob(text, model)
# 
# def rematch_prob(prob,mapping):
#     new_p = []
#     for i in range(1,len(mapping)-1):
#         if len(mapping[i])>1:
#             tmp = [prob[i,:]]*len(mapping[i])
#         else:
#             tmp = [prob[i,:]]
#         new_p.extend(tmp)
#     return new_p
# 
# new_p = rematch_prob(prob,mapping)
# case2 = list(case2)
# result = []
# import numpy as np
# for target in case2:
#     s,e = target[2:4]
#     pred = new_p[s:e]
#     label_id = np.argmax(pred,axis = 1)
#     label = [id2label[i//2] if i//2 in id2label else 'O' for i in label_id]
#     prob = np.max(pred,axis = 1)
#     line = [(label[i],prob[i]) for i in range(len(label))]
#     result.append([target,line])
# 
# #把错误率比较高的句子输出来。观察是不是分词的错误
# text[9:30]
# 
# =============================================================================

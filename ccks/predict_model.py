# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 07:34:35 2020
"""
from bert4keras.snippets import ViterbiDecoder, to_array  
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
import os 
from util import logfile,_init,set_value,get_value


class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    def recognize(self, text,model):
        '''输入一条样本，返回实体和标签的列表。
        '''                   
        tokenizer,id2label = get_value('tokenizer'),get_value('id2label')
        tokens = tokenizer.tokenize(text)                  # 对其token化,转换成列表，且加入头部和尾部。输出的依然是字。
        while len(tokens) > 512:                           #tokens截断到最大512.   
            tokens.pop(-2)
        mapping = tokenizer.rematch(text, tokens)       #重新匹配，句子和token序列。
        token_ids = tokenizer.tokens_to_ids(tokens)     #转换成id序列。
        segment_ids = [0] * len(token_ids)              #生成分区id。
        token_ids, segment_ids = to_array([token_ids], [segment_ids])       
        nodes = model.predict([token_ids, segment_ids])[0]      #预测该样本，得到的是crf的输出
        labels = np.argmax(nodes,-1)                           #对输出值进行维特比解码。
        entities, starting = [], False                      
        for i, label in enumerate(list(labels)):       #根据预测值，生成样本的实体和对应label的tuple对。
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
                for w, l in entities]
    def recognize1(self, text,model):
        '''输入一条样本，返回实体和标签的列表。
        '''                   
        tokenizer,id2label = get_value('tokenizer'),get_value('id2label')
        tokens = tokenizer.tokenize(text)                  # 对其token化,转换成列表，且加入头部和尾部。输出的依然是字。
        while len(tokens) > 512:                           #tokens截断到最大512.   
            tokens.pop(-2)
        mapping = tokenizer.rematch(text, tokens)       #重新匹配，句子和token序列。
        token_ids = tokenizer.tokens_to_ids(tokens)     #转换成id序列。
        segment_ids = [0] * len(token_ids)              #生成分区id。
        token_ids, segment_ids = to_array([token_ids], [segment_ids])       
        nodes = model.predict([token_ids, segment_ids])[0]      #预测该样本，得到的是crf的输出
        labels = np.argmax(nodes,-1)                           #对输出值进行维特比解码。
        entities, starting = [], False                      
        for i, label in enumerate(list(labels)):       #根据预测值，生成样本的实体和对应label的tuple对。
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l,mapping[w[0]][0],mapping[w[-1]][-1]+1)
                for w, l in entities]


NER = NamedEntityRecognizer()      #ner预测器
def evaluate(data,model):  #评测函数data为验证集数据。数据形式为list
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data): #得到每条数据
        text = ''.join([i[0] for i in d]) #将文本部分拼接起来。
        R = set(NER.recognize(text,model))              #输入文本和模型，返回预测的实体
        T = set([tuple(i) for i in d if i[1] != 'O'])  #得到实体和对应label tuple对(实体文本，label)。即使该实体出现多次，只要有一个预测准确就可以了。
        X += len(R & T)                                  #计算所有预测准确的实体数。
        Y += len(R)                                      #计算所有预测为实体的个数。
        Z += len(T)                                      #计算真实实体个数。
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z           #f1计算是正确 的，等于recall和precision的几何平均。
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback): #自定义回调函数类。
    def __init__(self): #开始的时候令最优验证f1为0。
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None): 
        valid_data,test_data = get_value('valid_data'),get_value('test_data')
        modeldata_path = get_value('modeldata_path')
        f1, precision, recall = evaluate(valid_data,self.model)        #输入验证集数据，计算f1,precision,recall值。
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            self.model.save_weights(os.path.join(modeldata_path,r'model/origin_model.weights'))
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        f1, precision, recall = evaluate(test_data,self.model)    #训练阶段不应该使用测试集的数据
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )
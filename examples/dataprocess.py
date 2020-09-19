# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:21:24 2020

@author: a
"""
import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm

maxlen = 256
epochs = 10
batch_size = 16
bert_layers = 12
learing_rate = 1e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率

# bert配置
config_path = r'C:\Users\a\Desktop\bert4keras-master\examples\datasets\chinese_L-12_H-768_A-12\bert_config.json'
checkpoint_path = r'C:\Users\a\Desktop\bert4keras-master\examples\datasets\chinese_L-12_H-768_A-12\bert_model.ckpt'
dict_path = r'C:\Users\a\Desktop\bert4keras-master\examples\datasets\chinese_L-12_H-768_A-12\vocab.txt'


#每行只能作为一个json文件打开。
data = []
data1_path =  r'.\datasets\yidu-s4k\subtask1_training_part1.txt'
data2_path =  r'.\datasets\yidu-s4k\subtask1_training_part1.txt'
test_data_path = r'.\datasets\yidu-s4k\subtask1_test_set_with_answer.json'
with open(data1_path,'r',encoding='gbk') as f:
    data.extend(f.readlines())
with open(data2_path,'r',encoding='gbk') as f:
    data.extend(f.readlines())


#对于不是实体的部分用O填充，补充进去。
def add_O(wordlabel,length):    
    d = []
    if len(wordlabel) == 0:
        d.append({'label_type':'O','start_pos':0,'end_pos':length})
    else:
        pre_x = wordlabel[0]
        prei1,prei2 = pre_x['start_pos'],pre_x['end_pos']
        if prei1 >0:
            d.append({'label_type':'O','start_pos':0,'end_pos':prei1})
        for x in wordlabel[1:]:
            i1,i2 = x['start_pos'],x['end_pos']
            if prei2<i1:
                d.append({'label_type':'O','start_pos':prei2,'end_pos':i1})
            prei1,prei2 = i1,i2
        if prei2<length:
            d.append({'label_type':'O','start_pos':prei2,'end_pos':length})
            
    wordlabel.extend(d)
    wordlabel = sorted(wordlabel,key = lambda x:x['start_pos'])  
    return wordlabel


#目标文件，[sent1,sent2,...]，sent1:[(w1,label1),(w2,label2),...].
def load_data(data):
    def f(x):
        d = {'实验室检验':'检验','影像检查':'检查'}
        return d.get(x,x)
    
    D = []
    for line in tqdm(data):
        x = json.loads(line)  
        sentences = x['originalText']
        length = len(sentences)
        wordLabel = x['entities']
        wordlabel = sorted(wordLabel,key = lambda x:x['start_pos'])  
        wordlabel = add_O(wordlabel,length)  #添加不是实体的部分文本。
        
        d = []
        for x in wordlabel:
            i_beg,i_end = x['start_pos'],x['end_pos']
            word = sentences[i_beg:i_end]
            d.append([word,f(x['label_type'])])
        D.append(d)
    return D

#加载测试集数据
test_data = []
with open(test_data_path,'r',encoding='gbk') as f:
    test_data.extend(f.readlines())
    
train_data = load_data(data)
train_num = int(len(train_data)*0.8)

dev_data = train_data[train_num:]
train_data = train_data[:train_num]
test_data = load_data(test_data)

# 建立分词器,do_lower_case:只包含小写字母，大写字母作为unk token处理。
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 类别映射
labels = ['疾病和诊断', '检查', '检验','手术','药物','解剖部位']
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 2 + 1

#继承了自定义的DataGenerator
class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):    #顺序取每条样本item，is_end为标记表示是否为最后一条记录。
            token_ids, labels = [tokenizer._token_start_id], [0] #cls的id和0
            for w, l in item:  #一条样本中的，每个word和label。
                w_token_ids = tokenizer.encode(w)[0][1:-1]  #对每段单词编码，得到所有字的id。
                if len(token_ids) + len(w_token_ids) < maxlen: #如果已有的字小于最大长度，就加上当前id。
                    token_ids += w_token_ids
                    if l == 'O': #如果label为O，labels就直接增加对应长度的0。
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 2 + 1 #否则就对应到它的开头和中间部分，
                        I = label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
                
            #得到每个样本的字id和label id，长度等于句子长度。token_ids,labels
            #如果有一个实体正好在最大长度处卡断了，就去除整个实体。
            token_ids += [tokenizer._token_end_id]
            labels += [0]                        #输入和输出都加上结束符。label的开始符和结束符都为0。
            segment_ids = [0] * len(token_ids)   #分区id都为0.
            batch_token_ids.append(token_ids)    #将入到batch变量中。token_id，seg_id和label_id都加入batch变量中。
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:       #如果是最后一个单词，或者这个batch已经满了。
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)           #batch中的每个样本都padding到统一长度。
                yield [batch_token_ids, batch_segment_ids], batch_labels #返回一个batch的样本。
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

if __name__ == '__main__':
    train_generator = data_generator(train_data, batch_size)
    x = next(train_generator.forfit())
    
    


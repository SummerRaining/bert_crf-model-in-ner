# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 22:16:29 2020

@author: tunan
"""


#加载模型，预测数据。
#对一个句子得到预测的概率，得到预测正确概率的平均值。
#根据输入句子和
import copy
import json,os,random
import numpy as np
from tqdm import tqdm

#查看数据
train_path = r'C:\Users\tunan\Desktop\bert_crf-model-in-ner\examples\datasets\tianchi\train'
test_path = r'C:\Users\tunan\Desktop\bert_crf-model-in-ner\examples\datasets\tianchi\chusai_xuanshou'

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
    for x in tqdm(data):
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

#把每一条数据整理成：[原句子，[{'':word1,'':begin,'':end}]}开头，结尾的形式。
def load_train_data():
    train_data = []             #读取一条数据
    for index in tqdm(range(1000)):
        sentence = open(os.path.join(train_path,'{}.txt'.format(index)),'r',encoding='utf-8').read()       #读取原句子
        with open(os.path.join(train_path,'{}.ann'.format(index)),'r',encoding='utf-8') as f:               #读取标注信息
            lines = f.readlines()
            
        wordLabel = []              #对标注信息打包
        for i in range(len(lines)):
            d = {}
            l,w = lines[i].strip().split('\t')[1:]
            d['word'] = w
            
            s,bgn,end = l.split(' ')
            d['label_type'] = s
            d['start_pos'] = int(bgn)
            d['end_pos'] = int(end)
            wordLabel.append(d)
            # print(d['word']," == ",sentence[d['start_pos']:d['end_pos']])       #test
        
        train_data.append({'originalText':sentence,'entities':wordLabel})      #该样本放入训练集中。
    return train_data


def find_sub_word(s,max_len):
    '''将s拆分成子句，并返回每个子句的开始和结束位置。
    '''
    X = [len(a)+1 for a in s.split("。")]
    left = 0
    right = left+1
    A = [0]         #A代表单句分割点，sum(X[A[0]:A[1]])为第一个句子的长度。
    while right < len(X):
        if right==len(X)-1:
            A.append(right)
        elif sum(X[left:right])<max_len and sum(X[left:right+1])>max_len:
            A.append(right)
            left = right
        right += 1
        
    sub_length = [sum(X[A[i]:A[i+1]]) for i in range(len(A)-1)]
    sub_index = [[sum(sub_length[0:i]),sum(sub_length[0:i+1])] for i in range(len(sub_length))]  #bgn_pos,end_pos
    sub_w = [s[b:e] for b,e in sub_index] 
    return sub_w,sub_index

def find_sub_label(bgn,end,labels):
    '''找出所有在bgn和end之间的实体，并返回实体在子句中的相对位置。
    '''
    result = []
    for label in labels:
        if label['start_pos']>=bgn and label['end_pos']<=end:
            _label = copy.deepcopy(label)
            _label['start_pos'] -= bgn
            _label['end_pos'] -= bgn
            result.append(_label)
    return result

def split_data(line,max_len = 512):
    '''一个样本分解成多个样本
    '''
    sentence = line['originalText']
    sub_w,sub_index = find_sub_word(sentence,max_len)  #找到sentence的所有子句
    data = []
    for i in range(len(sub_index)):
        bgn,end = sub_index[i]
        sub_label = find_sub_label(labels = line['entities'],bgn = bgn,end = end)   #找到子句下的所有子标签
        data.append({'originalText':sub_w[i],'entities':sub_label})  #生成一个样本，加入data
    
    #测试，第一个子句中所有实体的位置和实体实际位置
    s = data[0]
    for label in s['entities']:
        print(s['originalText'][label['start_pos']:label['end_pos']],'==', label['word'])
        
    return data

def split_sentence(d):
    data = []
    for line in d:
        data.extend(split_data(line))
    return data


if __name__ =="__main__":
    #加载训练集数据
    d = load_train_data()
    # d = split_sentence (d)      #分割数据集
    # split_sentence(d)
    


    
    
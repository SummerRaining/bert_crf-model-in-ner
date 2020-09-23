# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 22:16:29 2020

@author: tunan
"""


#加载模型，预测数据。
#对一个句子得到预测的概率，得到预测正确概率的平均值。
#根据输入句子和
from my_load_data import *
import copy

def find_sub_word(s):
    '''将s拆分成子句，并返回每个子句的开始和结束位置。
    '''
    X = [len(a)+1 for a in s.split("。")]
    left = 0
    right = left+1
    A = [0]         #A代表单句分割点，sum(X[A[0]:A[1]])为第一个句子的长度。
    while right < len(X):
        if right==len(X)-1:
            A.append(right)
        elif sum(X[left:right])<256 and sum(X[left:right+1])>256:
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

def split_data(line):
    '''一个样本分解成多个样本
    '''
    sentence = line['originalText']
    sub_w,sub_index = find_sub_word(sentence)  #找到sentence的所有子句
    data = []
    for i in range(len(sub_index)):
        bgn,end = sub_index[i]
        sub_label = find_sub_label(labels = line['entities'],bgn = bgn,end = end)   #找到子句下的所有子标签
        data.append({'originalText':sub_w[i],'entities':sub_label})  #生成一个样本，加入data
    
# =============================================================================
#     #测试，第一个子句中所有实体的位置和实体实际位置
#     s = data[1]
#     for label in s['entities']:
#         print(s['originalText'][label['start_pos']:label['end_pos']],'==', label['word'])
# =============================================================================
    return data


if __name__ =="__main__":
    data = []
    for line in d:
        data.extend(split_data(line))
    


    
    
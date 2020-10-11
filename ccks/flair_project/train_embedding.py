# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 22:07:22 2020

@author: tunan
"""

from gensim.models import Word2Vec
import gensim
from tqdm import tqdm
import json
from custom_embedding import WordEmbeddings

with open(r'C:\Users\tunan\Desktop\bert_crf-model-in-ner\ccks\flair_project\dataset\train.txt','r',encoding='utf-8') as f:
    train_out = f.read()
with open(r'C:\Users\tunan\Desktop\bert_crf-model-in-ner\ccks\flair_project\dataset\test.txt','r',encoding='utf-8') as f:
    test_out = f.read()
with open(r'C:\Users\tunan\Desktop\bert_crf-model-in-ner\ccks\flair_project\dataset\dev.txt','r',encoding='utf-8') as f:
    dev_out = f.read()
output = '\n\n'.join([train_out,dev_out,test_out])
corpus = []
for sample in output.split('\n\n'):
    s = [x.split(' ')[0] for x in sample.split('\n')]
    corpus.append(s)

# =============================================================================
# model = Word2Vec(corpus,
#                         sg = 1,     # 0为CBOW  1为skip-gram
#                         size = 300, # 特征向量的维度
#                         window = 5, # 表示当前词与预测词在一个句子中的最大距离是多少
#                         min_count = 5, # 词频少于min_count次数的单词会被
#                         sample = 1e-3, # 高频词汇的随机降采样的配置阈值
#                         iter = 200,  #训练的次数 
#                         hs = 1,  #为 1 用hierarchical softmax   0 negative sampling
#                         workers=8 # 开启线程个数
#                         )
# 
# path = r"C:\Users\tunan\.flair\embeddings\custom\embedding.kv" # 语料库保存地址
# model.wv.save(path)
# =============================================================================
char_data = [] 
for l in corpus:
    char_data.extend(l)
all_token = set(char_data)
embedding = gensim.models.KeyedVectors.load(
               r"C:\Users\tunan\.flair\embeddings\custom\embedding.kv")
preembeddings = gensim.models.KeyedVectors.load(
               r"C:\Users\tunan\.flair\embeddings\zh-wiki-fasttext-300d-1M")


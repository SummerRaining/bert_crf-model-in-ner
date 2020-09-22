# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 22:24:53 2020

@author: tunan
"""
#输出记录重定向。
import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')

    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)

    def flush(self):
	    pass

sys.stdout = Logger('./log_bert_10.txt', sys.stdout)
sys.stderr = Logger('./log_bert_10.txt', sys.stderr)		# redirect std err, if necessary

import json,os,random
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import ViterbiDecoder, to_array  
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense,TimeDistributed,Bidirectional,LSTM,Dropout
from keras.models import Model
from tqdm import tqdm
from keras_lr_multiplier import LRMultiplier
from keras.utils.np_utils import to_categorical

#查看数据
train_path = r'C:\Users\tunan\Desktop\bert_crf-model-in-ner\examples\datasets\tianchi\train'
test_path = r'C:\Users\tunan\Desktop\bert_crf-model-in-ner\examples\datasets\tianchi\chusai_xuanshou'
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

#加载训练集数据
d = load_train_data()
train_data = load_data(d)

random.shuffle(train_data)      #划分数据集
train_num = int(len(train_data)*0.8)
valid_data = train_data[train_num:]
train_data = train_data[:train_num]

# 建立分词器,do_lower_case:只包含小写字母，大写字母作为unk token处理。
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 类别映射
labels = ['DRUG', 'DRUG_INGREDIENT', 'DISEASE','SYMPTOM','SYNDROME','DISEASE_GROUP',
          'FOOD','FOOD_GROUP','PERSON_GROUP','DRUG_GROUP','DRUG_DOSAGE','DRUG_TASTE','DRUG_EFFICACY'] 
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 2 + 1

#继承了自定义的DataGenerator
class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):     #顺序取每条样本item，is_end为标记表示是否为最后一条记录。
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
                batch_labels = to_categorical(batch_labels, num_classes=num_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels #返回一个batch的样本。
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []
                
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
model.summary()

#模型损失使用CRF.sparse_loss,adam学习器，CRF的离散准确率。
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learing_rate),
    metrics=['accuracy']
)


class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    def recognize(self, text):                       #预测text的实体结果，text为一条样本
        tokens = tokenizer.tokenize(text)            # 对其token化,转换成列表，且加入头部和尾部。输出的依然是字。
        while len(tokens) > 512:                     #tokens截断到最大512.   
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
    
    def recognize1(self, text):                       #预测text的实体结果，text为一条样本
        tokens = tokenizer.tokenize(text)            # 对其token化,转换成列表，且加入头部和尾部。输出的依然是字。
        while len(tokens) > 512:                     #tokens截断到最大512.   
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

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l,mapping[w[0]][0],mapping[w[-1]][-1] + 1)
                for w, l in entities]


NER = NamedEntityRecognizer() 
def evaluate(data):  #评测函数data为验证集数据。数据形式为list
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data): #得到每条数据
        text = ''.join([i[0] for i in d]) #将文本部分拼接起来。
        R = set(NER.recognize(text)) #
        T = set([tuple(i) for i in d if i[1] != 'O'])  #得到实体和对应label tuple对(实体文本，label)。即使该实体出现多次，只要有一个预测准确就可以了。
        X += len(R & T)                                  #计算所有预测准确的实体数。
        Y += len(R)                                      #计算所有预测为实体的个数。
        Z += len(T)                                      #计算真实实体个数。
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z           #f1计算是正确的，等于recall和precision的几何平均。
    return f1, precision, recall

class Evaluator(keras.callbacks.Callback): #自定义回调函数类。
    def __init__(self): #开始的时候令最优验证f1为0。
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None): #每个epoch结束时调用。
        f1, precision, recall = evaluate(valid_data)        #输入验证集数据，计算f1,precision,recall值。
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights(os.path.join(modeldata_path,r'tianchi_model/best_model.weights'))
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        
#预测文件
def predict_test():
    test_path = r'C:\Users\tunan\Desktop\bert_crf-model-in-ner\examples\datasets\tianchi\test'
    src_path = r'C:\Users\tunan\Desktop\bert_crf-model-in-ner\examples\datasets\tianchi\chusai_xuanshou'
    for index in tqdm(range(1000,1500)):
        sentence = open(os.path.join(src_path,'{}.txt'.format(index)),'r',encoding='utf-8').read()       #读取原句子
        try:
            R = set(NER.recognize1(sentence))                       #预测部分
        except:
            print(sentence)
        R = sorted(list(R),key = lambda x:x[2])                 #排序
        
        content = []                
        for i,line in enumerate(R):
            c = 'T{}\t{} {} {}\t{}'.format(i+1,line[1],line[2],line[3],line[0])
            content.append(c)
        
        p = os.path.join(test_path,'{}.ann'.format(index))      #目标文件的名称。
        with open(p,'w',encoding = 'utf-8') as f:               #写入目标文件
             f.write('\n'.join(content))


if __name__ == '__main__':
# =============================================================================
#     evaluator = Evaluator() 
#     train_generator = data_generator(train_data, batch_size)
#     
#     from sgdr_implementation import LR_Cycle
#     sched = LR_Cycle(iterations = len(train_generator),cycle_mult = 2)
#     histoty = model.fit_generator(
#         train_generator.forfit(), 
#         steps_per_epoch=len(train_generator),
#         epochs=10,
#         callbacks=[evaluator]
#     )
#     #学习率退火，训练10个epoch
#     histoty = model.fit_generator(
#         train_generator.forfit(),
#         steps_per_epoch=len(train_generator),
#         epochs=10,
#         callbacks=[evaluator,sched]
#     )
#     
#     # predict_test()          #生成提交结果
# else:
# =============================================================================

    model.load_weights(os.path.join(modeldata_path,r'tianchi_model/best_model.weights'))
    predict_test()          #生成提交结果


# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 06:57:03 2020

@author: tunan
"""
import json,os
from tqdm import tqdm
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.utils.np_utils import to_categorical
from myutil import _init,set_value,get_value
from bert4keras.tokenizers import Tokenizer

def add_O(wordlabel,length):    
    '''输入:wordlabel(list(dict)),length(int),输入原句的长度和一条样本数据。
        输出:wordlabel(list(dict)),输出加入O实体后的一条样本。
        对一条样本加入O实体。
    '''
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

def load_data(data):
    '''输入(list(str)):
        输出(list(str1,str2)):所有样本的标注数据。str1是句子实体，str2是实体类型。
        给定数据集数据，加载样本。
    '''
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

#继承了自定义的DataGenerator
class data_generator(DataGenerator):
    """数据生成器
    初始化函数输入：data(list(str1,str2)),batch_size(int)。data是所有训练数据的集合，batch_size为样本大小。
    输出：对象.for_fit()方法返回一个迭代器。每次会返回一个batch的训练数据。
    """
    def __iter__(self, random=False):
        tokenizer,label2id,num_labels,token2id = get_value('tokenizer'),get_value('label2id'),get_value('num_labels'),get_value('token2id')
        batch_token_ids, batch_labels = [], []
        for is_end, item in self.sample(random):    #顺序取每条样本item，is_end为标记表示是否为最后一条记录。
            #对于一条样本，将文本装换成id。
            token_ids, labels = [], []
            for w, l in item:  #一条样本中的，每个word和label。
                w_token_ids = [token2id.get(token,len(token2id)) for token in tokenizer._tokenize(w)]  #对每段单词编码，得到所有字的id。
                if len(token_ids) + len(w_token_ids) < get_value('maxlen'): #如果已有的字小于最大长度，就加上当前id。
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
            batch_token_ids.append(token_ids)    #将入到batch变量中。token_id，seg_id和label_id都加入batch变量中。
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:       #如果是最后一个单词，或者这个batch已经满了。
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_labels = sequence_padding(batch_labels)           #batch中的每个样本都padding到统一长度。
                batch_labels = to_categorical(batch_labels, num_classes=num_labels)
                yield batch_token_ids, batch_labels #返回一个batch的样本。
                batch_token_ids, batch_labels = [], []
                
def my_initialize():
    '''
    用于生成token到id的字典。
    '''
    import pickle
    if os.path.exists('token2id.pkl'):
        [token2id,id2token] = pickle.load( open('token2id.pkl','rb'))
        set_value('token2id',token2id)
        set_value('id2token',id2token)
        print('load token from file!')
        return 
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

    #分割数据集
    X = load_data(data)     
    test_data = load_data(test_data) 
    tokenizer = Tokenizer(get_value('dict_path'), do_lower_case=True)
    def generate_one_sample(sample):
        '''说明：输入一条样本，生成对应行列格式的样本。
        输入sample(list(list))：一个样本，每个实体的文本值和对应的实体。
        输出d（list((str,str))):每个字（分词）和对应的实体标签label。
        '''
        d = []
        for i in range(len(sample)):
            entitle = sample[i][0]
            label = sample[i][1]
            e = tokenizer._tokenize(entitle)
            if label == 'O':
                l = len(e)*[label]
            else:
                l = ['B-'+label if i==0 else 'I-'+label for i in range(len(e))]
            d.extend(list(zip(e,l)))
        return d
    
    output = []
    for sample in X:
        d = generate_one_sample(sample)
        output.append(d)
    test_out = []
    for sample in test_data:
        d = generate_one_sample(sample)
        test_out.append(d)
    
    #生成单词到id的字典
    corpus = test_out+output
    char = []
    for s in corpus:
        char.extend([x[0] for x in s])
    char = set(char)
    token2id = {c:i for i,c in enumerate(char)}
    id2token = {i:c for c,i in token2id.items()}
    set_value('token2id',token2id)
    set_value('id2token',id2token)
    import pickle
    pickle.dump([token2id,id2token], open('token2id.pkl','wb'))
    


if __name__ == '__main__':
    pass

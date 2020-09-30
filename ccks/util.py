# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 16:47:58 2020

@author: tunan
"""
#输出记录重定向。
import sys,os

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')

    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)

    def flush(self):
	    pass

def logfile(file_name = './log.txt'):
    sys.stdout = Logger(file_name, sys.stdout)
    sys.stderr = Logger(file_name, sys.stderr)		# redirect std err, if necessary



def _init():#初始化
    global _global_dict   #全局的类变量
    _global_dict = {'maxlen':512
         ,'epochs':10
         ,'batch_size':8
         ,'bert_layers':12
         ,'learning_rate':1e-5
         ,'crf_lr_multiplier':1000
         }# 必要时扩大CRF层的学习率
    
    #路径配置
    modeldata_path = r'C:\Users\tunan\model_data\bert4keras'
    _global_dict['modeldata_path'] = r'C:\Users\tunan\model_data\bert4keras'
    _global_dict['config_path'] = os.path.join(modeldata_path,r'chinese_L-12_H-768_A-12\bert_config.json')
    _global_dict['checkpoint_path'] = os.path.join(modeldata_path,r'chinese_L-12_H-768_A-12\bert_model.ckpt')
    _global_dict['dict_path'] = os.path.join(modeldata_path,r'chinese_L-12_H-768_A-12\vocab.txt')
 
def set_value(key,value):
    _global_dict[key] = value
 
def get_value(key,defValue=None):
    try:
        return _global_dict[key]
    except KeyError:
        return defValue


if __name__ == "__main__":
    sys.stdout = Logger('./log_new.txt', sys.stdout)
    sys.stderr = Logger('./log_new.txt', sys.stderr)		# redirect std err, if necessary

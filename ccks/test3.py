# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:16:14 2020

@author: tunan
"""
class config(object):
    def _init():#初始化
        global _global_dict
        _global_dict = {}
        _global_dict['var_test']= '初始值'
     
    def set_value(key,value):
        _global_dict[key] = value
     
    def get_value(key,defValue=None):
        return _global_dict.get(key,defValue)



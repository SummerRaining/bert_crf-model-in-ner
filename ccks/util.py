# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 16:47:58 2020

@author: tunan
"""
#输出记录重定向。
import sys

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'w')

    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)

    def flush(self):
	    pass

def logfile(file_name = './log.txt'):
    sys.stdout = Logger(file_name, sys.stdout)
    sys.stderr = Logger(file_name, sys.stderr)		# redirect std err, if necessary


if __name__ == "__main__":
    sys.stdout = Logger('./log_new.txt', sys.stdout)
    sys.stderr = Logger('./log_new.txt', sys.stderr)		# redirect std err, if necessary

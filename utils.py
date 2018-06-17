import os
import sys
import random
import logging

from PIL import Image


'''
    
'''

def init_logger(log_addr, name=None):
    
    global logger
    logger = logging.getLogger(name)
    
    if os.path.exists(log_addr):
        os.remove(log_addr)
    os.mknod(log_addr)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s]%(message)s')
    
    fileHandle = logging.FileHandler(log_addr)
    streamHandle = logging.StreamHandler()

    fileHandle.setLevel(logging.INFO)
    fileHandle.setFormatter(formatter)
    streamHandle.setLevel(logging.INFO)
    streamHandle.setFormatter(formatter)
    
    logger.addHandler(fileHandle)
    logger.addHandler(streamHandle)
    

def log(msg):
    #global looger
    logger.info(msg)


def load_train_data(data_addr, offset):
    
    dic = {}
    with open(data_addr, 'r') as txtfile: 
        for row in txtfile:
            dic[row.strip().split('\t')[1]] = int(row.strip().split('\t')[0]) - offset
    return dic

def load_test_data(test_addr, offset):
    
    res = {}
    txt = open(test_addr, 'r')
    for row in txt:
        res[row.strip().split('\t')[1]] = int(row.strip().split('\t')[0]) - offset
    
    txt.close()
    return res

	
def shuffle(data):
    return random.shuffle(data)


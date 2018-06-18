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
    

def logger(msg):
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

def load_train_img(data_addr, offset, size):
    
    res = load_train_data(data_addr, offset)
    
    labels = [res[key] for key in res.keys()]

    iamges = []
    for key in res.keys():
        image = Image.open(key).resize(size)
	images.append(image)

    return  zip(images, labels)
    
def load_test_img(data_addr, offset, size):

    res = load_test_data(data_addr, offset)

    labels = [res[key] for key in res.keys()]

    images = []
    for key in res.keys():
        image = Image.open(key).resize(size)
	images.append(image)

    return zip(images, labels)

'''
    Generator of batch data
'''

def GetBatchData(data,  batch_size, shuffle=True):
    length = len(data)
    while True:
        for i in range(0, length, batch_size):
            try:
                yield data[i:i+batch_size]
            except:
                if shuffle:
                    shuffle_data(data)
                break

def Crop_data(data, crop_size, crop_num, size):
    
    crop_data = []
    h, w, channle = crop_size
    h1, w1, channle1 = size
    rand = h1 - h
    for image, label in data:
        for i in crop_num:
	    l_x = l_y = random.randint(rand)
	    img = Image.crop(l_x, l_y, l_x + h, l_y + w)
	    if random.randint(h) < 2/h:
	        img = Image.rotate(random.randint(0, 180))
	    crop_data.append((img, label))

    data += crop_data
    return data
    
	
def shuffle_data(data):
    return random.shuffle(data)


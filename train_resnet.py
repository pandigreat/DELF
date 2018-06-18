import os

import numpy as np

import torch
import argparse

from PIL import Image

from torch import nn
from torch.autograd import Variable 
import  torchvision.models as imgnet_models
from utils import *

'''
    The function to get hyper-params from the scripts
'''

def get_args():
    parser = argparse.ArgumentParser("Delf ")
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--model', type=str, help='model save dir')
    parser.add_argument('--n_classes', type=int, help='num of classes')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_step', type=int, default=200, help='steps of updating of lr')
    parser.add_argument('--gamma', type=float, default=0.1, help='updaterate')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--test_iter', type=int, default=1000)
    parser.add_argument('--log', type=str, help='dir of logs')
    parser.add_argument('--save_iter', type=int, default=2000)
    parser.add_argument('--offset', type=int, default=-1)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--iters', type=int, default = 5000)

    args = parser.parse_args()
    return args
 
def make_model(nclasses):
    resnet50 = image_models.resnet50(pretrained=True)
    for param in resnet50.parameters():
        param.requires_grad = True
    
    #Fix the output nclasses
    resnet50.fc = nn.Linear(2048, nclasses)

    return resnet50


if __name__ == '__main__':
    args = get_args()
    
    init_logger(log_addar=args.log, name='resnet50')
    logger('start the train_resnet50')
    logger('Set images resize size')
    resize_size = (250, 250, 3)
    crop_size = (224, 224, 3)
    logger('Set images resize size over ')

    logger('Start to load train data')
    train_data = load_train_img(args.train_data, args.offset, resize_size)
    logger('Loading train data over...')

    logger('Start croping train data')
    Crop_data(train_data, crop_size, crop_size, resize_size)
    logger('Croping data over...')

    logger('Start to load test data')
    test_data = load_test_img(args.test_data, args.offset, resize_size)
    logger('Loading test data over...')

    if args.shuffle:
        logger('Start shuffle data')
        shuffle_data(train_data)
        shuffle_data(test_data)
        logger('Shuffling data over')

    logger('Loading pretrain model, resnet 50')
    model = make_model(args.n_classes)
    model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SDG(model.parameters(), lr=args.lr, momentum=0.8)

    model.train()
    logger('Start train models')

        
    data2train = GetBatchData(train_data, args.batch_size)
    data2test = GetBatchData(test_data, args.test_batch_size)
    for iters in range(args.iters+1):

        image_train, labels_train = zip(data2train.next())
        image_train = np.asarray(image_train)
        label_train = np.asarray(label_train)

        t_images = Variable(torch.from_numpy(image_train).long()).cuda()
        t_labels = Variable(torch.from_numpy(label_trian).long()).cuda()

        
        optimizer.zero_grad()
        pre_labels = model(t_images)
        loss = criterion(pre_labels, t_labels)
        loss.backward()
        optimizer.step()
        train_loss = loss.data[0]

        pre_labels = pre_labels.cpu().data.numpy()
        
        if iters % args.display_iters == 0:
            acc = get_acc(pre_labels, label_train)
            logger('Iter [%d / %d] -loss: %.10f -acc: %.10f' %(iters, args.iters, train_loss, acc)

        if iters % args.test_iters == 0:
            image_test, labels_test = zip(data2test.next())
            
            image_test = np.asarray(image_test)
            label_test = np.asarray(label_test)

            t_image = Variable(torch.from_numpy(image_test).long()).cuda()

            model.eval()
            test_pre_labels = model(image_test)
            y_prob = test.pre.labels.cpu().data.numpy()

            test_acc = get_acc(y_prob, label_test)

            output('Iter Test [%d / %d] --acc: %.10f' %(iters, args.iters, test_acc)

            model.train()

        if iters % args.save_iter == 0:
            save_file_name = os.path.join(args.output_dir, 'resnet50_iter_%d.pkl'%(iters))
	    torch.save(model.state_dict(), save_file_name)
            logger('Iter save [%d / %d]  %s' %(iters, args.iters, save_file_name))

        if iters % args.lr_step == 0:
	    lr = optimizer.state_dict()['param_groups']	[0]['lr']
            lr *= args.gamma

	    output('Iter [%d / %d] lr -> %.10f' %(iters, args.iters, lr))
	    for param_group in optimizer.param_groups:
	        param_group['lr'] = lr






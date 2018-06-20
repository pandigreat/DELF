import os


import torch
import argparse


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
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--test_iters', type=int, default=1000)
    parser.add_argument('--log', type=str, help='dir of logs')
    parser.add_argument('--save_iter', type=int, default=2000)
    parser.add_argument('--offset', type=int, default=-1)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--iters', type=int, default = 5000)
    parser.add_argument('--display_iter', type=int, default=200)
    parser.add_argument('--output_dir', type=str, help='model save dir')

    args = parser.parse_args()
    return args
 
def make_model(nclasses, trained=True):
    resnet50 = imgnet_models.resnet50(pretrained=trained)
    for param in resnet50.parameters():
        param.requires_grad = True
    
    #Fix the output nclasses
    resnet50.fc = nn.Linear(2048, nclasses)

    return resnet50


if __name__ == '__main__':
    args = get_args()
    
    init_logger(log_addr=args.log, name='resnet50')
    logger('start the train_resnet50 batch_size: %d' %(args.batch_size))
    logger('Set images resize size')
    resize_size = (250, 250)
    crop_size = (224, 224)
    logger('Set images resize size over ')

    logger('Start to load train data')
    train_data = load_train_img(args.train_data, args.offset, resize_size)
    train_data = list(train_data)
    logger('Loading train data over..., data size: %d' %(len(train_data)))

    logger('Start to load test data')
    test_data = load_test_img(args.test_data, args.offset, resize_size)
    test_data = list(test_data)
    logger('Loading test data over..., data size: %d' %(len(test_data)))

    logger('Start croping train data')
    train_data = Crop_data(train_data, crop_size, 5, resize_size)
    test_data = Crop_data(test_data, crop_size, 5, resize_size)
    logger('Croping data over...')

    if args.shuffle:
        logger('Start shuffle data')
        shuffle_data(train_data)
        shuffle_data(test_data)
        logger('Shuffling data over')

    logger('Loading pretrain model, resnet 50')
    model = make_model(args.n_classes)
    model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.8)

    model.train()
    logger('Start train models')

        
    data2train = GetBatchData(train_data, args.batch_size)
    data2test = GetBatchData(test_data, args.test_batch_size)

    n_train_id = 0
    n_test_id = 0

    for iters in range(args.iters+1):
        
        if n_train_id + args.batch_size < len(train_data):
            zip_train_data = train_data[n_train_id:n_train_id+args.batch_size]
            n_train_id += args.batch_size
        else:
            shuffle_data(train_data)
            n_train_id = 0
            zip_train_data = train_data[n_train_id:n_train_id+args.batch_size]
            n_train_id += args.batch_size

    
        image_train, labels_train = zip(*zip_train_data)
        

        label_train = np.asarray(list(labels_train))
        np_image_train = Image2numpy(image_train)
        np_image_train = np.array(np_image_train, dtype=float)
        image_train = np.transpose(np_image_train, (0, 3, 1, 2))

        t_images = Variable(torch.from_numpy(image_train).float()).cuda()
        t_labels = Variable(torch.from_numpy(label_train).long()).cuda()

        
        optimizer.zero_grad()
        pre_labels = model(t_images)
        loss = criterion(pre_labels, t_labels)
        train_loss = loss.data[0]
        loss.backward()
        optimizer.step()

        pre_labels = pre_labels.cpu().data.numpy()
        
        if iters % args.display_iter == 0 and iters > 0:
            acc = get_acc(pre_labels, label_train)
            logger('Iter [%d / %d] -loss: %.10f -acc: %.10f' %(iters, args.iters, train_loss, acc))
        if iters % args.test_iters == 0:
            
            if n_test_id + args.test_batch_size < len(test_data):
                zip_test_data = test_data[n_test_id:n_test_id+args.test_batch_size]
                n_test_id += args.test_batch_size
            else:
                shuffle_data(test_data)
                n_test_id = 0
                zip_test_data = test_data[n_test_id:n_test_id+args.test_batch_size]
                n_test_id += args.batch_size
            
            image_test, labels_test = zip(*zip_test_data)
            
            image_test = Image2numpy(image_test)
            image_test = np.array(image_test, dtype=float)
            image_test = np.transpose(image_test, (0, 3, 1, 2))
            label_test = np.asarray(labels_test)

            t_image = Variable(torch.from_numpy(image_test).float()).cuda()

            model.eval()
            test_pre_labels = model(t_image)
            y_prob = test_pre_labels.cpu().data.numpy()

            test_acc = get_acc(y_prob, label_test)

            logger('Iter Test [%d / %d] --acc: %.10f' %(iters, args.iters, test_acc))
            model.train()

        if iters % args.save_iter == 0:
            save_file_name = os.path.join(args.output_dir, 'resnet50_iter_%d.pkl'%(iters))
            torch.save(model.state_dict(), save_file_name)
            logger('Iter save [%d / %d]  %s' %(iters, args.iters, save_file_name))

        if iters % args.lr_step == 0:
            lr = optimizer.state_dict()['param_groups']	[0]['lr']
            lr *= args.gamma
            logger('Iter [%d / %d] lr -> %.10f' %(iters, args.iters, lr))
	    
            for param_group in optimizer.param_groups:
	            param_group['lr'] = lr






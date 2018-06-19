from train_resnet import *



if __name__ == '__main__':
    arg = get_args()
    
    init_logger(log_addr='./log/test_resnet.log')
    logger('Start testing....')
    logger('Set images resize size ')
    resize_size = (250, 250)
    crop_size = (224,224)

    logger('Loading test data and croping')
    test_data = load_test_img(arg.test_data, arg.offset, resize_size)
    test_data = Crop_data(test_data, crop_size, 6, resize_size)
    
    logger('Done ...')

    if arg.shuffle:
        logger('Shuffle data ')
        shuffle_data(test_data)
        logger('Done..')

    logger('Load resnet50')
    state_dict = torch.load(arg.model)
    model = make_model(arg.n_classes)
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    data, label = zip(*test_data)
    data = Image2numpy(data)
    label = np.array(label, dtype=int)
    rs = 0
    
    errorcase = {}

    for i in range(len(data)):
        t_data = np.array(data[i:i+1])
         
        t_data = np.transpose(t_data, (0, 3, 1, 2))
        
        t_data = Variable(torch.from_numpy(t_data).float()).cuda()
        
        pre_label = model(t_data)
        p_label = pre_label.cpu().data.numpy()
        
        if np.argmax(p_label) == label[i]:
            rs += 1
        else:
            errorcase[i] = (np.argmax(p_label), label[i])
        if i % arg.display_iter == 0 and i > 0:
            
            logger('Iter[%d / %d] [%d : %d]' %(i, len(data), np.argmax(p_label), label[i]))
    
    

    logger (float(rs) / float(len(data)))
    logger (errorcase)
        


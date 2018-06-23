from src.train_resnet import *
import torch.nn.functional as F
'''
    Get middle output from network

'''



#def hook(module, input, output):
    

class BasicAttentionBlock(nn.Module):
    def __init__(self, input_dim=2048, n_filters=2048, kernel_size=1, padding=0, stride=1, nclasses=18):
        super(BasicAttentionBlock, self).__init__()

        self.feature = []
        self.att_feat = []
        self.attention_score = []
        self.conv1 = nn.Conv2d(input_dim, 512, kernel_size=kernel_size, padding=padding, stride=stride)
        self.activate = nn.ReLU()
        self.conv2 = nn.Conv2d(512,1, kernel_size=kernel_size, padding=padding, stride=stride)
        #self.atten_prob = F.softplus()        

    def forward(self ,x ):
        
        self.feature.append(x.view(x.size(0), 2048, -1))
        feat = x
        #att_feat = F.normalize(feat, p=2,dim=0)
        att_feat = feat
        out = self.conv1(x)
        out = self.activate(out)
        out = self.conv2(out)
        score = out
        score = out.view(out.size(0), -1)
        self.attention_score.append(out.view(out.size(0), -1))
        
        prob = nn.Softplus(out)
        prob = F.softplus(out)
        att_feat = att_feat.view(att_feat.size(0), 2048, -1)
        self.att_feat.append(att_feat)
        prob = prob.view(prob.size(0), 1, -1)
        prob = torch.transpose(prob, 1, 2)
        #print(prob.shape, score.shape, att_feat.shape, torch.t(prob).shape)
        #prob = self.atten_prob(out) 
        
        att_feat = torch.matmul(att_feat, (prob))
        #print(att_feat.shape)
        
        out = att_feat
        return out

    def getFeature(self):
        result = ( self.feature, self.attention_score, self.att_feat)

        self.feature.clear()
        self.attention_score.clear()
        self.att_feat.clear()
        
        return result

def make_model_with_attention(nclasses, model_addr):
    model = make_model(nclasses)
    state_dict = torch.load(model_addr)
    model.load_state_dict(state_dict)
    for para in model.parameters():
        para.requires_grad = False  

    model.avgpool = BasicAttentionBlock(input_dim=2048)
    model.fc = nn.Linear(2048 ,nclasses)
    
    for para in model.avgpool.parameters():
        para.requires_grad = True
    for para in model.fc.parameters():
        para.requires_grad = True

    return model



if __name__ == '__main__':
    
    arg = get_args()
    init_logger(log_addr=arg.log, name='attention')

    logger('start train attention, batch_size :%d ' %(arg.batch_size))
    logger('Set image resize size')
    resize_size = (900, 900)
    #crop_size = (720, 720)
    crop_size = (224, 224)
    logger('loading train data, data size:' )
    train_data = list(load_train_img(arg.train_data, arg.offset, resize_size))

    logger('Loading test data, data size:')
    test_data = list(load_test_img(arg.test_data, arg.offset, resize_size))
    
    logger('Test data size: %d, Train Data size: %d'%(len(train_data), len(test_data)))
    logger('Croping data...')
    train_data = Crop_data(train_data, crop_size, 3, resize_size)
    test_data = Crop_data(test_data, crop_size, 3, resize_size)

    if arg.shuffle:
        logger('Shuffling data...' )
        shuffle_data(train_data)
        shuffle_data(test_data)

    logger('Loading model ')
    model = make_model_with_attention(nclasses=arg.n_classes,model_addr=arg.model)
    if torch.cuda.device_count() > 1:
        nn.DataParallel(model).cuda()
        logger('gpu num: %d'%(torch.cuda.device_count()))
    else:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=arg.lr, momentum=0.8)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=arg.lr, momentum=0.8)
    a = filter(lambda p:p.requires_grad, model.parameters())  
    getIgnore(a)
    model.train()
    logger("Start training ")

    n_train_id = 0
    n_test_id = 0

    for iters in range(arg.iters+1):
        
        if n_train_id + arg.batch_size < len(train_data):
            zip_train_data = train_data[n_train_id:n_train_id+arg.batch_size]
        else:
            shuffle_data(train_data)
            n_train_id = 0
            zip_train_data = train_data[n_train_id:n_train_id+arg.batch_size]
        n_train_id += arg.batch_size

        image_train, label_train = zip(*zip_train_data)
        zip_train_data.clear()
        label_train = np.asarray(label_train)
        image_train = Image2numpy(image_train)
        image_train = np.array(image_train)
        image_train = np.transpose(image_train, (0,3,1,2))

        t_image = Variable(torch.from_numpy(image_train).float()).cuda()
        t_label = Variable(torch.from_numpy(label_train).long()).cuda()

        optimizer.zero_grad()
        pre_label = model(t_image)
        model.avgpool.getFeature()
        loss = criterion(pre_label, t_label)
        train_loss = loss.data[0]
        loss.backward()
        optimizer.step()
        pre_label = pre_label.cpu().data.numpy()

        if iters > 0 and iters %arg.display_iter == 0:
            acc = get_acc(pre_label, label_train)
            logger('Show Iter [%d / %d] -loss: %.10f, -acc:%.10f'%(iters, arg.iters, train_loss, acc))
        if iters % arg.test_iters == 0:
            if n_test_id + arg.test_batch_size < len(test_data):
                zip_test_data = test_data[n_test_id:n_test_id+arg.test_batch_size]
            else:
                shuffle_data(test_data)
                n_test_id = 0
                zip_test_data = test_data[n_test_id:n_test_id+arg.test_batch_size]
            n_test_id += arg.test_batch_size

            image_test, label_test = zip(*zip_test_data)
            zip_test_data.clear()
            image_test = np.array(Image2numpy(image_test))
            image_test = np.transpose(image_test, (0, 3, 1, 2))
            label_test = np.asarray(label_test)

            t_image = Variable(torch.from_numpy(image_test).float()).cuda()

            model.eval()
            test_pre_label = model(t_image)
            model.avgpool.getFeature()
            test_pre_label = test_pre_label.cpu().data.numpy()
            acc = get_acc(test_pre_label, label_test)

            logger('Test Iter [%d / %d] -acc: %.10f'%(iters, arg.iters, acc))
            model.train()

        if iters % arg.save_iter == 0 and iters > 0:
            save_file_name = os.path.join(arg.output_dir, 'attention_iter_%d.pkl'%(iters))
            torch.save(model.state_dict(), save_file_name)
            logger('Save Iter [%d / %d]  attention_iter_%d.pkl'%(iters, arg.iters, iters))

        if iters % arg.lr_step == 0 and iters > 0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            lr *= arg.gamma
            logger("Update Iter [%d / %d] lr -> %.10f" %(iters, arg.iters, lr))
            
            for param in optimizer.param_groups:
                param['lr'] = lr



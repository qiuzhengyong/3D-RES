import torch
import torchvision
import torchvision.transforms as transforms
import os
from scipy.io.wavfile import read
import scipy.io.wavfile as wav
import subprocess as sp
import numpy as np
import argparse
import random
import os
import sys
import torch.nn.init as init
from random import shuffle
import speechpy
import datetime
from DataProviderDevelopment import AudioDataset, CMVN, Feature_Cube, ToOutput
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
from torch.optim.lr_scheduler import StepLR
import shutil


# Useful function for arguments.
def str2bool(v):
    return v.lower() in ("yes", "true")


parser = argparse.ArgumentParser(description='Creating background model in development phase')

#################
# Dataset Flags #
#################
parser.add_argument('--development_path',
                    default=os.path.expanduser('D:/voxceleb_data/voxceleb1_development.txt'),
                    help='The file names for development phase')
# parser.add_argument('--audio_dir', default=os.path.expanduser('D:/voxceleb_data/voxceleb1_audio'),
#                     help='Location of sound files')
parser.add_argument('--audio_dir', default=os.path.expanduser('D:/voxceleb_data/voxceleb1_audio'),
                    help='Location of sound files')
# parser.add_argument('--development_path',
#                     default=os.path.expanduser('~/autodl-tmp/voxceleb_data/voxceleb1_development.txt'),
#                     help='The file names for development phase')
# parser.add_argument('--audio_dir', default=os.path.expanduser('~/autodl-tmp/voxceleb_data/voxceleb1_audio'),
#                     help='Location of sound files')

#############################
# Finetuning & Resume Flags #
#############################
parser.add_argument('--resume', default=None, type=str,
# parser.add_argument('--resume', default='weights/netepoch_180.pth', type=str,
                    help="Resume from checkpoint. ex: os.path.expanduser('~/weights/net_final.pth')")
parser.add_argument('--fine_tuning', default=None, type=str,
                    help="Fine_tuning from checkpoint ex: os.path.expanduser('~/weights/net_epoch_10.pth')")
parser.add_argument('--trainable_layers', default=None, type=list,
                    help="Trainable layer and the other layers will be freezed. If it is None, all layers will be trainable ex:['fc1', 'fc2']")
parser.add_argument('--exlude_layer_from_checkpoint', default=None, type=list,
                    help="Layers to be excluded from checkpoint ex: ['fc1','fc2']")

######################
# Optimization Flags #
######################

parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--epochs_per_lr_drop', default=450, type=float,
                    help='number of epochs for which the learning rate drops')

##################
# Training Flags #
##################
parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--num_epoch', default=200, type=int, help='Number of training iterations')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--save_folder', default=os.path.expanduser('weights'), help='Location to save checkpoint models')
parser.add_argument('--epochs_per_save', default=10, type=int,
                    help='number of epochs for which the model will be saved')
parser.add_argument('--batch_per_log', default=10, type=int, help='Print the log at what number of batches?')

# Add all arguments to parser
args = parser.parse_args()

# Checking the appropriate folder for saving
# save_folder的绝对路径为C:\Users\user/weights
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


################################
### Initialization functions ###
################################

def xavier(param):
    init.xavier_uniform_(param)
    # init.xavier_uniform(param)


# Initializer function
def weights_init(m):
    """
    Different type of initialization have been used for conv and fc layers.
    :param m: layer
    :return: Initialized layer. Return occurs in-place.
    """
    if isinstance(m, nn.Conv3d):
        xavier(m.weight.data)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)


#######################
### Save & function ###
#######################

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    print('saving model ...')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


##################################
####### Initiate the dataset #####
##################################
development_data = AudioDataset(files_path=args.development_path,
                                audio_dir=args.audio_dir,
                                transform=transforms.Compose([CMVN(), Feature_Cube((80, 40, 20)), ToOutput()]))
trainloader = torch.utils.data.DataLoader(development_data, batch_size=args.batch_size,
                                          # num_workers=args.num_workers,pin_memory=True)
                                          shuffle=True, num_workers=args.num_workers, pin_memory=True)


# Ex: get some random training images (not used!)
# dataiter = iter(trainloader)
# item = dataiter.next()
# images, labels = item

#########################
### Inception Networks ##
#########################
# out_channels格式 [32, 32, 32, 32, 32, 32]
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        # 第一个分支，1x1卷积
        self.branch1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        # 第二个分支，1x1卷积后接3x3卷积
        self.branch3x3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        # 第三个分支，1x1卷积后接5x5卷积
        self.branch5x5 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.Conv3d(out_channels, out_channels, kernel_size=5, padding=2)
        )

        # 第四个分支，3x3最大池化后接1x1卷积
        self.branch_pool = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out1x1 = self.branch1x1(x)
        out3x3 = self.branch3x3(x)
        out5x5 = self.branch5x5(x)
        out_pool = self.branch_pool(x)

        out = torch.cat([out1x1, out3x3, out5x5, out_pool], dim=1)
        # 沿指定维度拼接张量。它可以将多个张量按照指定维度连接起来。
        return out

#########################
### Residual Networks ###
#########################
class ResidualBlock(nn.Module):
    def __init__(self, channels,kernel_size,padding,stride):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size, stride ,padding)
        self.bn1=nn.BatchNorm3d(channels)
        self.relu=torch.nn.PReLU()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size, stride ,padding)
        self.bn2=nn.BatchNorm3d(channels)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(x + y)
#############
### Model ###
#############
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        ################
        ### Method 1 ###
        ################
        """
        torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1)
        in_channels输入通道
        out_channels输出通道
        kernel_size 例如(4,9,9)，表示过滤器每次处理4帧，卷积核大小为9x9
        stride步长，stride=1表示stride=（1，1，1），在三维方向上步长是1，在宽和高上步长也是1。
        """

        """
        有padding的情况：
        output_depth = floor((input_depth + 2 * padding_depth - kernel_depth) / stride_depth + 1)
        output_height = floor((input_height + 2 * padding_height - kernel_height) / stride_height + 1)
        output_width = floor((input_width + 2 * padding_width - kernel_width) / stride_width + 1)
        没有padding的情况
        output_depth = floor((input_depth - kernel_depth) / stride_depth + 1)
        output_height = floor((input_height - kernel_height) / stride_height + 1)
        output_width = floor((input_width - kernel_width) / stride_width + 1)
        """

        self.conv11 = nn.Conv3d(1, 16, (4, 9, 9), stride=(1, 2, 1))
        self.conv11_bn = nn.BatchNorm3d(16)  # 标准化
        self.conv11_activation = torch.nn.PReLU()
        self.rblock11=ResidualBlock(channels=16,kernel_size=5,padding=(2, 2, 2),stride=(1, 1, 1))
        self.inception11=InceptionModule(in_channels=16,out_channels=4)

        self.conv12 = nn.Conv3d(16, 16, (4, 9, 9), stride=(1, 1, 1))
        self.conv12_bn = nn.BatchNorm3d(16)
        self.conv12_activation = torch.nn.PReLU()
        self.rblock12=ResidualBlock(channels=16,kernel_size=5,padding=(2, 2, 2),stride=(1, 1, 1))
        self.inception12=InceptionModule(in_channels=16,out_channels=4)


        self.conv21 = nn.Conv3d(16, 32, (3, 7, 7), stride=(1, 1, 1))
        self.conv21_bn = nn.BatchNorm3d(32)
        self.conv21_activation = torch.nn.PReLU()
        self.rblock21=ResidualBlock(channels=32,kernel_size=5,padding=(2, 2, 2),stride=(1, 1, 1))
        self.inception21=InceptionModule(in_channels=32,out_channels=8)


        self.conv22 = nn.Conv3d(32, 32, (3, 7, 7), stride=(1, 1, 1))
        self.conv22_bn = nn.BatchNorm3d(32)
        self.conv22_activation = torch.nn.PReLU()
        self.rblock22=ResidualBlock(channels=32,kernel_size=5,padding=(2, 2, 2),stride=(1, 1, 1))
        self.inception22=InceptionModule(in_channels=32,out_channels=8)


        self.conv31 = nn.Conv3d(32, 64, (3, 5, 5), stride=(1, 1, 1))
        self.conv31_bn = nn.BatchNorm3d(64)
        self.conv31_activation = torch.nn.PReLU()
        self.rblock31=ResidualBlock(channels=64,kernel_size=5,padding=(2, 2, 2),stride=(1, 1, 1))
        self.inception31=InceptionModule(in_channels=64,out_channels=16)


        self.conv32 = nn.Conv3d(64, 64, (3, 5, 5), stride=(1, 1, 1))
        self.conv32_bn = nn.BatchNorm3d(64)
        self.conv32_activation = torch.nn.PReLU()
        self.rblock32=ResidualBlock(channels=64,kernel_size=5,padding=(2, 2, 2),stride=(1, 1, 1))
        self.inception32=InceptionModule(in_channels=64,out_channels=16)


        self.conv41 = nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 1, 1))
        self.conv41_bn = nn.BatchNorm3d(128)
        self.conv41_activation = torch.nn.PReLU()
        self.rblock41=ResidualBlock(channels=128,kernel_size=5,padding=(2, 2, 2),stride=(1, 1, 1))
        self.inception41=InceptionModule(in_channels=128,out_channels=32)


        # Fully-connected
        self.fc1 = nn.Linear(128 * 4 * 6 * 2, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc1_activation = torch.nn.PReLU()
        self.fc2 = nn.Linear(128, 1211)

    def features(self, x):
        """
        x的维度,N,C,D,H,W
        N:批量大小,Batch_size,就是一批数据包含的数据样本量
        C:数据的通道大小
        D:数据的深度
        H:数据的高度
        W:数据的宽度
        """
        x = self.conv11_activation(self.conv11_bn(self.conv11(x)))  #第一层卷积
        x = self.rblock11(x)#第一层残差
        x = self.inception11(x)

        # x =  self.conv12(x)
        # x = self.conv12_bn(x)
        # x=self.conv12_activation(x)
        x = self.conv12_activation(self.conv12_bn(self.conv12(x)))  #第二层卷积
        x=self.rblock12(x)#第二层残差
        x = self.inception12(x)

        x = self.conv21_activation(self.conv21_bn(self.conv21(x)))  #第三层卷积
        x=self.rblock21(x)#第三层残差
        x = self.inception21(x)

        x = self.conv22_activation(self.conv22_bn(self.conv22(x)))  #第四层卷积
        x=self.rblock22(x)#第四层残差
        x = self.inception22(x)

        x = self.conv31_activation(self.conv31_bn(self.conv31(x)))  #第五层卷积
        x=self.rblock31(x)#第五层残差
        x = self.inception31(x)

        x = self.conv32_activation(self.conv32_bn(self.conv32(x)))  #第六层卷积
        x=self.rblock32(x)#第六层残差
        x = self.inception32(x)

        x = self.conv41_activation(self.conv41_bn(self.conv41(x)))  #第七层卷积
        x=self.rblock41(x)#第七层残差
        x = self.inception41(x)

        x = x.view(-1, 128 * 4 * 6 * 2)
        x = self.fc1_bn(self.fc1(x))
        x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-12)
        # 将输入的x按照维度1做p范数运算，即将某一个维度除以那个维度对应的范数。

        # # Method Sequential
        # x = self.cnn(x)
        # x = x.view(-1, 128 * 4 * 6 * 2)
        # x = self.fc(x)

        return x

    def forward(self, x):
        # Method-1
        x = self.features(x)
        x = self.fc1_activation(x)
        x = self.fc2(x)

        return x


# Call the net
model = Net()

############
### Cuda ###
############

# Multi GPU calling
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

# Put operations on GPU
if args.cuda and torch.cuda.is_available():
    print("USE GPU")
    model.cuda()

#################
### Optimizer ###
#################

# Metric for evaluation
criterion = nn.CrossEntropyLoss()

# Get the trainable variable list.
# model.state_dict()返回模型的所有参数，key 是网络层名称，value 则是该层的参数。
model_dict = model.state_dict()
if args.trainable_layers is not None:
    keys = [k for k, v in model_dict.items()]
    trainable_variable_list = []
    for key in keys:
        for layer_name in args.trainable_layers:
            if layer_name in key:
                trainable_variable_list.append(key)

    # Define optimizer with ignoring variables.
    # The variables that are not trainable, get the learning rate of zero!
    parameters_indicator = []
    for name, param in model.named_parameters():
        if name in trainable_variable_list:
            parameters_indicator.append({'params': param})
        else:
            parameters_indicator.append({'params': param, 'lr': 0.00001})

    # Define optimizer with putting the learning rate of some variables to zero!
    # The variables that we want to freeze them.
    optimizer = optim.SGD(parameters_indicator, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
else:
    # If args.trainable_layers == None, all layers set to be trainable.
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# Learning rate policy
scheduler = StepLR(optimizer, step_size=args.epochs_per_lr_drop, gamma=args.gamma)

#########################
### Resume & Finetune ###
#########################

if not args.resume and not args.fine_tuning:
    print('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    model.apply(weights_init)

else:
    # Proper assertion (we have to either start from a pretrained model or resume training)
    assert args.resume is None or args.fine_tuning is None, 'You want to resume or fine-tuning from a pretrained model\nboth flags cannot be true!?'
    if args.resume:
        if os.path.isfile(args.resume):
            print('Resume the training ...')
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch_from_resume = checkpoint['epoch']
            best_accuracy = checkpoint['best_accuracy']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    elif args.fine_tuning:
        '''
        Finetuning from pretrained weights.
        Finetuning from pretrained weights.

           * We do not want to resume, we just want the pretrained weights!
           * For this part we filtering out the keys that are available in pretrained weights and not the model at hand
           * we also filterout the keys that are not supposed to be loaded from checkpoint(excluded).
        '''

        # We only load the 'state_dict' related parameters which are the weights.
        print('Loading base network...')
        pretrained_dict = torch.load(os.path.join(args.save_folder, args.fine_tuning))['state_dict']
        model_dict = model.state_dict()

        # This part is for filtering out the keys that are available in pretrained weights and not the model at hand
        # & filtering out the keys that are not supposed to be loaded from checkpoint.
        # Get all the keys for defined layers to be excluded from checkpoint
        if args.exlude_layer_from_checkpoint is not None:
            keys = [k for k, v in model_dict.items()]
            exclude_model_dict = []
            for key in keys:
                for layer_name in args.exlude_layer_from_checkpoint:
                    if layer_name in key:
                        exclude_model_dict.append(key)

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in exclude_model_dict}
        else:
            # This part is for just filtering out the keys that are available in pretrained weights and not the model
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        # 3. load the new state dict
        model.load_state_dict(model_dict)

######################
### Training loop ####
######################
best_accuracy = 0.0
num_batches = len(trainloader)
print("num_batches:", num_batches)
# Start epochs from resume
if args.resume:
    start = start_epoch_from_resume
else:
    start = 0
for epoch in range(start, args.num_epoch):  # loop over the dataset multiple times

    # # Step the lr scheduler each epoch!
    # scheduler.step()

    # Running loss would be initiated for each iteration.
    running_loss = 0.0
    running_accuracy = 0.0
    # 调用trainloader,在DataProviderDevelopment中调用speechpy.processing.stack_frames
    # 里面有print(numframes,length_signal,frame_sample_length,frame_stride)
    # 可以去注释掉
    for iteration, data in enumerate(trainloader, 1):

        t0 = time.time()

        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if args.cuda:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        # Loss
        loss = criterion(outputs, labels)

        # Prediction
        _, predictions = torch.max(outputs, dim=1)
        # pred == y returns a ByteTensor, which has only an 8-bit range. Hence, after a particular batch-size, the sum may overflow
        # and hence shoot the wrong results.

        # correct_count = (predictions == labels).double().sum().data[0]
        correct_count = (predictions == labels).double().sum().item()
        print("correct_count:", correct_count);
        accuracy = float(correct_count) / args.batch_size  # accuracy最大为1

        # best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy

        # forward, backward & optimization
        loss.backward()
        optimizer.step()

        # print statistics
        # running_loss += loss.data[0]
        running_loss += loss.item()
        running_accuracy += accuracy
        t1 = time.time()
        duration_estimate = t1 - t0

        if iteration % args.batch_per_log == 0:
            print('Estimated time for each batch: {:.4f} sec.\n'.format(duration_estimate), end=' ')
            print((
                    'epoch {:2d} ' + '|| batch {:2d} of {:2d} ||' + ' Loss: {:.4f} ||' + ' Batch-Accuracy: {:.4f} ||'+' best_accuracy: {:.4f} ||\n').format(
                epoch + 1, iteration, num_batches, running_loss / args.batch_per_log, accuracy, best_accuracy), end=' ')
            running_loss = 0.0
    # Step the lr scheduler each epoch!
    scheduler.step()

    # Print the averaged accuracy for epoch
    print('The averaged accuracy for each epoch: {:.4f}.\n'.format(100.0 * running_accuracy / num_batches), end=' ')
    if int(epoch + 1) % args.epochs_per_save == 0:
        # Save the model after some epochs.
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_accuracy': best_accuracy,
            'optimizer': optimizer.state_dict(),
        }, is_best=accuracy == best_accuracy, filename=os.path.join(args.save_folder, 'net_epoch_' +
                                                                    str(epoch + 1)) + '.pth')

# Save the final model at the end of training.
save_checkpoint({
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),
    'best_accuracy': best_accuracy,
    'optimizer': optimizer.state_dict(),
}, is_best=accuracy == best_accuracy, filename=os.path.join(args.save_folder, 'net_final') + '.pth')
print('Finished Training')

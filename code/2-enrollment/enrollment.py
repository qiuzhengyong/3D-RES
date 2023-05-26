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
sys.path.insert(0, '../InputPipeline')
from DataProviderEnrollment import AudioDataset, CMVN, Feature_Cube, ToOutput
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import Normalizer
import shutil

# Useful function for arguments.
def str2bool(v):
    return v.lower() in ("yes", "true")


######################
# Optimization Flags #
######################
parser = argparse.ArgumentParser(description='Creating background model in development phase')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')

parser.add_argument('--batch_size', default=40, type=int, help='Batch size for training')
# parser.add_argument('--batch_size', default=5, type=int, help='Batch size for training')
parser.add_argument('--basenet', default=None, help='pretrained base model')
parser.add_argument('--load_weights', default='../1-development/weights/net_final.pth', type=str, help='Load weights')
# parser.add_argument('--load_weights', default='~/weights/net_final.pth', type=str, help='Load weights')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
# parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--batch_per_log', default=10, type=int, help='Print the log at what number of batches?')
parser.add_argument('--enrollment_path',
                    # default=os.path.expanduser('~/autodl-tmp/voxceleb_data/voxceleb1_enrollment.txt'),
                    default=os.path.expanduser('D:/voxceleb_data/voxceleb1_enrollment.txt'),
                    help='The file names for enrollment phase')
parser.add_argument('--save_folder', default='model', help='Location to save models')
parser.add_argument('--audio_dir',
                    default=os.path.expanduser('D:/voxceleb_data/voxceleb1_audio'),
                    # default=os.path.expanduser('~/autodl-tmp/voxceleb_data/voxceleb1_audio'),
                    help='Location of sound files')
parser.add_argument('--epochs_per_save', default=10, type=int,
                    help='number of epochs for which the model will be saved')
args = parser.parse_args()

# Checking the appropriate folder for saving
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

##################################
####### Initiate the dataset #####
##################################
enrollment_data = AudioDataset(files_path=args.enrollment_path,
                                audio_dir=args.audio_dir,
                                transform=transforms.Compose([CMVN(), Feature_Cube((80, 40, 20)), ToOutput()]))

dataloader = torch.utils.data.DataLoader(enrollment_data, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.num_workers)

# Ex: get some random training images (not used!)
# dataiter = iter(dataloader)
# images, labels = dataiter.next()

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
        self.conv11 = nn.Conv3d(1, 16, (4, 9, 9), stride=(1, 2, 1))
        self.conv11_bn = nn.BatchNorm3d(16)  # 标准化
        self.conv11_activation = torch.nn.PReLU()
        self.rblock11 = ResidualBlock(channels=16, kernel_size=5, padding=(2, 2, 2), stride=(1, 1, 1))
        self.inception11 = InceptionModule(in_channels=16, out_channels=4)

        self.conv12 = nn.Conv3d(16, 16, (4, 9, 9), stride=(1, 1, 1))
        self.conv12_bn = nn.BatchNorm3d(16)
        self.conv12_activation = torch.nn.PReLU()
        self.rblock12 = ResidualBlock(channels=16, kernel_size=5, padding=(2, 2, 2), stride=(1, 1, 1))
        self.inception12 = InceptionModule(in_channels=16, out_channels=4)

        self.conv21 = nn.Conv3d(16, 32, (3, 7, 7), stride=(1, 1, 1))
        self.conv21_bn = nn.BatchNorm3d(32)
        self.conv21_activation = torch.nn.PReLU()
        self.rblock21 = ResidualBlock(channels=32, kernel_size=5, padding=(2, 2, 2), stride=(1, 1, 1))
        self.inception21 = InceptionModule(in_channels=32, out_channels=8)

        self.conv22 = nn.Conv3d(32, 32, (3, 7, 7), stride=(1, 1, 1))
        self.conv22_bn = nn.BatchNorm3d(32)
        self.conv22_activation = torch.nn.PReLU()
        self.rblock22 = ResidualBlock(channels=32, kernel_size=5, padding=(2, 2, 2), stride=(1, 1, 1))
        self.inception22 = InceptionModule(in_channels=32, out_channels=8)

        self.conv31 = nn.Conv3d(32, 64, (3, 5, 5), stride=(1, 1, 1))
        self.conv31_bn = nn.BatchNorm3d(64)
        self.conv31_activation = torch.nn.PReLU()
        self.rblock31 = ResidualBlock(channels=64, kernel_size=5, padding=(2, 2, 2), stride=(1, 1, 1))
        self.inception31 = InceptionModule(in_channels=64, out_channels=16)

        self.conv32 = nn.Conv3d(64, 64, (3, 5, 5), stride=(1, 1, 1))
        self.conv32_bn = nn.BatchNorm3d(64)
        self.conv32_activation = torch.nn.PReLU()
        self.rblock32 = ResidualBlock(channels=64, kernel_size=5, padding=(2, 2, 2), stride=(1, 1, 1))
        self.inception32 = InceptionModule(in_channels=64, out_channels=16)

        self.conv41 = nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 1, 1))
        self.conv41_bn = nn.BatchNorm3d(128)
        self.conv41_activation = torch.nn.PReLU()
        self.rblock41 = ResidualBlock(channels=128, kernel_size=5, padding=(2, 2, 2), stride=(1, 1, 1))
        self.inception41 = InceptionModule(in_channels=128, out_channels=32)


        # Fully-connected
        self.fc1 = nn.Linear(128 * 4 * 6 * 2, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc1_activation = torch.nn.PReLU()
        self.fc2 = nn.Linear(128, 1211)

    def forward(self, x):
        x = self.conv11_activation(self.conv11_bn(self.conv11(x)))  # 第一层卷积
        x = self.rblock11(x)  # 第一层残差
        x = self.inception11(x)

        # x =  self.conv12(x)
        # x = self.conv12_bn(x)
        # x=self.conv12_activation(x)
        x = self.conv12_activation(self.conv12_bn(self.conv12(x)))  # 第二层卷积
        x = self.rblock12(x)  # 第二层残差
        x = self.inception12(x)

        x = self.conv21_activation(self.conv21_bn(self.conv21(x)))  # 第三层卷积
        x = self.rblock21(x)  # 第三层残差
        x = self.inception21(x)

        x = self.conv22_activation(self.conv22_bn(self.conv22(x)))  # 第四层卷积
        x = self.rblock22(x)  # 第四层残差
        x = self.inception22(x)

        x = self.conv31_activation(self.conv31_bn(self.conv31(x)))  # 第五层卷积
        x = self.rblock31(x)  # 第五层残差
        x = self.inception31(x)

        x = self.conv32_activation(self.conv32_bn(self.conv32(x)))  # 第六层卷积
        x = self.rblock32(x)  # 第六层残差
        x = self.inception32(x)

        x = self.conv41_activation(self.conv41_bn(self.conv41(x)))  # 第七层卷积
        x = self.rblock41(x)  # 第七层残差
        x = self.inception41(x)

        x = x.view(-1, 128 * 4 * 6 * 2)
        # x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-12)
        x=self.fc1(x)
        x=self.fc1_bn(x)
        # x=self.fc1_activation(x)
        # x = self.fc1_activation(self.fc1(x))
        # x = self.fc2(x)
        return x


# Call the net
net = Net()

#################
### Optimizer ###
#################
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
############
### load ###
############

weights = torch.load(os.path.join(args.load_weights))
# We only load the 'state_dict' related parameters which are the weights.
# The reason is that we do not want to resume, we just want the pretrained weights!
net.load_state_dict(weights['state_dict'])
net.fc2 = nn.Linear(128, 40)

############
### Cuda ###
############
# Multi GPU calling
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)

# Put operations on GPU
if args.cuda and torch.cuda.is_available():
    net.cuda()

######################
### Training loop ####
######################
num_batches = len(dataloader)
running_loss = 0.0

num_enrollment = 200
# output_numpy = np.zeros(shape=[num_enrollment,5,40],dtype=np.float32)
# model = np.zeros(shape=[5,40],dtype=np.float32)
output_numpy = np.zeros(shape=[num_enrollment,40,128],dtype=np.float32)
model = np.zeros(shape=[40,128],dtype=np.float32)


for i in range(num_enrollment):
    running_accuracy = 0.0
    for iteration, data in enumerate(dataloader, 1):
        # get the inputs
        inputs, labels = data

        if(labels.shape[0]<args.batch_size):
            break
        # wrap them in Variable
        if args.cuda:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)


        # forward + backward + optimize
        t0 = time.time()
        outputs = net(inputs)
        output_numpy[i] = outputs.cpu().data.numpy()
        for index, label in enumerate(labels):
            enrollment_model = output_numpy[i]
            enrollment_model = Normalizer(norm='l2').fit_transform(enrollment_model)
            model[label] += enrollment_model[index]
        # zero the parameter gradients
        optimizer.zero_grad()

        # Loss
        loss = criterion(outputs, labels)

        # Prediction
        _, predictions = torch.max(outputs, dim=-1)
        # pred == y returns a ByteTensor, which has only an 8-bit range. Hence, after a particular batch-size, the sum may overflow
        # and hence shoot the wrong results.

        # correct_count = (predictions == labels).double().sum().data[0]
        correct_count = (predictions == labels).double().sum().item()
        accuracy = float(correct_count) / args.batch_size

        # forward, backward & optimization
        t1 = time.time()
        duration_estimate = t1 - t0

        # forward, backward & optimization
        loss.backward()
        optimizer.step()

        # print statistics
        # running_loss += loss.data[0]
        running_loss += loss.item()
        running_accuracy += accuracy
        if (iteration+1) % args.batch_per_log == 0:
            print('Estimated time for each batch: {:.4f} sec.\n'.format(duration_estimate), end=' ')
            print(('epoch {:2d} ' + '|| batch {:2d} of {:2d} ||' + ' Loss: {:.4f} ||' + ' Batch-Accuracy: {:.4f} ||\n').format(
                i + 1, iteration, num_batches, running_loss / args.batch_per_log, accuracy), end=' ')
                # epoch + 1, iteration, num_batches, running_loss / args.batch_per_log, accuracy), end=' ')
            running_loss = 0.0

        # Print the averaged accuracy for epoch
        print('The averaged accuracy for each epoch: {:.4f}.\n'.format(100.0 * running_accuracy / num_batches), end=' ')

        if int(i + 1) % args.epochs_per_save == 0:
            # Save the model after some epochs.
            torch.save(net.state_dict(), './/weights//net_epoch_' + str(i + 1) + '.pth')

torch.save(net.state_dict(), ".//weights//net_final.pth")


# for i in range(output_numpy.shape[0]):
#     enrollment_model = output_numpy[i]
#     enrollment_model = Normalizer(norm='l2').fit_transform(enrollment_model)
#     model += enrollment_model
model = model / float(num_enrollment)

# Save the final model at the end of training.

print('Finished Training')

#model = outputs.cpu().data.numpy()
np.save(os.path.join(args.save_folder,'model.npy'),model)
print("model saved!")


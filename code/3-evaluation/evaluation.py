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
from sklearn.preprocessing import Normalizer

sys.path.insert(0, '../InputPipeline')
# from DataProviderEvaluation import   AudioDataset,CMVN, Feature_Cube, ToOutput
from DataProviderEnrollment import AudioDataset,CMVN, Feature_Cube, ToOutput
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics.pairwise import cosine_similarity


# Useful function for arguments.
def str2bool(v):
    return v.lower() in ("yes", "true")


parser = argparse.ArgumentParser(description='Creating background model in development phase')
parser.add_argument('--batch_size', default=5, type=int, help='Batch size for training')
parser.add_argument('--basenet', default=None, help='pretrained base model')
parser.add_argument('--load_weights', default='../2-enrollment/weights/net_final.pth', type=str, help='Load weights')
# parser.add_argument('--load_weights', default='../2-enrollment/weights/model_best.pth', type=str, help='Load weights')
parser.add_argument('--enrollment_dir', default='../2-enrollment', type=str, help='Load weights')
parser.add_argument('--evaluation_dir', default='../3-evaluation/ROC_DATA', type=str, help='Load weights')
# parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--batch_per_log', default=10, type=int, help='Print the log at what number of batches?')
parser.add_argument('--evaluation_path',
                    # default=os.path.expanduser('D:/voxceleb_data/voxceleb1_evaluation.txt'),
                    default=os.path.expanduser('D:/voxceleb_data/voxceleb1_enrollment.txt'),
                    help='The file names for evaluation phase')
parser.add_argument('--save_folder', default='model', help='Location to save models')
parser.add_argument('--audio_dir',
                    default=os.path.expanduser('D:/voxceleb_data/voxceleb1_audio'),
                    help='Location of sound files')
args = parser.parse_args()

# Checking the appropriate folder for saving
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

##################################
####### Initiate the dataset #####
##################################
evaluation_data = AudioDataset(files_path=args.evaluation_path,
                               audio_dir=args.audio_dir,
                               transform=transforms.Compose([CMVN(), Feature_Cube((80, 40, 20)), ToOutput()]))

dataloader = torch.utils.data.DataLoader(evaluation_data, batch_size=args.batch_size,
                                         shuffle=True, num_workers=args.num_workers)


# Ex: get some random training images (not used!)
# dataiter = iter(dataloader)
# images, labels = dataiter.next()


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

        self.conv12 = nn.Conv3d(16, 16, (4, 9, 9), stride=(1, 1, 1))
        self.conv12_bn = nn.BatchNorm3d(16)
        self.conv12_activation = torch.nn.PReLU()
        self.rblock12 = ResidualBlock(channels=16, kernel_size=5, padding=(2, 2, 2), stride=(1, 1, 1))

        self.conv21 = nn.Conv3d(16, 32, (3, 7, 7), stride=(1, 1, 1))
        self.conv21_bn = nn.BatchNorm3d(32)
        self.conv21_activation = torch.nn.PReLU()
        self.rblock21 = ResidualBlock(channels=32, kernel_size=5, padding=(2, 2, 2), stride=(1, 1, 1))

        self.conv22 = nn.Conv3d(32, 32, (3, 7, 7), stride=(1, 1, 1))
        self.conv22_bn = nn.BatchNorm3d(32)
        self.conv22_activation = torch.nn.PReLU()
        self.rblock22 = ResidualBlock(channels=32, kernel_size=5, padding=(2, 2, 2), stride=(1, 1, 1))

        self.conv31 = nn.Conv3d(32, 64, (3, 5, 5), stride=(1, 1, 1))
        self.conv31_bn = nn.BatchNorm3d(64)
        self.conv31_activation = torch.nn.PReLU()
        self.rblock31 = ResidualBlock(channels=64, kernel_size=5, padding=(2, 2, 2), stride=(1, 1, 1))

        self.conv32 = nn.Conv3d(64, 64, (3, 5, 5), stride=(1, 1, 1))
        self.conv32_bn = nn.BatchNorm3d(64)
        self.conv32_activation = torch.nn.PReLU()
        self.rblock32 = ResidualBlock(channels=64, kernel_size=5, padding=(2, 2, 2), stride=(1, 1, 1))

        self.conv41 = nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 1, 1))
        self.conv41_bn = nn.BatchNorm3d(128)
        self.conv41_activation = torch.nn.PReLU()
        self.rblock41 = ResidualBlock(channels=128, kernel_size=5, padding=(2, 2, 2), stride=(1, 1, 1))


        # Fully-connected
        self.fc1 = nn.Linear(128 * 4 * 6 * 2, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc1_activation = torch.nn.PReLU()
        self.fc2 = nn.Linear(128, 40)

    def forward(self, x):
        x = self.conv11_activation(self.conv11_bn(self.conv11(x)))  # 第一层卷积
        x = self.rblock11(x)  # 第一层残差
        x = self.conv12_activation(self.conv12_bn(self.conv12(x)))  # 第二层卷积
        x = self.rblock12(x)  # 第二层残差
        x = self.conv21_activation(self.conv21_bn(self.conv21(x)))  # 第三层卷积
        x = self.rblock21(x)  # 第三层残差
        x = self.conv22_activation(self.conv22_bn(self.conv22(x)))  # 第四层卷积
        x = self.rblock22(x)  # 第四层残差
        x = self.conv31_activation(self.conv31_bn(self.conv31(x)))  # 第五层卷积
        x = self.rblock31(x)  # 第五层残差
        x = self.conv32_activation(self.conv32_bn(self.conv32(x)))  # 第六层卷积
        x = self.rblock32(x)  # 第六层残差
        x = self.conv41_activation(self.conv41_bn(self.conv41(x)))  # 第七层卷积
        x = self.rblock41(x)  # 第七层残差
        x = x.view(-1, 128 * 4 * 6 * 2)
        x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-12)
        x = self.fc1_activation(self.fc1(x))
        x = self.fc2(x)
        # x = self.fc1_bn(self.fc1(x))
        # x = F.relu(self.conv2_bn(self.conv42(x))
        return x


# Call the net
net = Net()

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

#################
### Optimizer ###
#################
criterion = nn.CrossEntropyLoss()

############
### load ###
############

weights = torch.load(os.path.join(args.load_weights))
print('Loading base network...')
# We only load the 'state_dict' related parameters which are the weights.
# The reason is that we do not want to resume, we just want the pretrained weights!
net.load_state_dict(weights)

######################
### Training loop ####
######################
num_batches = len(dataloader)

running_loss = 0.0
running_accuracy = 0.0
feature_vector = np.zeros((num_batches * args.batch_size, 40), dtype=np.float32)
label_vector = np.zeros((num_batches * args.batch_size), dtype=np.float32)

for iteration, data in enumerate(dataloader, 0):

    # get the inputs
    inputs, labels = data

    # wrap them in Variable
    if args.cuda:
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)

    # forward + backward + optimize
    t0 = time.time()
    outputs = net(inputs)

    if labels.cpu().data.numpy().shape[0] == args.batch_size:
        feature_vector[iteration * args.batch_size:(iteration + 1) * args.batch_size] = outputs.cpu().data.numpy()
        label_vector[iteration * args.batch_size:(iteration + 1) * args.batch_size] = labels.cpu().data.numpy()
    else:
        feature_vector[iteration * args.batch_size:iteration * args.batch_size + labels.cpu().data.numpy().shape[
            0]] = outputs.cpu().data.numpy()
        label_vector[iteration * args.batch_size:iteration * args.batch_size + labels.cpu().data.numpy().shape[
            0]] = labels.cpu().data.numpy()

        feature_vector = np.delete(feature_vector,np.s_[iteration * args.batch_size + labels.cpu().data.numpy().shape[0]:], 0)
        label_vector = np.delete(label_vector,np.s_[iteration * args.batch_size + labels.cpu().data.numpy().shape[0]:], 0)

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

    # print statistics
    # running_loss += loss.data[0]
    running_loss += loss.item()
    running_accuracy += accuracy
    if (iteration + 1) % args.batch_per_log == 0:
        print('Estimated time for each batch: {:.4f} sec.\n'.format(duration_estimate), end=' ')
        print(('|| batch {:2d} of {:2d} ||' + ' Loss: {:.4f} ||' + ' Batch-Accuracy: {:.4f} ||\n')
              .format(iteration,num_batches,running_loss / args.batch_per_log,accuracy),
              end=' ')
        running_loss = 0.0

# Print the averaged accuracy for epoch
# print('The averaged accuracy: {:.4f}.\n'.format(100.0 * running_accuracy / num_batches), end=' ')
print('The averaged accuracy: {:.4f}.\n'.format(100.0 * running_accuracy / iteration), end=' ')




########################################
########## SCORE COMPUTATION ###########
########################################

speaker_model_path = os.path.join(args.enrollment_dir, 'model', 'model.npy')
# speaker_model_path = os.path.join(args.enrollment_dir, 'model', 'model_original.npy')
MODEL = np.load(speaker_model_path)

NumClasses = MODEL.shape[0]
NumFeatures = MODEL.shape[1]
score_vector = np.zeros((feature_vector.shape[0]*NumClasses, 1))
target_label_vector = np.zeros((feature_vector.shape[0]*NumClasses, 1))


for i in range(feature_vector.shape[0]):
    if i % 100 ==0:
        print("processing file %d from %d" %(i,feature_vector.shape[0]))
    for j in range(NumClasses):
        model = MODEL[j:j+1, :]
        feature = feature_vector[i:i + 1, :]

        # Unit norm
        feature = Normalizer(norm='l2').fit_transform(feature) #归一标准化
        score = cosine_similarity(feature, model)
        score_vector[i*NumClasses + j] = score
        # Class labels starts from 0 and not 1!
        if j == label_vector[i]:
            target_label_vector[i*NumClasses + j] = 1
        else:
            target_label_vector[i * NumClasses + j] = 0

# Save the score and label vector.
if not os.path.exists(args.evaluation_dir):
    os.makedirs(args.evaluation_dir)
np.save(os.path.join(args.evaluation_dir,'score_vector.npy'),score_vector)
np.save(os.path.join(args.evaluation_dir,'target_label_vector.npy'),target_label_vector)

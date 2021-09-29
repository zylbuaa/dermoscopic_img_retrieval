# -*- coding:utf-8 -*-
'''
@Updatingtime: 2021/9/29 11:41
@Author      : Yilan Zhang
@Filename    : train.py
@Email       : zhangyilan@buaa.edu.cn
'''

import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from collections import OrderedDict
from PIL import Image
import argparse
import os
import time
import numpy as np
import torch.nn.functional as F
from network import LossFunction
'''auxiliary function'''
# define function for saving model
def modelSnapShot(model, newModelPath, oldModelPath=None, onlyBestModel=False):
    if onlyBestModel and oldModelPath:
        os.remove(oldModelPath)

    stateDict = OrderedDict()
    for name, param in model.state_dict().items():
        if param.is_cuda:
            param = param.cpu()
        stateDict[name] = param
    torch.save(stateDict, newModelPath)

'''import model'''
from network import DH_ResNet18,Attention_ResNet18

'''generate model'''
model= Attention_ResNet18.resnet18(pretrained=False)

# device_ids = [0,1]
# model = torch.nn.DataParallel(model, device_ids=device_ids)
# model = model.cuda(device=device_ids[0])

# print model structure and parameters
print(model)
for name, param in model.named_parameters():
    print(name)
    print(param)

# save the initial model
state_dict = OrderedDict()
#for k, v in model.state_dict().items():
#    if v.is_cuda:
#        v = v.cpu()
#    state_dict[k] = v
# print(v.size())
#torch.save(state_dict, './initialModel.pth')

'''path of trainning log'''
trainLogPath="/"
'''set training parameters'''
dataPath="/"

parser = argparse.ArgumentParser(description='my code for classification task')
parser.add_argument('--classNum', default=8, help='number of classes')
parser.add_argument('--samplesNumOfTestset', default=[452,1287,332,262], help='number of classes')

parser.add_argument('--hashbit', default=16, help='hash bits')

parser.add_argument('--trainLogPath', default=trainLogPath, help='folder path of saving the train log and models')
parser.add_argument('--dataPath', default=dataPath, help='folder path of data')
parser.add_argument('--batchSize', type=int, default=64, help='batch size for training(default: 32)')
parser.add_argument('--weightDecay', type=float, default=0.0005, help='weight decay')

parser.add_argument('--gpu', default=0, help='index of gpus to use')
parser.add_argument('--gpuNum', type=int, default=1, help='number of gpus to use')
parser.add_argument('--seed', type=int, default=1, help='random seed(default: 1)')
parser.add_argument('--testEpochInterval', type=int, default=1,  help='how many epochs to wait before another test')

parser.add_argument('--learningRate', type=float, default=0.01, help='learning rate (default: 1e-2)')
parser.add_argument('--epochsOfDecreaseLearningRate', default='40,60,80,100', help='decreasing learning rate strategy')
parser.add_argument('--epochs', type=int, default=120, help='number of epochs to train (default: 100)')

args = parser.parse_args()

'''build a logger of training process and show it'''
print("=================Training Params==================")
for name, param in args.__dict__.items():
    print('{}: {}'.format(name, param))
print("==================================================")
File = open(trainLogPath+'trainingLog.txt', 'w')
File.write("=================Training Params=================="+"\n")
for name, param in args.__dict__.items():
    File.write('{}: {}'.format(name, param)+"\n")
File.write("=================================================="+"\n")

'''set random seed'''
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

'''get filename path list of training data and test data'''
trainDataFilenamePath=''
testDataFilenamePath=''
trainDataFilename=open(trainDataFilenamePath)
trainDataFilenameList=[]
for line in trainDataFilename:
    trainDataFilenameList.append(line.rstrip('\n'))
print('number of training samples:',len(trainDataFilenameList))

testDataFilename=open(testDataFilenamePath)
testDataFilenameList=[]
for line in testDataFilename:
    testDataFilenameList.append(line.rstrip('\n'))
print('number of test samples:',len(testDataFilenameList))


'''data augmentation'''
trainDataTransformation=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testDataTransformation=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


'''data loader'''
class trainDataLoader(data.Dataset):
    def __init__(self, dataFilenameList, transform=None):
        self.dataFilenameList = dataFilenameList
        self.transform = transform

    def __getitem__(self, index):
        path = self.dataFilenameList[index]
        data=Image.open(path)
        label = int(self.dataFilenameList[index].split('/')[-2])

        if self.transform is not None:
            data=self.transform(data)

        label = torch.LongTensor([label])

        return data,label

    def __len__(self):
        return len(self.dataFilenameList)

class testDataLoader(data.Dataset):
    def __init__(self, dataFilenameList, transform=None):
        self.dataFilenameList = dataFilenameList
        self.transform = transform

    def __getitem__(self, index):
        path = self.dataFilenameList[index]
        data=Image.open(path)
        label = int(self.dataFilenameList[index].split('/')[-2])
        if self.transform is not None:
            data=self.transform(data)

        label = torch.LongTensor([label])

        return data,label

    def __len__(self):
        return len(self.dataFilenameList)


'''load data'''
trainData = data.DataLoader(trainDataLoader(dataFilenameList=trainDataFilenameList, transform=trainDataTransformation), batch_size=args.batchSize,shuffle=True, num_workers=4)
testData = data.DataLoader(testDataLoader(dataFilenameList=testDataFilenameList,transform=testDataTransformation), batch_size=1,shuffle=False,num_workers=2)


# set cudnn.benchmark for classification or fixed input image
cudnn.benchmark = True
# use cuda
if args.cuda:
    model.cuda()

# set optimizer
optimizer = optim.SGD(model.parameters(), lr=args.learningRate, weight_decay=args.weightDecay, momentum=0.9)

# get the epochs of decrease learning rate
epochsOfDecreaseLearningRate = list(map(int, args.epochsOfDecreaseLearningRate.split(',')))

'''Start training'''
print('Start training...')
File.write('Start training...\n')
File.flush()
bestAverageAccuracy = 0
oldModelPath = None
startTime = time.time()

try:
    min_loss=0
    min_epoch=0
    for epoch in range(args.epochs):
        # train model
        model.train()
        if epoch in epochsOfDecreaseLearningRate:
            optimizer.param_groups[0]['lr'] *= 0.1

        startTimeOfEpoch=time.time()
        iter1 = iter(trainData)
        for batchIndex, (data, target) in enumerate(trainData):
            label = target.clone()
            target=torch.squeeze(target)

            if args.cuda:
                data, target = data.cuda(), target.cuda()
                label=label.cuda()
            data, target = Variable(data), Variable(target)
            label=Variable(label)

            optimizer.zero_grad()

            output,output_hash = model(data)

            #Weighted Cross Entropy Loss
            labels = target.cpu().numpy()
            weights = np.zeros(args.classNum)
            for i in range(np.shape(labels)[0]):
                weights[labels[i]] = weights[labels[i]] + 1
            weights = weights/np.sum(weights)
            weights = torch.FloatTensor(weights).cuda()

            loss_CE = F.cross_entropy(output, target, weight=weights)

            #CRI term
            x=data
            n = x.data.shape[3]-1

            #Rotate inputs
            list=[]
            for i in range(n,-1,-1):
                list.append(i)
            indices = np.array(list)
            x1 = x
            indices = Variable(torch.from_numpy(indices)).cuda()
            x1r90 = torch.transpose(torch.index_select(x1, 3, indices), 2, 3) #索引
            x1r180 = torch.transpose(torch.index_select(x1r90, 3, indices), 2, 3)
            x1r270 = torch.transpose(torch.index_select(x1r180, 3, indices), 2, 3)
            _,out1 = model(x1)
            _,out2 = model(x1r90)
            _,out3 = model(x1r180)
            _,out4 = model(x1r270)

            loss_rotinv =LossFunction.rotation_invariance_loss(out1,out2,out3,out4)

            loss = loss_CE + 0.5 * loss_rotinv #原本是0.1
            loss.backward()
            optimizer.step()

        print('Training epoch: {}, LR: {}, Loss: {:.6f}'.format(epoch, optimizer.param_groups[0]['lr'], loss.item()))
        File.write('Training epoch: {}, LR: {}, Loss: {:.6f}'.format(epoch,optimizer.param_groups[0]['lr'], loss.item())+"\n")

        # test model
        if epoch % args.testEpochInterval == 0:
            model.eval()
            testLoss = 0

            correctResultsOfEachClass = [0 for cols in range(args.classNum)]
            totalCorrectResultsNum = 0
            accuracyOfEachClass = [0.0 for cols in range(args.classNum)]
            accuracyOfTestset = 0.0

            for data, target in testData:
                target=torch.squeeze(target,1)
                label = int(target.cpu().numpy())
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                output,_ = model(data)

                testLoss += F.cross_entropy(output, target).item()

                predictedResult = output.data.max(1)[1]
                totalCorrectResultsNum += predictedResult.cpu().eq(label).sum()
                for category in range(args.classNum):
                    if label == category:
                        correctResultsOfEachClass[category] += predictedResult.cpu().eq(label).sum()

            testLoss = testLoss / len(testData)
            accuracyOfTestset = 100.0 * float(totalCorrectResultsNum) / len(testData)
            for category in range(args.classNum):
                accuracyOfEachClass[category] = 100.0 * float(correctResultsOfEachClass[category]) / args.samplesNumOfTestset[category]

            averageAccuracy = 0.0
            for category in range(args.classNum):
                averageAccuracy += accuracyOfEachClass[category]
            averageAccuracy/=args.classNum

            print('Test results: Average loss: {:.6f}, Accuracy: {:.6f}%, Average accuracy: {:.6f}%'
                  .format(testLoss, accuracyOfTestset, averageAccuracy))
            print('Accuracy of each class:[', end=" ")
            for category in range(args.classNum):
                print('{:.1f}%'.format(accuracyOfEachClass[category]), end=" ")
            print(']')

            File.write('Test results: Average loss: {:.6f}, Accuracy: {:.6f}%, Average accuracy: {:.6f}%'
                       .format(testLoss, accuracyOfTestset, averageAccuracy))
            File.write('\nAccuracy of each class:[')
            for category in range(args.classNum):
                File.write(' {:.1f}% '.format(accuracyOfEachClass[category]))
            File.write(']\n')
            File.flush()

        endTimeOfEpoch=time.time()

        trainingTimeOfEpoch = endTimeOfEpoch - startTimeOfEpoch
        totalTrainingTime = time.time() - startTime
        estimatedRemainingTime = trainingTimeOfEpoch * args.epochs - totalTrainingTime

        print("Total training time: {:.2f}s, {:.2f} s/epoch, Estimated remaining time: {:.2f}s".format(totalTrainingTime, trainingTimeOfEpoch, estimatedRemainingTime))
        File.write("Total training time: {:.2f}s, {:.2f} s/epoch, Estimated remaining time: {:.2f}s".format(totalTrainingTime, trainingTimeOfEpoch, estimatedRemainingTime)+"\n")
        File.flush()

        #Save model
        if averageAccuracy > bestAverageAccuracy:
            newModelPath = os.path.join(args.trainLogPath, 'bestmodel-{}.pth'.format(epoch))
            modelSnapShot(model, newModelPath, oldModelPath=oldModelPath, onlyBestModel=True)
            bestAverageAccuracy = averageAccuracy
            oldModelPath = newModelPath

        if epoch== 0 or loss < min_loss:
            min_loss = loss
            min_epoch=epoch
            modelSnapShot(model, os.path.join(args.trainLogPath, 'min_loss_model.pth'))

        modelSnapShot(model, os.path.join(args.trainLogPath, 'latest.pth'))

except Exception as e:
    import traceback
    traceback.print_exc()

finally:
    print('The min_loss epoch: {}, Min loss: {:.6f}'.format(min_epoch, min_loss))
    File.write('The min_loss epoch: {}, Min loss: {:.6f}'.format(min_epoch, min_loss)+"\n")
    print("Total training time: {:.2f}, Best Result: {:.1f}%".format(time.time()-startTime, bestAverageAccuracy))
    File.write("\nTotal training time: {:.2f}, Best Result: {:.1f}%".format(time.time()-startTime, bestAverageAccuracy)+"\n")
    File.close()
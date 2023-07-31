# -*- coding:utf-8 -*-
'''
@Updatingtime: 2021/9/29 15:26
@Author      : Yilan Zhang
@Filename    : HashRetrieval.py
@Email       : zhangyilan@buaa.edu.cn
'''

import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from PIL import Image
import argparse
import os
import time
import numpy as np

'''import model'''
from network import DH_ResNet18,Attention_ResNet18

'''generate model'''
model=Attention_ResNet18.resnet18(pretrained=False)

# print model structure and parameters
print(model)
for name, param in model.named_parameters():
    print(name)
    print(param)
cudnn.benchmark = True
model.cuda()
model.eval()

'''set test parameters'''
parser = argparse.ArgumentParser(description='my code for classification task')
parser.add_argument('--classNum', default=4, help='number of classes')
parser.add_argument('--samplesNumOfTestset', default=[452,1287,332,262], help='number of classes')

args = parser.parse_args()

# use gpu 0 as cuda device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

'''build a logger of testing process and show it'''
testLogPath = "/"
File=open(testLogPath+'TestingLog.txt','w')

'''get filename path list of training data and test data'''
trainDataFilenamePath = "/"
testDataFilenamePath = "/"

trainDataFilename = open(trainDataFilenamePath)
trainDataFilenameList = []
for line in trainDataFilename:
    trainDataFilenameList.append(line.rstrip('\n'))
print('number of training samples:',len(trainDataFilenameList))
File.write('number of training samples:{}'.format(len(trainDataFilenameList))+'\n')

testDataFilename=open(testDataFilenamePath)
testDataFilenameList=[]
for line in testDataFilename:
    testDataFilenameList.append(line.rstrip('\n'))
print('number of test samples:',len(testDataFilenameList))
File.write('number of test samples:{}'.format(len(testDataFilenameList))+'\n')


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

        return data, label, path

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

        return data, label, path

    def __len__(self):
        return len(self.dataFilenameList)


'''load data'''
trainData = data.DataLoader(trainDataLoader(dataFilenameList=trainDataFilenameList, transform=trainDataTransformation), batch_size=1,shuffle=False, num_workers=4)
testData = data.DataLoader(testDataLoader(dataFilenameList=testDataFilenameList,transform=testDataTransformation), batch_size=1,shuffle=False,num_workers=2)


'''build hashtable & continuous feature database'''
print('Build hashtable and continuous feature database...')
hashTable = {}
featureDatabase = {}

# 0.5 for deephash, 0.0 for others
threshhold = 0.5
if threshhold == 0.5:
    print('for deep hash')

for data, target, path in trainData:
    target = torch.squeeze(target, 1)
    data, target = data.cuda(), target.cuda()
    data, target = Variable(data), Variable(target)

    _,continuousHashCode = model(data)


    feature = continuousHashCode

    if not hashTable:
        print("The dimension of the hash key:", continuousHashCode.shape[1])
        File.write("The dimension of the hash key:{}".format(continuousHashCode.shape[1])+'\n')
        print("The dimension of the feature:", feature.shape[1])
        File.write("The dimension of the feature:{}".format(feature.shape[1])+'\n')
        File.flush()


    # continuous feature database
    featureDatabase[path] = [feature.cpu().detach().numpy()]

    # hash table
    hashCode=(torch.sign(continuousHashCode - threshhold)).cpu().detach().numpy()
    hashKey=tuple(hashCode[0])

    if hashKey not in hashTable.keys():
        hashTable[hashKey]=[path]
    else:
        hashTable[hashKey].append(path)

'''test image retrieval'''
print('Strat testing image retrieval...')


topNumOfRetrievalResults = 10

#mAP
retrievalResultsAccuracyOfEachSampleOfDifferentClass={}
for category in range(args.classNum):
    retrievalResultsAccuracyOfEachSampleOfDifferentClass[category]=[]
numOfDoneImages=0

# mRR
ReciprocalRankofEachSampleofDiffentCalss={}
for category in range(args.classNum):
    ReciprocalRankofEachSampleofDiffentCalss[category]=[]

startTime = time.time()

for data, target, path in testData:
    target = torch.squeeze(target, 1)
    label = int(target.cpu().numpy())
    data, target = data.cuda(), target.cuda()
    data, target = Variable(data), Variable(target)

    _,continuousHashCode=model(data)

    # get feature
    feature=continuousHashCode.cpu().detach().numpy()

    # get hashcode
    hashCode=torch.sign(continuousHashCode - threshhold).cpu().detach().numpy()
    hashKey = tuple(hashCode[0])

    rawRetrievalResult={}
    for keyValue in hashTable.keys():
        hammingDistance = 0
        for i in range(0, len(hashKey)):
            if hashKey[i] != keyValue[i]:
                hammingDistance = hammingDistance + 1
        if hammingDistance <=4:
            for imgpath in hashTable[keyValue]:
                # for Euclidean distance
                # rawRetrievalResult[imgpath]=np.sqrt(np.sum(np.square(feature - featureDatabase[imgpath])))
                # for Cosine distance
                rawRetrievalResult[imgpath] = 1.0 - np.dot(feature[0],featureDatabase[imgpath][0][0])/(np.linalg.norm(feature[0],ord=2)*np.linalg.norm(featureDatabase[imgpath][0][0],ord=2))
                # for Manhattan distance
                # rawRetrievalResult[imgpath] = np.sum(np.linalg.norm(feature[0] - featureDatabase[imgpath][0][0], ord=1))

    preciseRetrievalResult = zip(rawRetrievalResult.values(), rawRetrievalResult.keys())
    preciseRetrievalResult = sorted(preciseRetrievalResult) #默认为距离升序

    correctRetrievalSamplesOfEachResult = 0
    num=0
    firstcorrectresult=0
    for euclideanDistance, imgpath in list(preciseRetrievalResult)[0:topNumOfRetrievalResults]:
        imgpath = ''.join(str(i) for i in imgpath)
        resultLabel = int(imgpath.split('/')[-2])
        num += 1
        if label == resultLabel:
            correctRetrievalSamplesOfEachResult = correctRetrievalSamplesOfEachResult + 1
            if firstcorrectresult==0:
                firstcorrectresult=1/num
            else:
                continue


    #这里加一些评价指标
    if len(preciseRetrievalResult) == 0:
        accuracy = 0
    elif len(preciseRetrievalResult) < topNumOfRetrievalResults:
        accuracy=correctRetrievalSamplesOfEachResult * 1.0 / len(preciseRetrievalResult)
    else:
        accuracy=correctRetrievalSamplesOfEachResult * 1.0 / topNumOfRetrievalResults


    #mAP
    retrievalResultsAccuracyOfEachSampleOfDifferentClass[label].append(accuracy)
    #mRR
    ReciprocalRankofEachSampleofDiffentCalss[label].append(firstcorrectresult)

    # show percent of done images
    numOfDoneImages+=1
    print("\r",'{:.2f}% done'.format(100*numOfDoneImages/len(testData)),end="",flush=True)
print("\n")

totalTestingTime = time.time() - startTime

'''test result of image retrieval'''
averagePrecision=[0.0 for cols in range(args.classNum)]
averageRR=[0.0 for cols in range(args.classNum)]


for category in range(args.classNum):
    averagePrecision[category]=np.mean(retrievalResultsAccuracyOfEachSampleOfDifferentClass[category])
    averageRR[category]=np.mean(ReciprocalRankofEachSampleofDiffentCalss[category])
    print('AP of Class '+str(category)+': '+str(averagePrecision[category]))
    File.write('AP of Class {}:{}'.format(str(category),str(averagePrecision[category]))+'\n')

    print('RR of Class '+str(category)+': '+str(averageRR[category]))
    File.write('RR of Class {}:{}'.format(str(category),str(averageRR[category]))+'\n')


print('mAP of all classes: '+str(np.mean(averagePrecision)))
File.write('mAP of all classes:{} '.format(str(np.mean(averagePrecision)))+'\n')
print('mRR of all classes: '+str(np.mean(averageRR)))
File.write('mRR of all classes:{} '.format(str(np.mean(averageRR)))+'\n')

perTestingTime = totalTestingTime / len(testDataFilenameList)
print("Per testing time: ", perTestingTime)
File.write("Per testing time:{}".format(perTestingTime))
File.close()
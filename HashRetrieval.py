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
# from DeepHashModel import Resnet_hash

'''import my moduel'''

from Resnet_MySpatialAttention import resnet34
# from AttentionModel.Resnet_SEnet import resnet18
# from AttentionModel.Resnet_CA import resnet18
# from AttentionModel.Resnet_CBAM2 import resnet18
# from AttentionModel.Resnet_BAM import resnet18


'''generate model'''
# model=Resnet_hash.resnet34(pretrained=True)
model=resnet34(pretrained=True)
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
# for old data
# parser.add_argument('--samplesNumOfTestset', default=[144,370,175,375,270,86,39,143], help='number of classes')
args = parser.parse_args()

# use gpu 0 as cuda device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

'''build a logger of testing process and show it'''
testLogPath="/home/zyl/zyl/Experiment_result/MIA/Attention/HDSCA/ResNet34_2/"
File=open(testLogPath+'testingLog_5.txt','w')


'''get filename path list of training data and test data'''
trainDataFilenamePath="/home/zyl/zyl/data/ISIC2019_morebalanced_224/train/DataPath.txt"
testDataFilenamePath="/home/zyl/zyl/data/ISIC2019_morebalanced_224/test/DataPath.txt"
# trainDataFilenamePath='/home/zyl/zyl/data/224_cropped_5/train/DataPath.txt'
# testDataFilenamePath='/home/zyl/zyl/data/224_cropped_5/test/DataPath.txt'

trainDataFilename=open(trainDataFilenamePath)
trainDataFilenameList=[]
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
    # transforms.RandomCrop(224),#for ResNet,VGG
    # transforms.RandomCrop(299),#for InceptionV3
    # transforms.Resize(224),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testDataTransformation=transforms.Compose([
    # transforms.CenterCrop(224),
    # transforms.CenterCrop(299),
    # transforms.Resize(224),
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


# for LSH, ITQ, SDH
hashcodeDim = 16

# for building LSH hashing
# gaussianSample = LSH.LSH_Gaussian(512,hashcodeDim)
# print("The LSH building is done.")

# for training ITQ
# train_data=np.zeros((len(trainDataFilenameList), 512))
# i=0
# for data, target, path in trainData:
#    data = data.cuda()
#    data = Variable(data)
#    feature = model(data).cpu().detach().numpy()
#    train_data[i]=feature
#    i=i+1
# train_data_mean, train_data_std, R, pca = ITQ.train(train_data, hashcodeDim)
# print("The ITQ training is done.")

# for training SDH, SIQDH
# train_data=np.zeros((len(trainDataFilenameList), 512))
# train_targets=np.zeros((len(trainDataFilenameList), 1))
# i=0
# for data, target, path in trainData:
#    data = data.cuda()
#    data = Variable(data)
#    feature = model(data).cpu().detach().numpy()
#    train_data[i]=feature
#    train_targets[i]=target
#    i=i+1
# train_data_mean, train_data_std, anchor, P = SDH.train(train_data, train_targets, hashcodeDim)
# print("The SDH training is done.")
# train_data_mean, train_data_std, anchor, P, R = SIQDH.train(train_data, train_targets, hashcodeDim)
# print("The SIQDH training is done.")

'''build hashtable & continuous feature database'''
print('Build hashtable and continuous feature database...')
hashTable={}
featureDatabase={}

# 0.5 for deephash, 0.0 for others
threshhold = 0.5
if threshhold == 0.5:
    print('for deep hash')

for data, target, path in trainData:
    target = torch.squeeze(target, 1)
    data, target = data.cuda(), target.cuda()
    data, target = Variable(data), Variable(target)
    _,continuousHashCode = model(data)
    # continuousHashCode=torch.tanh(continuousHashCode)
    # for LSH
    # feature = model(data)
    # continuousHashCode = np.dot(feature.cpu().detach().numpy(),gaussianSample.transpose())

    # for ITQ
    # feature = model(data).cpu().detach().numpy()
    # continuousHashCode = ITQ.generate_code(feature, train_data_mean, train_data_std, R, pca)

    # for SDH, SIQDH
    # feature = model(data).cpu().detach().numpy()
    # continuousHashCode = SDH.generate_code(feature, train_data_mean, train_data_std, anchor, P)
    # continuousHashCode = SIQDH.generate_code(feature, train_data_mean, train_data_std, anchor, P, R)

    feature = continuousHashCode

    if not hashTable:
        print("The dimension of the hash key:", continuousHashCode.shape[1])
        File.write("The dimension of the hash key:{}".format(continuousHashCode.shape[1])+'\n')
        print("The dimension of the feature:", feature.shape[1])
        File.write("The dimension of the feature:{}".format(feature.shape[1])+'\n')
        File.flush()


    # continuous feature database
    featureDatabase[path] = [feature.cpu().detach().numpy()]
    # featureDatabase[path] = feature

    # hash table
    hashCode=(torch.sign(continuousHashCode - threshhold)).cpu().detach().numpy()
    # hashCode=np.sign(continuousHashCode - threshhold)
    hashKey=tuple(hashCode[0])
    #print(hashKey)

    if hashKey not in hashTable.keys():
        hashTable[hashKey]=[path]
    else:
        hashTable[hashKey].append(path)

# print(hashTable)
# print(len(featureDatabase.keys()))


'''test image retrieval'''
print('Strat testing image retrieval...')

topNumOfRetrievalResults = 10#10
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
    # torch.cuda.empty_cache()
    target = torch.squeeze(target, 1)
    label = int(target.cpu().numpy())
    data, target = data.cuda(), target.cuda()
    data, target = Variable(data), Variable(target)
    # print(data)
    _,continuousHashCode=model(data)
    # continuousHashCode=torch.tanh(continuousHashCode)

    # for LSH
    # feature = model(data)
    # continuousHashCode = np.dot(feature.cpu().detach().numpy(),gaussianSample.transpose())

    # for ITQ
    # feature = model(data).cpu().detach().numpy()
    # continuousHashCode = ITQ.generate_code(feature, train_data_mean, train_data_std, R, pca)

    # for SDH, SIQDH
    # feature = model(data).cpu().detach().numpy()
    # continuousHashCode = SDH.generate_code(feature, train_data_mean, train_data_std, anchor, P)
    # continuousHashCode = SIQDH.generate_code(feature, train_data_mean, train_data_std, anchor, P, R)

    # get feature
    feature=continuousHashCode.cpu().detach().numpy()
    # feature = continuousHashCode

    # get hashcode
    hashCode=torch.sign(continuousHashCode - threshhold).cpu().detach().numpy()
    # hashCode=np.sign(continuousHashCode - threshhold)
    hashKey = tuple(hashCode[0])
    # print(hashkey)

    rawRetrievalResult={}
    for keyValue in hashTable.keys():
        hammingDistance = 0
        for i in range(0, len(hashKey)):
            if hashKey[i] != keyValue[i]:
                hammingDistance = hammingDistance + 1
        if hammingDistance <=4:
            # rawResult = hashTable[keyvalue]
            ##print(label)
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
        #print(euclideanDistance,imgpath)
        # print(len(preciseRetrievalResult))
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

    # if len(preciseRetrievalResult) >= 10:
    #     accuracy = correctRetrievalSamplesOfEachResult * 1.0 / topNumOfRetrievalResults



    #mPA
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
    print('average precision of Class '+str(category)+': '+str(averagePrecision[category]))
    File.write('average precision of Class {}:{}'.format(str(category),str(averagePrecision[category]))+'\n')

    print('RR of Class '+str(category)+': '+str(averageRR[category]))
    File.write('RR of Class {}:{}'.format(str(category),str(averageRR[category]))+'\n')
    #print(str(averagePrecision[category]))

#PA
PA=0
totalnum=0
for category in range(args.classNum):
    PA+=averagePrecision[category]*int(args.samplesNumOfTestset[category])
    totalnum+=args.samplesNumOfTestset[category]
PA=PA/totalnum

print('PA : '+str(PA))
File.write('PA : '.format(str(PA))+'\n')

print('mAP of all classes: '+str(np.mean(averagePrecision)))
File.write('mAP of all classes:{} '.format(str(np.mean(averagePrecision)))+'\n')
print('mRR of all classes: '+str(np.mean(averageRR)))
File.write('mRR of all classes:{} '.format(str(np.mean(averageRR)))+'\n')
#print(str(np.mean(averagePrecision)))

perTestingTime = totalTestingTime / len(testDataFilenameList)
print("Per testing time: ", perTestingTime)
File.write("Per testing time:{}".format(perTestingTime))
File.close()
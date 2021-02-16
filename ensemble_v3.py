import numpy as np 
import pandas as pd
import json
from PIL import Image
import os
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
import torch.optim as optim
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet

#Get Data
BATCH = 16
EPOCHS = 20

LR = 0.0001
IM_SIZE = 256

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = './cassava-leaf-disease-classification/train_images/'
TEST_DIR = './cassava-leaf-disease-classification/test_images/'
labels = json.load(open("./cassava-leaf-disease-classification/label_num_to_disease_map.json"))
train = pd.read_csv('./cassava-leaf-disease-classification/train.csv')


X_Train, Y_Train = train['image_id'].values, train['label'].values
X_Test = [name for name in (os.listdir(TEST_DIR))]

class GetData(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.lbs = Labels
        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        x = Image.open(os.path.join(self.dir, self.fnames[index]))
        if "train" in self.dir:            
            return self.transform(x), self.lbs[index]            
        elif "test" in self.dir:            
            return self.transform(x), self.fnames[index]

Transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((IM_SIZE, IM_SIZE)),
     transforms.RandomRotation(90),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#create trainset and validation set
train_set = GetData(TRAIN_DIR, X_Train, Y_Train, Transform)

sample_num = len(train_set)
train1_size_ind = int(sample_num * 0.45)
train2_size_ind = int(sample_num * 0.9)


subset1_indices = list(range(0, train1_size_ind))
subset2_indices = list(range(train1_size_ind, train2_size_ind))
subset3_indices = list(range(train2_size_ind, sample_num))

trainset1 = Subset(train_set, subset1_indices)
trainset2 = Subset(train_set, subset2_indices)
valset = Subset(train_set, subset3_indices)

trainloader1 = DataLoader(trainset1, batch_size=BATCH, shuffle=True, num_workers=16)
trainloader2 = DataLoader(trainset2, batch_size=BATCH, shuffle=True, num_workers=16)
valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=16)

testset = GetData(TEST_DIR, X_Test, None, Transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=16)

#define neural network models
class Cnn_2(nn.Module):
    def __init__(self):
        super(Cnn_2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128, 128, 3)
        self.conv6 = nn.Conv2d(128, 256, 4)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv7 = nn.Conv2d(256, 256, 3)
        self.conv8 = nn.Conv2d(256, 512, 3)
        self.pool4 = nn.MaxPool2d(2, 2)


        self.fc1 = nn.Linear(512 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.batch_norm1(x)
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.batch_norm2(x)
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.batch_norm3(x)
        x = self.pool3(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool4(x)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

cnn_2_1 = Cnn_2()
cnn_2_2 = Cnn_2()
resnext_1 = torchvision.models.resnext50_32x4d()
resnext_1.fc = nn.Linear(2048, 5, bias=True)
resnext_2 = torchvision.models.resnext50_32x4d()
resnext_2.fc = nn.Linear(2048, 5, bias=True)
efficient_net_1 = EfficientNet.from_name('efficientnet-b3')
efficient_net_1.fc = nn.Linear(2048, 5, bias=True)
efficient_net_2 = EfficientNet.from_name('efficientnet-b3')
efficient_net_2.fc = nn.Linear(2048, 5, bias=True)

#load pretrained models
model_path = 'cnn_2_v3_1.pth'
cnn_2_1.load_state_dict(torch.load(model_path))
model_path = 'cnn_2_v3_2.pth'
cnn_2_2.load_state_dict(torch.load(model_path))

model_path = 'resnext_v3_1.pth'
resnext_1.load_state_dict(torch.load(model_path))
model_path = 'resnext_v3_2.pth'
resnext_2.load_state_dict(torch.load(model_path))

model_path = 'efficient_net_v3_1.pth'
efficient_net_1.load_state_dict(torch.load(model_path))
model_path = 'efficient_net_v3_2.pth'
efficient_net_2.load_state_dict(torch.load(model_path))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#infer
cnn_2_1.to(device)
inf_cnn_2_1 = []
cnn_2_2.to(device)
inf_cnn_2_2 = []

resnext_1.to(device)
inf_resnext_1 = []
resnext_2.to(device)
inf_resnext_2 = []

efficient_net_1.to(device)
inf_efficient_net_1 = []
efficient_net_2.to(device)
inf_efficient_net_2 = []

false_pred = [0,0,0,0,0]
num_correct = 0
num_wrong = 0
no_correct_ans = 0
inference = []
inf_len_1 = len(inf_resnext_1)
inf_len_2 = len(inf_resnext_2)

with torch.no_grad():
    cnn_2_1.eval()
    resnext_1.eval()
    efficient_net_1.eval()
    cnn_2_2.eval()
    resnext_2.eval()
    efficient_net_2.eval()
    for image, ans in tqdm(valloader): 
        #predict using cnn_2_1
        image = image.to(device)
        logits = cnn_2_1(image)        
        ps = torch.exp(logits)        
        _, top_class = ps.topk(1, dim=1)
        
        for pred in top_class:
            inf_cnn_2_1.append(pred.item())
        

        #predict using cnn_2_2
        image = image.to(device)
        logits = cnn_2_2(image)        
        ps = torch.exp(logits)        
        _, top_class = ps.topk(1, dim=1)
        
        for pred in top_class:
            inf_cnn_2_2.append(pred.item())
        

        #predict using resnext_1
        logits = resnext_1(image)        
        ps = torch.exp(logits)        
        _, top_class = ps.topk(1, dim=1)
        
        for pred in top_class:
            inf_resnext_1.append(pred.item())

        
        #predict using resnext_2
        logits = resnext_2(image)        
        ps = torch.exp(logits)        
        _, top_class = ps.topk(1, dim=1)
        
        for pred in top_class:
            inf_resnext_2.append(pred.item())


        #predict using efficient_net_1
        logits = efficient_net_1(image)        
        ps = torch.exp(logits)        
        _, top_class = ps.topk(1, dim=1)
        
        for pred in top_class:
            inf_efficient_net_1.append(pred.item())


        #predict using efficient_net_2
        logits = efficient_net_2(image)        
        ps = torch.exp(logits)        
        _, top_class = ps.topk(1, dim=1)
        
        for pred in top_class:
            inf_efficient_net_2.append(pred.item())


        #ensemble
        infs = []
        infs = [inf_cnn_2_1[-1], inf_cnn_2_2[-1], inf_resnext_1[-1], inf_resnext_2[-1], inf_efficient_net_1[-1], inf_efficient_net_2[-1]]
        inference.append(statistics.mode(infs))

        s = 0
        for i in range(len(infs)):
            if infs[i] == ans.item():
                s += 1
        if s == 0:
            no_correct_ans += 1


        if inference[-1] == ans.item():
            num_correct += 1
        else:
            num_wrong += 1
            if ans.item() == 0:
                false_pred[0] += 1
            elif ans.item() == 1:
                false_pred[1] += 1
            elif ans.item() == 2:
                false_pred[2] += 1
            elif ans.item() == 3:
                false_pred[3] += 1
            elif ans.item() == 4:
                false_pred[4] += 1


print("Accuracy:" + str(float(num_correct)/ (float(num_correct) + float(num_wrong))))
print("no correct answer in any model:" + str(float(no_correct_ans)/(float(num_wrong))))
print("wrongly classified 0:" + str(float(false_pred[0])/float(num_wrong)))
print("wrongly classified 1:" + str(float(false_pred[1])/float(num_wrong)))
print("wrongly classified 2:" + str(float(false_pred[2])/float(num_wrong)))
print("wrongly classified 3:" + str(float(false_pred[3])/float(num_wrong)))
print("wrongly classified 4:" + str(float(false_pred[4])/float(num_wrong)))

        

 
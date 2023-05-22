import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler,Sampler
import numpy as np
import os
from collections import Counter
############param init############
BATCH_SIZE = 16
# 1S = 30 FPS
data_path=r'F:\WZU\shy\智能运动\coding\model\lstm_ac\train\22_9_30\train\total_data\train_no_h5.npy'
test_data_path=r'F:\WZU\shy\智能运动\coding\model\lstm_ac\train\22_9_30\train\total_data\test_h5.npy'
############param init############

class Data(Dataset):
    def __init__(self, data):
        self.data=data

    def __getitem__(self, idx):
        return self.data[idx][0],self.data[idx][1]

    def __len__(self):
        return len(self.data)

##############样本均衡################
data=np.load(data_path,allow_pickle=True)

labels=data[:,-1]
num=Counter(labels)

data_len=len(labels)
sampler_list=[data_len/num.get(label) for label in labels]

sampler = WeightedRandomSampler(sampler_list,len(sampler_list),True)

train_data=Data(data)
train_loader=DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
##############样本均衡################

############### 生成测试集 loader ###########
test_data=np.load(test_data_path,allow_pickle=True)
test_labels=test_data[:,-1]
test_num=Counter(test_labels)

test_data_len=len(test_labels)
test_sampler_list=[test_data_len/num.get(label) for label in test_labels]

test_sampler = WeightedRandomSampler(test_sampler_list,len(test_sampler_list)//10,True)

test_data=Data(test_data)
test_loader=DataLoader(test_data,batch_size=BATCH_SIZE,sampler=test_sampler,drop_last=True)



if __name__ == '__main__':
    pass




















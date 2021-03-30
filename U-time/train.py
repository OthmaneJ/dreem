



import h5py
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import random
import sys 
import pickle 
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
#dataloading 
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
# optimizer
from torch.optim import Adam
from sklearn.metrics import f1_score , precision_recall_fscore_support, confusion_matrix
import torch.nn.functional as F
from seaborn import heatmap
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss
import numpy as np
from torch.autograd import Variable

from sklearn.preprocessing import normalize

import torch.nn as nn
from torch.nn import Linear, RNN, LSTM, GRU
import torch.nn.functional as F
from torch.nn.functional import softmax, relu
from torch.autograd import Variable

from torch.optim import lr_scheduler

# from utils import make_loader

from model_multi import UTime

n_epochs = 5
num_channels = 8



def normalize_channels(X):
    for i in range(8):
        X[:,9000*i:9000*(i+1)] = normalize(X[:,9000*i:9000*(i+1)],norm='l2')
    return X


    
def make_loader(X,y,batch_size=8,shuffle=True):
  X = torch.Tensor(X).reshape((X.shape[0],num_channels,-1))
  X = torch.Tensor(X)
  print(X.shape)
  print(y.shape)
  y = torch.Tensor(y).unsqueeze(1)
  data = TensorDataset(X,y)
  dataloader = DataLoader(data,batch_size=batch_size,shuffle=shuffle)
  return dataloader



def train(epoch):
  utime.train()
  losses=[]
  
  with tqdm(train_loader,unit="batch") as tepoch:
    for data,target in tepoch:
        optimizer.zero_grad()
        tepoch.set_description(f'epoch {epoch}')
        output = utime(data.to(device))
        # loss = loss_fct(output,target.long().to(device),binary=True)
        loss = loss_fct(output,target.to(device))
        # loss = loss_fct(output,target.long().to(device))
        # loss = loss_fct(output.unsqueeze(2),target.long().to(device))
        loss.backward()
        # losses.append(loss.item())
        optimizer.step()
        tepoch.set_postfix(loss = loss.item())
        del data
        del target


def eval():
    utime.eval()
    losses=[]
    out = []
    y_target = []

    with tqdm(val_loader) as tepoch:
        for data,target in tepoch:
            tepoch.set_description('evaluation')
            output = utime(data.to(device))
            # loss = loss_fct(output,target.long().to(device),binary=True)
            loss = loss_fct(output.view((-1,90,1)),target.to(device).view((-1,90,1)))
            # print(output.shape,target.shape)
            # loss = loss_fct(output,target.long().to(device))
            # loss = loss_fct(output.unsqueeze(2),target.long().to(device))
            losses.append(loss.item())
            out.append(output)
            
            y_target.append(target.long().numpy())
            tepoch.set_postfix(loss = np.average(losses))

            del loss
            del data
            del target
    
        out = torch.cat(out,dim=0).squeeze(1)
        # print(out.shape,y_val.shape)
        out = 1*(out.cpu().detach().numpy()>=0)
        # out = out.cpu().argmax(axis=1)
        # print('f1 score',f1_score(y_val,out,average='micro'))
        tepoch.set_postfix(loss=np.average(losses),f1_score =f1_score(y_val.flatten(),out.flatten(),average='binary'))
        # tepoch.set_postfix(loss=np.average(losses),f1_score =f1_score(y_val,out,average='binary'))
        return out,y_target
        





if __name__ == "__main__":

    PATH_TO_TRAINING_DATA = "/content/drive/MyDrive/dreem_files/X_train.h5"
    PATH_TO_TRAINING_TARGET = "/content/drive/MyDrive/dreem/y_train.csv"
    h5_file = h5py.File(PATH_TO_TRAINING_DATA,'r')
    
    X = h5_file['data'][:,2:]
    y = pd.read_csv(PATH_TO_TRAINING_TARGET,index_col=0).to_numpy()

    # normalizing per channel
    for i in range(8):
        X[:,9000*i:9000*(i+1)] = normalize(X[:,9000*i:9000*(i+1)],norm='l2')

    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.05)


    train_loader = make_loader(X_train,y_train)
    val_loader = make_loader(X_val,y_val)



    utime = UTime()
    device = torch.device('cuda')
    utime.to(device)
    
    optimizer=Adam(utime.parameters(),lr=1e-2,)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.8, min_lr=1e-8) # Using ReduceLROnPlateau schedule
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.8, min_lr=1e-8) 

    loss_fct = nn.BCEWithLogitsLoss(pos_weight = torch.Tensor([2]).cuda())

    eval()
    for epoch in range(1,n_epochs+1):
        train(epoch)
        out,y_target = eval()
        y_target = np.concatenate(y_target).squeeze(axis=1)
        print('precision {0}, recall {1}, f1 score {2}'.format(*precision_recall_fscore_support(out.flatten(),y_target.flatten(),average='binary')))
        # print(f1_score(out.flatten(),y_target.flatten()))
        scheduler.step(f1_score(out.flatten(),y_target.flatten()))
        


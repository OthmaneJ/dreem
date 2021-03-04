

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss
import numpy as np
from torch.autograd import Variable


num_channels = 8
num_classes = 2



class EncoderLayer(nn.Module):

  def __init__(self,maxpool=10,cf=1):
    super(EncoderLayer,self).__init__()
    self.conv1 = nn.Conv1d(in_channels=num_channels*cf,out_channels=num_channels*cf*2,kernel_size=5,padding=2)
    self.batch_norm1 = nn.BatchNorm1d(num_channels*cf*2)
    self.conv2 = nn.Conv1d(in_channels=num_channels*cf*2,out_channels=num_channels*cf*2,kernel_size=5,padding=2)
    self.batch_norm2 = nn.BatchNorm1d(num_channels*cf*2)
    self.maxpool = nn.MaxPool1d(maxpool)
    self.relu = nn.ReLU()
  
  def forward(self,input):
    output = self.conv1(input)
    output = self.relu(self.batch_norm1(output))
    output = self.conv2(output)
    output = self.relu(self.batch_norm2(output))
    output_features = output
    output = self.maxpool(output)
    return output,output_features




class DecoderLayer(nn.Module):

  def __init__(self,kernel_size=10,cf=16):
    super(DecoderLayer,self).__init__()
    self.upsample = nn.Upsample(scale_factor=kernel_size)
    # self.conv1 = nn.Conv1d(in_channels=num_channels*cf,out_channels=num_channels*(cf//2),kernel_size=kernel_size,padding=(kernel_size-1)//2)
    # self.batch_norm1 = nn.BatchNorm1d(num_channels*(cf//2))
    # self.conv2 = nn.Conv1d(in_channels=num_channels*(cf//2),out_channels=num_channels*(cf//2),kernel_size=kernel_size,padding=(kernel_size+1)//2)
    # self.batch_norm2 = nn.BatchNorm1d(num_channels*(cf//2))
    self.conv3 = nn.Conv1d(in_channels=num_channels*cf,out_channels=num_channels*(cf//2),kernel_size=kernel_size,padding=(kernel_size-1)//2)
    self.batch_norm3 = nn.BatchNorm1d(num_channels*(cf//2))
    self.conv4 = nn.Conv1d(in_channels=num_channels*(cf//2),out_channels=num_channels*(cf//4),kernel_size=kernel_size,padding=(kernel_size+1)//2)
    self.batch_norm4 = nn.BatchNorm1d(num_channels*(cf//4))
    self.relu = nn.ReLU()

  def forward(self,input,encoder_output):
    output = self.upsample(input)
    # output = self.relu(self.conv1(output))
    # output = self.batch_norm1(output)
    # output = self.relu(self.conv2(output))
    # output = self.batch_norm2(output)

    # print(output.shape)

    diff = encoder_output.shape[2] - output.shape[2]
    # print(diff)
    # encoder_output = encoder_output.narrow(2,diff,output.shape[2])
    output = nn.functional.pad(output,(diff//2,diff//2))
    # print(output.shape,encoder_output.shape)

    output = torch.cat((output,encoder_output),dim=1)

    # print(output.shape)

    output = self.conv3(output)
    output = self.relu(self.batch_norm3(output))
    output = self.conv4(output)
    output = self.relu(self.batch_norm4(output))

    # print(output.shape)

    return output





class Bridge(nn.Module):

    def __init__(self,cf=16):
        super(Bridge,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_channels*cf,out_channels=num_channels*cf,kernel_size=5,padding=2)
        self.batch_norm1 = nn.BatchNorm1d(num_channels*cf)
        self.conv2 = nn.Conv1d(in_channels=num_channels*cf,out_channels=num_channels*cf,kernel_size=5,padding=2)
        self.batch_norm2 = nn.BatchNorm1d(num_channels*cf)
        self.relu = nn.ReLU()
    
    def forward(self,input):
        output = self.conv1(input)
        output = self.relu(self.batch_norm1(output))
        output = self.conv2(output)
        output = self.relu(self.batch_norm2(output))
        
        return output



class UTime(nn.Module):
  
  def __init__(self):
    super(UTime,self).__init__()
    self.encoder1 = EncoderLayer(10,cf=1)
    self.encoder2 = EncoderLayer(8,cf=2)
    self.encoder3 = EncoderLayer(6,cf=4)
    self.encoder4 = EncoderLayer(4,cf=8)

    self.decoder1 = DecoderLayer(10,cf=4)
    self.decoder2 = DecoderLayer(8,cf=8)
    self.decoder3 = DecoderLayer(6,cf=16)
    self.decoder4 = DecoderLayer(4,cf=32)

    self.bridge = Bridge()

    #segment classifier
    self.avg_pool = nn.AvgPool1d(kernel_size=100,stride=100)
    self.conv3 = nn.Conv1d(in_channels=num_channels,out_channels=num_classes-1,kernel_size=1)
    
    self.relu = nn.ReLU()

  def forward(self,input):
    output,output_features1 = self.encoder1(input)
    output,output_features2 = self.encoder2(output)
    output,output_features3 = self.encoder3(output)
    output,output_features4 = self.encoder4(output)

    # print(output.shape)

    output = self.bridge(output)

    # print(output.shape)

    output = self.decoder4(output,output_features4)
    output = self.decoder3(output,output_features3)
    output = self.decoder2(output,output_features2)
    output = self.decoder1(output,output_features1)

    output = self.avg_pool(output)
    output = self.conv3(output)

    return output



# def train(epoch):
#   utime.train()
#   losses=[]
  
#   with tqdm(train_loader,unit="batch") as tepoch:
#     for data,target in tepoch:
#         optimizer.zero_grad()
#         tepoch.set_description(f'epoch {epoch}')
#         output = utime(data.to(device))
#         # loss = loss_fct(output,target.long().to(device),binary=True)
#         loss = loss_fct(output,target.to(device))
#         # loss = loss_fct(output,target.long().to(device))
#         # loss = loss_fct(output.unsqueeze(2),target.long().to(device))
#         loss.backward()
#         # losses.append(loss.item())
#         optimizer.step()
#         tepoch.set_postfix(loss = loss.item())
#         del data
#         del target




# def eval():
#     utime.eval()
#     losses=[]
#     out = []
#     y_target = []

#     with tqdm(val_loader) as tepoch:
#         for data,target in tepoch:
#             tepoch.set_description('evaluation')
#             output = utime(data.to(device))
#             # loss = loss_fct(output,target.long().to(device),binary=True)
#             loss = loss_fct(output.view((-1,90,1)),target.to(device).view((-1,90,1)))
#             # print(output.shape,target.shape)
#             # loss = loss_fct(output,target.long().to(device))
#             # loss = loss_fct(output.unsqueeze(2),target.long().to(device))
#             losses.append(loss.item())
#             out.append(output)
            
#             y_target.append(target.long().numpy())
#             tepoch.set_postfix(loss = np.average(losses))

#             del loss
#             del data
#             del target
    
#         out = torch.cat(out,dim=0).squeeze(1)
#         # print(out.shape,y_val.shape)
#         out = 1*(out.cpu().detach().numpy()>=0)
#         # out = out.cpu().argmax(axis=1)
#         # print('f1 score',f1_score(y_val,out,average='micro'))
#         tepoch.set_postfix(loss=np.average(losses),f1_score =f1_score(y_val.flatten(),out.flatten(),average='binary'))
#         # tepoch.set_postfix(loss=np.average(losses),f1_score =f1_score(y_val,out,average='binary'))
#         return out,y_target

    

# train and validate 

# n_epochs = 50

# utime = UTime()
# device = torch.device('cuda')
# utime.to(device)
# print(' ')

# optimizer=Adam(utime.parameters(),lr=1e-4,)
# # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.8, min_lr=1e-8) # Using ReduceLROnPlateau schedule
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,)

# # loss_fct = FocalLoss(gamma=2,alpha=0.1)
# # loss_fct = nn.CrossEntropyLoss()
# # loss_fct = nn.BCELoss()
# loss_fct = nn.BCEWithLogitsLoss(pos_weight = torch.Tensor([10]).cuda())
# # loss_fct = nn.BCEWithLogitsLoss()

# # loss_fct = focal_loss
# # loss_fct = WeightedFocalLoss(alpha=5,gamma = 3)
# # loss_fct = DiceLoss()
# # loss_fct = dice.dice_loss

# eval()
# for epoch in range(1,n_epochs+1):
#     train(epoch)
#     out,y_target = eval()
#     y_target = np.concatenate(y_target).squeeze(axis=1)
#     print('precision {0}, recall {1}, f1 score {2}'.format(*precision_recall_fscore_support(out.flatten(),y_target.flatten(),average='binary')))
#     scheduler.step(f1_score(out.flatten(),y_target.flatten()))

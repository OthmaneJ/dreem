import torch 
from torch.utils.data import TensorDataset, DataLoader


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




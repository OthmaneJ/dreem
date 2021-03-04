
import torch.nn as nn
import torch


class EncoderLayer(nn.Module):

  def __init__(self,maxpool=10):
    super(EncoderLayer,self).__init__()
    self.conv1 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=5,padding=2)
    self.batch_norm1 = nn.BatchNorm1d(1)
    self.conv2 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=5,padding=2)
    self.batch_norm2 = nn.BatchNorm1d(1)
    self.maxpool = nn.MaxPool1d(maxpool)
  
  def forward(self,input):
    output = self.conv1(input)
    output = self.batch_norm1(output)
    output = self.conv2(output)
    output = self.batch_norm2(output)
    output_features = output
    output = self.maxpool(output)
    return output,output_features




class DecoderLayer(nn.Module):

  def __init__(self,kernel_size=10):
    super(DecoderLayer,self).__init__()
    self.upsample = nn.Upsample(scale_factor=kernel_size)
    self.conv1 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
    self.batch_norm1 = nn.BatchNorm1d(1)
    self.conv2 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=(kernel_size+1)//2)
    self.batch_norm2 = nn.BatchNorm1d(1)
    self.conv3 = nn.Conv1d(in_channels=2,out_channels=1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
    self.batch_norm3 = nn.BatchNorm1d(1)
    self.conv4 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=(kernel_size+1)//2)
    self.batch_norm4 = nn.BatchNorm1d(1)

  def forward(self,input,encoder_output):
    output = self.upsample(input)
    output = self.conv1(output)
    output = self.batch_norm1(output)
    output = self.conv2(output)
    output = self.batch_norm2(output)

    diff = encoder_output.shape[2] - output.shape[2]
    # encoder_output = encoder_output.narrow(2,diff,output.shape[2])
    output = nn.functional.pad(output,(diff//2,diff//2))

    output = torch.cat((output,encoder_output),dim=1)

    output = self.conv3(output)
    output = self.batch_norm3(output)
    output = self.conv4(output)
    output = self.batch_norm4(output)

    return output



class UTime(nn.Module):
  
  def __init__(self):
    super(UTime,self).__init__()
    self.encoder1 = EncoderLayer(10)
    self.encoder2 = EncoderLayer(8)
    self.encoder3 = EncoderLayer(6)
    self.encoder4 = EncoderLayer(4)

    self.decoder1 = DecoderLayer(10)
    self.decoder2 = DecoderLayer(8)
    self.decoder3 = DecoderLayer(6)
    self.decoder4 = DecoderLayer(4)

    self.conv1 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=5,padding=2)
    self.batch_norm1 = nn.BatchNorm1d(1)
    self.conv2 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=5,padding=2)
    self.batch_norm2 = nn.BatchNorm1d(1)

    self.segment_classifier = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=100,stride=100)



  def forward(self,input):
    output,output_features1 = self.encoder1(input)
    output,output_features2 = self.encoder2(output)
    output,output_features3 = self.encoder3(output)
    output,output_features4 = self.encoder4(output)

    output = self.conv1(output)
    output = self.batch_norm1(output)
    output = self.conv2(output)
    output = self.batch_norm2(output)

    output = self.decoder4(output,output_features4)
    output = self.decoder3(output,output_features3)
    output = self.decoder2(output,output_features2)
    output = self.decoder1(output,output_features1)

    output = self.segment_classifier(output)


    

    return output
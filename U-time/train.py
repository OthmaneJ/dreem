
X = h5_file['data'][:,2:9002]
y_train = pd.read_csv(PATH_TO_TRAINING_TARGET,index_col=0)
y = y_train.to_numpy()
print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val= train_test_split(X,y,test_size=0.3,random_state=42)
#dataloading 
from torch.utils.data import TensorDataset, DataLoader
def make_loader(X,y,batch_size=10):
  X = torch.Tensor(X).unsqueeze(1)
  y = torch.Tensor(y).unsqueeze(1)
  data = TensorDataset(X,y)
  dataloader = DataLoader(data,batch_size=batch_size)
  return dataloader

train_loader = make_loader(X_train,y_train,)
val_loader = make_loader(X_val,y_val)

loss_fct = nn.BCEWithLogitsLoss()

device = torch.device('cuda')


def train(epoch):
    utime.train()
    losses=[]

    with tqdm(train_loader,unit="batch") as tepoch:
        for data,target in tepoch:
            tepoch.set_description(f'epoch {epoch}')
            output = utime(data.to(device))
            loss = loss_fct(output,target.to(device))
            loss.backward()
            losses.append(loss.detach().item())
            optimizer.step()
            tepoch.set_postfix(loss = np.average(losses))


def eval():
    utime.eval()
    losses=[]

    with tqdm(val_loader) as tepoch:
        for data,target in tepoch:
            tepoch.set_description('evaluation')
            output = utime(data.to(device))
            loss = loss_fct(output,target.to(device))
            losses.append(loss.detach().item())
            tepoch.set_postfix(loss = np.average(losses))


if __name__ == 'main':
    utime = UTime()
    utime.to(device)


    optimizer=Adam(utime.parameters())
    optimizer.zero_grad()
        for epoch in range(1,10):
            train(epoch)
            eval()




import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import trainEEGDataset,infEEGDataset
from models import Net
import numpy as np
from tqdm import tqdm
from torchvision.transforms import Compose
from audiomentations import Gain,Shift,AddGaussianNoise
from augmentation import RandomMask

def train_one_fold(train_index, test_index,fold,EEG_DATA,lbls):
    
    data_transform = Compose([
            AddGaussianNoise(min_amplitude=0.00001, max_amplitude=0.05, p=.7),
            Shift(max_shift=0.2,min_shift=-0.2,p=.7,rollover=True),
            Gain(min_gain_db=-0.2,max_gain_db=0.4,p=.7),
            RandomMask(p=.7,fade=False,pr=0.02),
        ])


    transformed_dataset1 = trainEEGDataset(EEG_DATA,data_transform,train_index)
    train_loader = DataLoader(transformed_dataset1, batch_size=512,shuffle=True)
    
    inf_dataset1 = infEEGDataset(EEG_DATA,data_transform,train_index,lbls,train_index)
    inf_dataset2 = infEEGDataset(EEG_DATA,data_transform,test_index,lbls,train_index)
    exec_loader = DataLoader(inf_dataset1, batch_size=512,shuffle=False)
    test_loader = DataLoader(inf_dataset2, batch_size=128,shuffle=False)
    
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-4
    EPOCHS = 150

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = xm.xla_device()
    model=Net().float()
    model.to(device)
    #######################
    epsilon = torch.zeros_like(torch.cat([param.view(-1) for param in model.cnn2.parameters()]))
    epsilon = epsilon.detach()

    epsilon_cnn1 = torch.zeros_like(torch.cat([param.view(-1) for param in model.cnn1.parameters()]))
    epsilon_cnn2 = torch.zeros_like(torch.cat([param.view(-1) for param in model.cnn2.parameters()]))
    epsilon_cnn1 = epsilon_cnn1.detach()
    epsilon_cnn2 = epsilon_cnn2.detach()
    target_decay_rate = 0.1
    
    ##############################
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay=1e-4)
    # Training loop
    minloss=5000

    # Train your model
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):#train_loader:
            optimizer.zero_grad()
            x1, x2 = batch
            x1, x2 = x1[0],x1[1]
            x1 = x1.float().to(device)
            x2 = x2.float().to(device)
           

            loss = model(x1,x2)
            loss+=model(x2,x1)
            loss/=2
            loss.backward()

            optimizer.step()
            ####################
            
            epsilon_cnn2 = target_decay_rate * epsilon_cnn2 + (1 - target_decay_rate) * torch.cat([param.view(-1) for param in model.cnn1.parameters()]).detach()

            # Update target network parameters for cnn2
            for param, epsilon_value in zip(model.cnn2.parameters(), epsilon_cnn2.split([param.numel() for param in model.cnn2.parameters()])):
                param.data = epsilon_value.view(param.shape)
           

            ##########################################
            total_loss += loss.item()
        if minloss>(total_loss/ len(train_loader)):
            minloss=(total_loss/ len(train_loader))
            torch.save(model.state_dict(), '/kaggle/working/model-fold'+str(fold)+'.pt')
        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {average_loss:.4f}")
    res=np.array([])
    
    model.load_state_dict(torch.load('/kaggle/working/model-fold'+str(fold)+'.pt'))
    with torch.no_grad():
        for batch in exec_loader:
            x,y= batch
            x = x.float().to(device)
            if res.shape[0]==0:
                res=np.concatenate((model.cnn1(x).cpu().detach().numpy(), y[:, None]), axis=1)
            else:
                res=np.concatenate((res,np.concatenate((model.cnn1(x).cpu().detach().numpy(), y[:, None]), axis=1)),axis=0)
        np.save('train_fold_'+str(fold)+'.npy', res)
            #test_infer
        res=np.array([])   
        for batch in test_loader:
            x,y= batch
            x = x.float().to(device)
            if res.shape[0]==0:
                res=np.concatenate((model.cnn1(x).cpu().detach().numpy(), y[:, None]), axis=1)
            else:
                res=np.concatenate((res,np.concatenate((model.cnn1(x).cpu().detach().numpy(), y[:, None]), axis=1)),axis=0)
        np.save('test_fold_'+str(fold)+'.npy', res)
    return model
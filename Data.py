import torch

#from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import numpy as np



def Generate_Gaussian():
    # Training data is 100 points in [0,1] inclusive regularly spaced
    train_x = torch.linspace(0, 1, 100)
    # True function is sin(2*pi*x) with Gaussian noise
    train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
    test_x = torch.linspace(0,1,51)
    test_y = torch.sin(test_x * (2 * math.pi))
    return train_x,train_y,test_x,test_y

def Load_BOSTON():
    #X, y = load_boston(return_X_y=True)
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    y = raw_df.values[1::2, 2]
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(np.expand_dims(y, -1))

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=.25,
                                                        random_state=42)

    X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
    X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()

    ds_train = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=16, shuffle=True)

    ds_test = torch.utils.data.TensorDataset(X_test, y_test)
    dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=16, shuffle=True)

    return dataloader_train,X_test,y_test


def Load_expdata(datapth='/home/user/wyn/Fedbay/data/experimentdata.xlsx'):
    df = pd.read_excel(datapth)
    #X = df.iloc[3::9,1:9].values
    #y = df.iloc[3::9,9].values
    X = df.iloc[1:,1:9].values 
    y = df.iloc[1:,9].values
    Min,Max = np.min(y),np.max(y)
    new_Min,new_Max = 0.2,0.6
    y = new_Min + (new_Max-new_Min)*(y-Min)/(Max-Min)
    X = X*0.01
    #print(Min,Max)
    #print(y)
    print("用全部数据")
    return X,y

def Load_expdata1(datapth='/home/user/wyn/Fedbay/data/experimentdata.xlsx'):
    df = pd.read_excel(datapth)
    test_X = df.iloc[3::10,1:9].values
    test_y = df.iloc[3::10,9].values
    test_indices = df.index[3::10]
    # Use these indices to filter out the test data from the DataFrame
    train_df = df.drop(test_indices)
    # Extract train_X and train_y
    train_X = train_df.iloc[1:, 1:9].values
    train_y = train_df.iloc[1:, 9].values
    #print(train_y)
    #print(test_y)
    #train_X = df.iloc[1:405,1:9].values
    #train_y = df.iloc[1:405,9].values
    #test_X = df.iloc[405:,1:9].values
    #test_y = df.iloc[405:,9].values
    Min,Max = np.min(train_y),np.max(train_y)
    new_Min,new_Max = 0.2,0.6
    train_y = new_Min + (new_Max-new_Min)*(train_y-Min)/(Max-Min)
    test_y = new_Min + (new_Max-new_Min)*(test_y-Min)/(Max-Min)
    train_X,test_X = train_X*0.01,test_X*0.01
    #print(test_y)
    #print("用全部数据")
    return train_X,train_y,test_X,test_y

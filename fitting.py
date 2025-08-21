import numpy as np
import matplotlib.pyplot as plt 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import torch
import torch.nn as nn
import torch.optim as optim
import Data
from Modules import BayesianRegressor,evaluate_regression
import bnlearn as bn
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(model,X,y):
    y_pred, y_std = model.predict(X, return_std=True)
    print("y_pred",y_pred)
    print("y",y)
    #print("y_std",y_std)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    x = np.arange(len(y))
    plt.figure(figsize=(30, 6))
    plt.scatter(x, y, color='blue', label='True Data')
    plt.scatter(x, y_pred, color='red', label='Predict Data')
    plt.legend()
    plt.show()
    plt.savefig('2GPplot.png')

    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")


def GP():
    train_X,train_y,test_X,test_y = Data.Load_expdata1()
    #X,y = Data.Load_expdata()
    kernel = ConstantKernel(1.0) * RBF(length_scale=0.01)
    model = GaussianProcessRegressor(kernel=kernel, random_state=25)
    model.fit(train_X,train_y)
    evaluate(model,test_X,test_y)
def GP_Full():
    X,y = Data.Load_expdata()
    kernel = ConstantKernel(1.0) * RBF(length_scale=10.0)
    model = GaussianProcessRegressor(kernel=kernel,alpha = 1e-10, random_state=525)
    model.fit(X,y)
    print("Optimized kernel:", model.kernel_)
    evaluate(model,X,y)

def Bayesian():
    #X,y = Data.Load_expdata()
    train_X,train_y,test_X,test_y = Data.Load_expdata1()
    #X = X.astype(np.float32)
    #y = y.astype(np.float32)
    train_X,train_y = train_X.astype(np.float32),train_y.astype(np.float32)
    test_X,test_y = test_X.astype(np.float32),test_y.astype(np.float32)

    # 将 X 和 y 转换为 PyTorch Tensor
    #X = torch.tensor(X, dtype=torch.float32)  # 默认 float32
    #y = torch.tensor(y, dtype=torch.float32)  # 默认 float32
    train_X,train_y = torch.tensor(train_X, dtype=torch.float32),torch.tensor(train_y, dtype=torch.float32)
    test_X,test_y = torch.tensor(test_X, dtype=torch.float32),torch.tensor(test_y, dtype=torch.float32)
    model = BayesianRegressor(input_dim=8, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 训练模型
    n_epochs = 100
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        #output = model(X)
        loss = model.sample_elbo(inputs=train_X.to(device),
                           labels=train_y.to(device),
                           criterion=criterion,
                           sample_nbr=3,
                           complexity_cost_weight=1/train_X.shape[0])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')

    ic_acc, upper_acc, lower_acc = evaluate_regression(model, test_X, test_y)
    print(f'Confidence Interval Accuracy: {ic_acc:.4f}')
    print(f'Upper Bound Accuracy: {upper_acc:.4f}')
    print(f'Lower Bound Accuracy: {lower_acc:.4f}')


def Bayesian_Full():
    X,y = Data.Load_expdata()
    X,y = X.astype(np.float32),y.astype(np.float32)
    X,y = torch.tensor(X, dtype=torch.float32),torch.tensor(y, dtype=torch.float32)

    model = BayesianRegressor(input_dim=8, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    n_epochs = 300
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        #output = model(X)
        loss = model.sample_elbo(inputs=X.to(device),
                           labels=y.to(device),
                           criterion=criterion,
                           sample_nbr=5,
                           complexity_cost_weight=1/X.shape[0])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')

    ic_acc, upper_acc, lower_acc = evaluate_regression(model, X, y)
    print(f'Confidence Interval Accuracy: {ic_acc:.4f}')
    print(f'Upper Bound Accuracy: {upper_acc:.4f}')
    print(f'Lower Bound Accuracy: {lower_acc:.4f}')

if __name__ == '__main__':
    GP()
    #Bayesian_Full()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
import pandas as pd
#from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os


#from matplotlib import pyplot as plt
import gpytorch
from gpytorch import kernels, means, models, mlls, settings
from gpytorch import distributions as distr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]
#X, y = load_boston(return_X_y=True)
X = StandardScaler().fit_transform(X)
y = StandardScaler().fit_transform(np.expand_dims(y, -1))

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=.25,
                                                    random_state=42)

X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()
halfl = len(X_train)//2
X_train = X_train[90:100]
y_train = y_train[90:100]

ds_train = torch.utils.data.TensorDataset(X_train, y_train)
dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=16, shuffle=True)

ds_test = torch.utils.data.TensorDataset(X_test, y_test)
dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=16, shuffle=True)



@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #self.linear = nn.Linear(input_dim, output_dim)
        self.blinear1 = BayesianLinear(input_dim, 512)
        self.blinear2 = BayesianLinear(512, output_dim)
        
    def forward(self, x):
        x_ = self.blinear1(x)
        x_ = F.relu(x_)
        return self.blinear2(x_)

def Plot(means,upper,lower,y):
    # 创建 x 轴数据，可以简单地使用索引
    means = means.flatten()

    x = np.arange(len(means))
    plt.figure(figsize=(30, 6))
    plt.scatter(x, y, color='blue', label='True Data')

    plt.plot(x, means, color='red', label='Predicted Mean')
    plt.fill_between(x, upper.flatten(), lower.flatten(), color='pink', alpha=0.5, label='Confidence Interval')
    plt.legend()
    plt.show()

    plt.savefig('bayplot.png')

def evaluate_regression(regressor,
                        X,
                        y,
                        samples = 50,
                        std_multiplier = 1):
    preds = [regressor(X) for i in range(samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    #print("means",means)
    #print("y",y)
    stds = preds.std(axis=0)
    #print("stds",stds)
    
    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)

    Plot(means.detach().numpy(),ci_upper.detach().numpy(),ci_lower.detach().numpy(),y)
    ic_acc = (ci_lower <= y) * (ci_upper >= y)
    ic_acc = ic_acc.float().mean()

    
    mse = mean_squared_error(y,means.detach().numpy())
    mae = mean_absolute_error(y, means.detach().numpy())
    r2 = r2_score(y, means.detach().numpy())
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")
    return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()


def train_model(regressor,eps,lr = 0.01):
    optimizer = optim.Adam(regressor.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    iteration = 0 
    for epoch in range(eps):
        for i, (datapoints, labels) in enumerate(dataloader_train):
            optimizer.zero_grad()
            
            loss = regressor.sample_elbo(inputs=datapoints.to(device),
                            labels=labels.to(device),
                            criterion=criterion,
                            sample_nbr=3,
                            complexity_cost_weight=1/X_train.shape[0])
            loss.backward()
            optimizer.step()
            
            iteration += 1
            if iteration%100==0:
                ic_acc, under_ci_upper, over_ci_lower = evaluate_regression(regressor,
                                                                        X_test.to(device),
                                                                        y_test.to(device),
                                                                        samples=25,
                                                                        std_multiplier=3)
            
                print("CI acc: {:.2f}, CI upper acc: {:.2f}, CI lower acc: {:.2f}".format(ic_acc, under_ci_upper, over_ci_lower))
                print("Loss: {:.4f}".format(loss))

    return X_train.shape[0]

def test_model(regressor):
    ic_acc, under_ci_upper, over_ci_lower = evaluate_regression(regressor,
                                                                X_test.to(device),
                                                                y_test.to(device),
                                                                samples=25,
                                                                std_multiplier=3)
    #print("In Test Model!")
    print("CI acc: {:.2f}, CI upper acc: {:.2f}, CI lower acc: {:.2f}".format(ic_acc, under_ci_upper, over_ci_lower))                                                               
    #print("type ",type(X_test.shape[0]),type[ic_acc])
    return ic_acc,int(X_test.shape[0]),float(under_ci_upper),float(over_ci_lower)



class ExactGPModel(models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = means.ConstantMean()
        self.covar_module = kernels.ScaleKernel(kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return distr.MultivariateNormal(mean_x, covar_x)
    
def train_Guassian(model,likelihood,train_x,train_y,ep,lr=0.1):
    '''smoke_test = ('CI' in os.environ)
    training_iter = 2 if smoke_test else 50'''

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(ep):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        if i % 5 == 4:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, ep, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
        optimizer.step()
    
def test_Guassian(model,likelihood,testx,testy):
    f_preds = model(testx)
    y_preds = likelihood(model(testx))

    f_mean = f_preds.mean
    f_var = f_preds.variance
    f_covar = f_preds.covariance_matrix
    f_samples = f_preds.sample(sample_shape=torch.Size((1000,)))

    lower_bound = f_mean - 1.96 * torch.sqrt(f_var)
    upper_bound = f_mean + 1.96 * torch.sqrt(f_var)

    # 检查 testy 是否在置信区间内
    in_confidence_interval = (testy >= lower_bound) & (testy <= upper_bound)
    interval = in_confidence_interval.sum().item()/in_confidence_interval.size(0)
    #print("f_mean ",f_mean)
    #print("interval ",in_confidence_interval.sum().item()/in_confidence_interval.size(0))
    return float(interval),f_mean.tolist()[0]

if __name__ == '__main__':
    pass
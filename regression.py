import pandas as pd 
import numpy as np 
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white')
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.cluster import MiniBatchKMeans
input_path = '../data/' # original dataset
result_path = '../results/'

train = pd.read_csv(input_path+'taxi.csv')

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures 
N= 10000
mod_train = train.iloc[:N, :]
x = mod_train.iloc[:, :-1]
y = mod_train.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim 
class Dataset_py(Dataset):
  def __init__(self, x, y):
    self.x = torch.tensor(x, dtype=torch.float32)
    self.y = torch.tensor(y, dtype=torch.float32)

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]



y_train_array, y_test_array = y_train.values, y_test.values
y_train_array = y_train_array.reshape((len(y_train_array), 1))
y_test_array = y_test_array.reshape((len(y_test_array), 1))

batch_size = 32
train_ds = Dataset_py(x_train, y_train_array)
# test_ds = Dataset_py(x_test, y_test_array)
train_dl = DataLoader(train_ds, batch_size=batch_size)
xtrain_t = torch.tensor(x_train, dtype=torch.float32)
xtest_t = torch.tensor(x_test, dtype=torch.float32)

class MLP(nn.Module):
  def __init__(self, input_size, output_size):
    super(MLP, self).__init__()
    self.net = nn.Sequential(
    nn.Linear(input_size, 64), 
    nn.ReLU(), 
    nn.Linear(64, 32), 
    nn.ReLU(),
    nn.Linear(32, output_size),
    )

  def forward(self, x):
    pred = self.net(x)
    return pred 


nn_model = MLP(x_train.shape[1], 1)
cost = nn.MSELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
loss_list = []

epoch_num = 100
for epoch in range(epoch_num):
  for x_batch, y_batch in train_dl:
      pred = nn_model(x_batch)
      loss = torch.sqrt(cost(pred, y_batch))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  print('epoch %s, error %s'%(epoch, loss.item()))
  loss_list.append(loss.item())

nn_model.eval()
nn_y = nn_model(xtest_t).detach().numpy()
nn_yt = nn_model(xtrain_t).detach().numpy()
nn_test_error = np.sqrt(mean_squared_error(nn_y, y_test_array))
nn_train_error = np.sqrt(mean_squared_error(nn_yt, y_train_array))
print('train_error, test_error', nn_train_error, nn_test_error)


plt.figure(figsize=(6,4))
plt.plot(loss_list)
plt.xlabel('Epcoh num', fontsize=12)
plt.ylabel('Loss',fontsize=12)
plt.tight_layout()
plt.savefig(result_path+'taxi_nn_learning_curve'+'.png', dpi=100)
plt.show()



# model_list = ['nn']
# plt.figure(figsize=(6,4))
# plt.bar(np.arange(len(model_list)), train_error_list, width=0.2,  color='lightblue', align='center', label='Train')
# plt.bar(np.arange(len(model_list))-0.2, test_error_list, width=0.2, color='y', align='center', label='Test')
# plt.xticks(np.arange(len(model_list)), model_list, rotation=90)
# # plt.xlim([-1, x_train.shape[1]])
# plt.xlabel('Models', fontsize=12)
# plt.ylabel('Error', fontsize=12)
# plt.legend(loc=0, fontsize=12)
# plt.tight_layout()
# plt.savefig(result_path+'taxi_models_error'+'.png', dpi=100)
# plt.show()


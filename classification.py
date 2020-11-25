import numpy as np 
import pandas as pd 
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
import ml_metrics as metrics 
from sklearn.ensemble import RandomForestClassifier
from utils import *

input_path = '../data/' # original dataset
result_path = '../results/'

print('load data')
train = pd.read_csv(input_path+'hotel.csv', nrows=10000)

print('Normalize Data')
x = train.iloc[:, :-1].values 
y = train.iloc[:, -1].values
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
		self.y = torch.tensor(y, dtype=torch.long)

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]


y_train_array = y_train.reshape((len(y_train), 1))
y_test_array = y_test.reshape((len(y_test), 1))

batch_size = 64

train_ds = Dataset_py(x_train, y_train)

train_dl = DataLoader(train_ds, batch_size=batch_size)

xtrain_t = torch.tensor(x_train, dtype=torch.float32)

xtest_t = torch.tensor(x_test, dtype=torch.float32)

label_num = np.max(train['hotel_cluster'].values)+1


class MLP(nn.Module):
	def __init__(self, input_size, output_size):
		super(MLP, self).__init__()
		self.net = nn.Sequential(
		nn.Linear(input_size, 128), 
		nn.ReLU(), 
		nn.Linear(128, 64), 
		nn.ReLU(),
		nn.Linear(64, output_size),
		# nn.Softmax(),
		)

	def forward(self, x):
		pred = self.net(x)
		return pred 


nn_model = MLP(x_train.shape[1], label_num)
cost = nn.CrossEntropyLoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.01)
loss_list = []

epoch_num = 200
for epoch in range(epoch_num):
  for x_batch, y_batch in train_dl:
      pred = nn_model(x_batch)
      loss = cost(pred, y_batch)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  print('epoch %s, error %s'%(epoch, loss.item()))
  loss_list.append(loss.item())

plt.figure(figsize=(6,4))
plt.plot(loss_list)
plt.xlabel('Epcoh num', fontsize=12)
plt.ylabel('Loss',fontsize=12)
plt.tight_layout()
plt.savefig(result_path+'hotel_nn_learning_curve'+'.png', dpi=100)
plt.show()

nn_model.eval()
train_preds = torch.log_softmax(nn_model(xtrain_t), dim=1).detach().numpy()
test_preds = torch.log_softmax(nn_model(xtest_t), dim=1).detach().numpy()
# print('train_preds', train_preds)
train_index = top_k_labels(train_preds, 5)
test_index = top_k_labels(test_preds, 5)
train_labels = [[l.tolist()] for l in y_train]
test_labels = [[l.tolist()] for l in y_test]

nn_train_score = metrics.mapk(train_labels, train_index.tolist(), 5)
nn_test_score = metrics.mapk(test_labels, test_index.tolist(), 5)
print('nn_train, %s, nn_test %s'%(nn_train_score, nn_test_score))


# model_list = [ 'Neural Network']
# train_error_list = [nn_train_score]
# test_error_list = [nn_test_score]

# plt.figure(figsize=(6,4))
# plt.bar(np.arange(len(model_list)), train_error_list, width=0.2,  color='lightblue', align='center', label='Train')
# plt.bar(np.arange(len(model_list))-0.2, test_error_list, width=0.2, color='y', align='center', label='Test')
# plt.xticks(np.arange(len(model_list)), model_list, rotation=45)
# plt.ylim([0, 1])
# plt.xlabel('Models', fontsize=12)
# plt.ylabel('Error', fontsize=12)
# plt.legend(loc=0, fontsize=12)
# plt.tight_layout()
# plt.savefig(result_path+'hotel_models_error'+'.png', dpi=100)
# plt.show()



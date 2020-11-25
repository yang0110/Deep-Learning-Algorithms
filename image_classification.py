import torch 
import torchvision
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white')
input_path = '../data/' 
result_path = '../results/'

size_train = 64
size_test = 64

train_loader = DataLoader(torchvision.datasets.MNIST(
				input_path, 
				train=True, 	
				download=True,
				transform=torchvision.transforms.Compose([
        		torchvision.transforms.ToTensor(),
               	torchvision.transforms.Normalize((0.1307,), (0.3081,))])), 
				batch_size=size_train, shuffle=True)

print('len(train_loader)', len(train_loader))

test_loader = DataLoader(
				torchvision.datasets.MNIST(
				input_path, 
				train=False, 
				download=True,
				transform=torchvision.transforms.Compose([
           		torchvision.transforms.ToTensor(),
				torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
				batch_size=size_test, shuffle=True)
print('len(test_loader)', len(test_loader))

import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.lin1 = nn.Linear(320, 50)
		self.lin2 = nn.Linear(50, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = F.max_pool2d(x, 2)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.max_pool2d(x, 2)
		x = F.relu(x)
		x = x.view(-1, 320)
		x = self.lin1(x)
		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.lin2(x)
		return F.log_softmax(x, dim=1)


model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.01)


def train():
	model.train()

	for data, target in train_loader:
		optimizer.zero_grad()
		out = model(data)
		# print('out.shape', out.shape)
		# print('target.shape', target.shape)
		loss = F.nll_loss(out, target)
		loss.backward()
		optimizer.step()

	return loss.item()

@torch.no_grad()
def test():
	model.eval()
	for data, target in test_loader:
		pred = model(data).max(1)[1]
		acc = pred.eq(target.data.view_as(pred)).sum()
	return acc 

epoch_num = 100
loss_list = []
acc_list = []

for epoch in range(epoch_num):
	loss = train()
	acc = test()
	loss_list.append(loss)
	acc_list.append(acc)
	print('epoch %s, loss %.2f, acc %.2f'%(epoch, loss, acc))

plt.figure(figsize=(6,4))
plt.plot(loss_list)
plt.xlabel('epoch', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,4))
plt.plot(acc_list)
plt.xlabel('epoch', fontsize=12)
plt.ylabel('acc', fontsize=12)
plt.tight_layout()
plt.show()


















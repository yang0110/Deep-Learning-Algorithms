import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white')
import numpy as np 
import pandas as pd 

df = sns.load_dataset('flights')

plt.figure(figsize=(6,4))
plt.plot(df['passengers'])
plt.ylabel('Total Passengers')
plt.xlabel('Months')
plt.autoscale(axis='x',tight=True)
plt.tight_layout()
plt.show()

data = df.passengers.values.astype(float)
train = data[:-12]
test = data[-12:]
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(-1,1))
train = sc.fit_transform(train.reshape(-1,1))
test = sc.transform(test.reshape(-1, 1))

train = torch.FloatTensor(train).view(-1)
test = torch.FloatTensor(test).view(-1)

def sequence(data, bw):
	seqs = []
	L = len(data)
	for i in range(L-bw):
		seq1 = data[i:i+bw]
		seq2 = data[i+bw: i+bw+1]
		seqs.append((seq1, seq2))
	return seqs 

bw = 12
train_seqs = sequence(train, bw)


class LSTM(nn.Module):
	def __init__(self, input_size=1, hidden_size=100, output_size=1):
		super().__init__()
		self.lstm = nn.LSTM(input_size, hidden_size)
		self.linear = nn.Linear(hidden_size, output_size)
	def forward(self, input_seq):
		lstm_out = self.lstm(input_seq.view(len(input_seq), 1, -1))
		preds = self.linear(lstm_out.view(len(input_seq), -1))
		return preds[-1]

model = LSTM()
cost = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
epoch_num = 100
for epoch in range(epoch_num):
	for seq, target in train_seqs:
		optimizer.zero_grad()
		pred = model(seq)
		loss = cost(pred, target)
		loss.backward()
		optimizer.step()
	print('epoch %s, loss %.2f'%(epoch, loss.item()))

fut_pred = 12
test_seqs = train[-bw:].tolist()
model.eval()
for i in range(fut_pred):
	seq = torch.FloatTensor(test_seqs[-bw:])
	with torch.no_grad():
		test_seqs.append(model(seq).item())

pred = sc.inverse_transform(np.array(test_seqs[bw:]).reshape(-1,1))
x = np.arange(132, 144, 1)

# plt.title('Month vs Passenger')
# plt.ylabel('Total Passengers')
# plt.grid(True)
# plt.autoscale(axis='x', tight=True)
# plt.plot(flight_data['passengers'])
# plt.show()

plt.figure(figsize=(6,4))
plt.plot(df['passengers'])
plt.plot(x, pred)
plt.ylabel('Total Passengers')
plt.xlabel('Months')
plt.autoscale(axis='x',tight=True)
plt.tight_layout()
plt.show()









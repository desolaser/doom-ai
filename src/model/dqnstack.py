import torch.nn as nn
import torch.nn.functional as F
import math

class dqnstack(nn.Module):
	""" This NN can read RGB images, because of that has better
	performance """

	def __init__(self, available_actions_count, linear_input):
		super(dqnstack, self).__init__()
		self.linear_input = linear_input
		self.inter_input = math.trunc(linear_input / 2)

		self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=2)       
		self.conv1_bn = nn.BatchNorm2d(32)
		self.pool1 = nn.MaxPool2d(2)    
		self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=1)       
		self.conv2_bn = nn.BatchNorm2d(32)   
		self.pool2 = nn.MaxPool2d(2)  

		self.fc1 = nn.Linear(linear_input, self.inter_input)
		self.fc2 = nn.Linear(self.inter_input, available_actions_count)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.conv1_bn(x)
		x = self.pool1(x)
		x = F.relu(self.conv2(x))
		x = self.conv2_bn(x) 
		x = self.pool2(x)
		x = x.view(-1, self.linear_input)
		x = F.relu(self.fc1(x))
		return self.fc2(x)

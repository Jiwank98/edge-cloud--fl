# dqn.py
# DQN 정의
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import random


class DQN(nn.Module):

    def __init__(self, name, params):
        super(DQN, self).__init__()
        self.name = name
        self.params = params
        self.fc_variable_no = params.fc_variable_no
        self.x_dim = params.x_dim
        self.y_dim = params.y_dim

        # network 용 변수
        self.fc_in = nn.Linear(self.x_dim, self.fc_variable_no)
        self.fc_hidden1 = nn.Linear(self.fc_variable_no, self.fc_variable_no)
        self.fc_hidden2 = nn.Linear(self.fc_variable_no, self.fc_variable_no)
        self.fc_hidden3 = nn.Linear(self.fc_variable_no, self.fc_variable_no)
        self.fc_hidden4 = nn.Linear(self.fc_variable_no, self.fc_variable_no)
        self.fc_out = nn.Linear(self.fc_variable_no, self.y_dim)
        self.relu = nn.ReLU()
        torch.nn.init.xavier_uniform_(self.fc_in.weight)
        torch.nn.init.xavier_uniform_(self.fc_hidden1.weight)
        torch.nn.init.xavier_uniform_(self.fc_hidden2.weight)
        torch.nn.init.xavier_uniform_(self.fc_hidden3.weight)
        torch.nn.init.xavier_uniform_(self.fc_hidden4.weight)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, x):
        x = torch.reshape(x, [-1, self.x_dim])
        x = self.relu(self.fc_in(x))
        x = self.relu(self.fc_hidden1(x))
        x = self.relu(self.fc_hidden2(x))
        x = self.relu(self.fc_hidden3(x))
        x = self.relu(self.fc_hidden4(x))
        x = self.fc_out(x)
        return x
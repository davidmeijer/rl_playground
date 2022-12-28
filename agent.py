#!/usr/bin/env python3
"""
Adapted from:
https://towardsdatascience.com/deep-q-learning-for-the-cartpole-44d761085c2f
https://github.com/ritakurban/Practical-Data-Science/blob/master/DQL_CartPole.ipynb
https://towardsdatascience.com/create-your-own-reinforcement-learning-environment-beb12f4151ef
https://github.com/shivaverma/Orbit/blob/master/Paddle/DQN_agent.py 

https://github.com/neunms/Reinforcement-learning-on-graphs-A-survey

https://arxiv.org/pdf/2002.07717.pdf
https://mlg-blog.com/2021/04/30/reinforcement-learning-for-3d-molecular-design.html

Diffusion approach?
"""
import argparse 
import random
import math  
import copy
import time 

import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential
from torch.autograd import Variable

from torch_geometric.nn import NNConv, Set2Set

import numpy as np


class Agent(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.lin0 = torch.nn.Linear(input_dim, hidden_dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, hidden_dim * hidden_dim))
        self.conv = NNConv(hidden_dim, hidden_dim, nn, aggr='mean')
        self.gru = GRU(hidden_dim, hidden_dim)

        self.set2set = Set2Set(hidden_dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state):
        out = F.relu(self.lin0(state.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, state.edge_index, state.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, state.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)


class Agent:
    def __init__(self, lr: float = 0.05) -> None:
        self.criterion = torch.nn.MSELoss()
        self.model = MPNN(input_dim=1, hidden_dim=16)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def update(self, state: Data, y):
        # TODO
        y_pred = self.model(torch.Tensor(state))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        # TODO
        with torch.no_grad():
            return self.model(torch.Tensor(state))

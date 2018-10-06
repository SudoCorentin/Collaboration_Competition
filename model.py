import torch as th
import torch.nn as nn
import torch.nn.functional as F
import pdb; 
import numpy as np


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return th.Tensor(size).uniform_(-v, v)

class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action,seed, fc_units_1=64, fc_units_2=64,fc_units_3=64,init_w=3e-3):
        super(Critic, self).__init__()
        self.seed = th.manual_seed(seed)
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation * n_agent
        act_dim = self.dim_action * n_agent

        self.bn0 = nn.BatchNorm1d(obs_dim)
        self.FC1 = nn.Linear(obs_dim, fc_units_1)
        self.bn1 = nn.BatchNorm1d(fc_units_1)
        self.FC2 = nn.Linear(fc_units_1+act_dim, fc_units_2)
        self.bn2 = nn.BatchNorm1d(fc_units_2)
        self.FC3 = nn.Linear(fc_units_2, 1)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.FC1.weight.data = fanin_init(self.FC1.weight.data.size())
        self.FC2.weight.data = fanin_init(self.FC2.weight.data.size())
        self.FC3.weight.data.uniform_(-init_w, init_w)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        result = self.bn0(obs)
        result = F.relu(self.FC1(result))
        #result = self.bn1(result)
        combined = th.cat([result, acts], 1)
        result = F.relu(self.FC2(combined))
        #result = self.bn2(result)
        return self.FC3(result)

class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action,seed, fc_units_1=64, fc_units_2=32, init_w=3e-3):
        super(Actor, self).__init__()
        self.seed = th.manual_seed(seed)

        self.FC1 = nn.Linear(dim_observation, fc_units_1)
        self.bn1 = nn.BatchNorm1d(fc_units_1)
        self.FC2 = nn.Linear(fc_units_1, fc_units_2)
        self.bn2 = nn.BatchNorm1d(fc_units_2)
        self.FC3 = nn.Linear(fc_units_2, dim_action)
        self.bn3 = nn.BatchNorm1d(dim_action)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.FC1.weight.data = fanin_init(self.FC1.weight.data.size())
        self.FC2.weight.data = fanin_init(self.FC2.weight.data.size())
        self.FC3.weight.data.uniform_(-init_w, init_w)

    def forward(self, obs):
        #pdb.set_trace()
        result = F.relu(self.FC1(obs))
        result = self.bn1(result)
        result = F.relu(self.FC2(result))
        result = self.bn2(result)
        result = F.tanh(self.bn3(self.FC3(result)))
        return result

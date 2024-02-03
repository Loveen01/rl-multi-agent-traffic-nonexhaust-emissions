import torch 
import torch.nn as nn 
import numpy as np 

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, no_hidden_layers):
        super().__init__()
        self.flatten = nn.Flatten()
        self.soft_max = nn.Softmax()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.no_hidden_layers = no_hidden_layers
        
        self.linear_network = nn.Sequential(
            nn.Linear(self.state_dim, self.no_hidden_layers), # 2 - 3 layers. 84 neurones because each policy absorbs the entire observation space 
            nn.ReLU(),
            nn.Linear(self.no_hidden_layers,self.no_hidden_layers), # 100 neurones to start with - neurones should be approx within range of no. features
            nn.ReLU(),
            nn.Linear(self.no_hidden_layers,self.action_dim), # 4 represents the phases each intersection entails. 
        )

    def forward(self, x):
        if type(x)==dict:
            arr = np.array(x.values())
            x = torch.from_numpy(arr)
            print(x)
        if type(x)==np.ndarray:
            x = torch.from_numpy(arr)
        output = self.linear_network(x.float())
        logits = self.soft_max(output)
        return logits

class Critic(nn.Module):
    def __init__(self, state_dim, no_hidden_layers):
        super().__init__()
        self.flatten = nn.Flatten()
        self.soft_max = nn.Softmax()
        self.state_dim = state_dim
        self.no_hidden_layers = no_hidden_layers
        
        self.linear_network = nn.Sequential(
            nn.Linear(self.state_dim,self.no_hidden_layers), # 2 - 3 layers. 84 neurones because each policy absorbs the entire observation space 
            nn.ReLU(),
            nn.Linear(self.no_hidden_layers,self.no_hidden_layers), # 100 neurones to start with - neurones should be approx within range of no. features
            nn.ReLU(),
            nn.Linear(self.no_hidden_layers,1), # outputs the value of being in a particular state 
        )

    def forward(self, x):
        if type(x)==dict:
            arr = np.array(x.values())
            x = torch.from_numpy(arr)
        if type(x)==np.ndarray:
            x = torch.from_numpy(x)
        output = self.linear_network(x.float())
        return output 
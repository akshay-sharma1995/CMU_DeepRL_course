import numpy as np
import gym
import sys
import copy
from collections import deque
import os
import random
import pdb
import torch 
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):

    # This class essentially defines the network architecture. 
    # The network should take in state of the world as an input, 
    # and output Q values of the actions available to the agent as the output. 
    def __init__(self, environment_name,obs_space,action_space,lr):
            # Define your network architecture here. It is also a good idea to define any training operations 
            # and optimizers here, initialize your variables, or alternately compile your model here.  
            #pdb.set_trace()
        super().__init__()

        self.environment_name = environment_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_dim = obs_space.shape[0]

        if (environment_name=="CartPole-v0"):
            self.linear1 = nn.Linear(self.input_dim,32)
            self.output_layer = nn.Linear(32,action_space.n)

        else:

            self.linear1 = nn.Linear(self.input_dim,64)
            self.linear2 = nn.Linear(64,64)
            self.linear3 = nn.Linear(64,64)
            self.output_layer = nn.Linear(64,action_space.n)

        self.initialize_parameters()

        self.optimizer = torch.optim.Adam(self.parameters(),lr)
            
    def forward(self,X):
        X = X.float().to(device=DEVICE)
        if (self.environment_name=="CartPole-v0"):
                x_em = F.relu(self.linear1(X))

        else:
                x_em = F.relu(self.linear1(X))
                x_em = F.relu(self.linear2(x_em))
                x_em = torch.tanh(self.linear3(x_em))

        
        out = self.output_layer(x_em)

        return out


    def initialize_parameters(self):
        for param in self.parameters():
            if(len(param.shape)>1):
                torch.nn.init.xavier_normal_(param)
            else:
                torch.nn.init.constant_(param, 0.0)

    def save_checkpoint(self, checkpoint_save_path, num_episodes_trained, replay_mem):
        # Helper function to save your model / weights. 
        torch.save({'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'num_episodes_trained': num_episodes_trained,
                        'replay_mem': replay_mem},
                        checkpoint_save_path),
        print("checkpoint_saved")


    def load_checkpoint(self,checkpoint_path):
        # Helper funciton to load model weights. 
        # e.g.: self.model.load_state_dict(torch.load(model_file))
        #self.model.load_weights(model_file)
        # weight_file: full path of the pth file
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        num_episodes_trained = checkpoint['num_episodes_trained']
        replay_mem = checkpoint['replay_mem']
        
        return num_episodes_trained, replay_mem

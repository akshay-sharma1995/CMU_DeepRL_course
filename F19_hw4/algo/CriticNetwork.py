import numpy
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import pdb

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400



def create_critic_network(state_size, action_size, learning_rate):
	"""Creates a critic network.

	Args:
		state_size: (int) size of the input.
		action_size: (int) size of the action.
		learning_rate: (float) learning rate for the critic.
	Returns:
		model: an instance of tf.keras.Model.
		state_input: a tf.placeholder for the batched state.
		action_input: a tf.placeholder for the batched action.
	"""
	raise NotImplementedError
	model = tf.keras.Model(inputs=[state_input, action_input], outputs=value)
	model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
	return model, state_input, action_input


class CriticNetwork(nn.Module):
	def __init__(self, state_size, action_size,learning_rate,device):
		"""Initialize the CriticNetwork
		This class internally stores both the critic and the target critic
		nets. It also handles computation of the gradients and target updates.

		Args:
			sess: A Tensorflow session to use.
			state_size: (int) size of the input.
			action_size: (int) size of the action.
			batch_size: (int) the number of elements in each batch.
			tau: (float) the target net update rate.
			learning_rate: (float) learning rate for the critic.
		"""
		super(CriticNetwork,self).__init__()
		self.state_size = state_size
		self.action_size = action_size
		self.device = device
		# self.tau = tau

		self.state_em = nn.Sequential(nn.Linear(state_size,(state_size+action_size)//2),
									nn.ReLU(),
									)

		self.action_em = nn.Sequential(nn.Linear(action_size,(state_size+action_size)//2),
									nn.ReLU(),
									)

		self.critic = nn.Sequential(nn.Linear(2*(state_size+action_size)//2,HIDDEN1_UNITS),
									nn.ReLU(),
									nn.Linear(HIDDEN1_UNITS,HIDDEN2_UNITS),
									nn.ReLU(),
									nn.Linear(HIDDEN2_UNITS,1)
									)

		self.initialize_params()
		self.optimizer = torch.optim.Adam(self.parameters(),learning_rate)
		self.mse_loss = nn.MSELoss()
	
	def initialize_params(self):
		for param in self.parameters():
			if(len(param.shape)>1):
				nn.init.xavier_uniform_(param)
			else:
				nn.init.constant_(param,0.0)

	def forward(self,state,action):
		state = state.float().to(device=self.device)
		action = action.float().to(device=self.device)

		s_em = self.state_em(state)
		a_em = self.action_em(action)

		val = self.critic(torch.cat((s_em,a_em),dim=1))

		return val     

		
	def gradients(self, states, actions):
		"""Computes dQ(s, a) / da.
		Note that tf.gradients returns a list storing a single gradient tensor,
		so we return that gradient, rather than the singleton list.

		Args:
			states: a batched numpy array storing the state.
			actions: a batched numpy array storing the actions.
		Returns:
			grads: a batched numpy array storing the gradients.
		"""
		pass

	def update_target(self):
		"""Updates the target net using an update rate of tau."""
		pass

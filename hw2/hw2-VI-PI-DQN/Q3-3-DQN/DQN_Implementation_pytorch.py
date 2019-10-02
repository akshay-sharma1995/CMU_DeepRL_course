#!/usr/bin/env python
#import tensorflow as tf
#import keras
#from keras.models import Sequential, Model
#from keras.layers import Dense, Input
import numpy as np
import gym
import sys
import copy
import argparse
from collections import deque
import os
import random
import pdb
import torch 
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
class QNetwork(nn.Module):

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

	def __init__(self, environment_name,obs_space,action_space):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here.  
		#pdb.set_trace()
		super().__init__()
		self.obs_space = obs_space
		self.action_space = action_space
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print(self.device)
		self.input_dim = self.obs_space.shape[0]

		self.linear1 = nn.Linear(self.input_dim,32)
		self.linear2 = nn.Linear(32,16)
		self.linear3 = nn.Linear(16,self.action_space.n)
		
	def forward(self,X):
		X = torch.tensor(X).to(device=self.device).float()
		x_em = F.relu(self.linear1(X))
		x_em = F.relu(self.linear2(x_em))
		out = self.linear3(x_em)

		return out


		
	def save_model_weights(self, suffix):
		# Helper function to save your model / weights. 
		file_path = os.path.join("./model_weights/",suffix)		
		self.model.save_weights(file_path)
		pass

	def load_model(self, model_file):
		# Helper function to load an existing model.
		# e.g.: torch.save(self.model.state_dict(), model_file)
		
		#self.model = tf.keras.load_model(os.path.join("./model_weights",model_file))
		pass	

	def load_model_weights(self,weight_file):
		# Helper funciton to load model weights. 
		# e.g.: self.model.load_state_dict(torch.load(model_file))
		#self.model.load_weights(model_file)
		# weight_file: full path of the pth file
		self.model.load_state_dict(torch.load(weight_file))
		
		pass



class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=10000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
		
		# Hint: you might find this useful:
		# 		collections.deque(maxlen=memory_size)
		
		
		self.replay_deque = deque(maxlen=memory_size)
		self.memory_size = memory_size
		self.burn_in = burn_in
		pass

	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
		# You will feed this to your model to train.
		sampled_transitions_ids = random.sample(range(0,len(self.replay_deque)),batch_size)
		sampled_transitions = [self.replay_deque[transition_id] for transition_id in sampled_transitions_ids]
		return sampled_transitions

	def append(self, transition):
		# Appends transition to the memory. 
		if(len(self.replay_deque)==self.memory_size):
			self.replay_deque.pop_left()
		self.replay_deque.append(transition)
		pass


class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
	def __init__(self, environment_name, render=False, learning_rate=1e-5):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc. 
		self.environment_name = environment_name
		self.env = gym.make(self.environment_name)
		self.test_env = gym.make(self.environment_name)
		self.obs_space = self.env.observation_space
		self.action_space = self.env.action_space
		self.Q_net = QNetwork(self.environment_name,self.obs_space, self.action_space)
		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if(torch.cuda.is_available()):
			self.Q_net.cuda()
		
		self.optimizer = torch.optim.Adam(self.Q_net.parameters(),learning_rate)

		self.replay_buffer = Replay_Memory()
		self.burn_in_memory(self.replay_buffer.burn_in)
		self.gamma = 0.99
		self.batch_size = 32
		pass 

	def epsilon_greedy_policy(self, q_values,epsilon):
		# Creating epsilon greedy probabilities to sample from. 
		
		go_greedy = np.random.choice(2,size=1,p=[epsilon,1-epsilon])[0]
		if(go_greedy):
			action = np.argmax(q_values)
		else:
			action = np.random.randint(0,q_values.shape[1])
		
		return action

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time. 
		action = np.argmax(q_values)
		return action
		pass 

	def train(self,num_episodes=1000,test_after=100,eval_episodes=20,summary_writer=None):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# When use replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.
		
		state_t = self.env.reset()
		loss_per_step = []
		test_reward = []
		self.Q_net.train()
		done = False
		step_num = 0
		for ep in range(num_episodes):
			#pdb.set_trace()
			## select action using an epsilon greedy policy
			#epsilon = max((0.5 - 0.000045*ep),0.05)	
			while not(done):
				epsilon = max((0.5 - 0.000045*step_num),0.05)
				if(summary_writer):
					summary_writer.add_scalar("epsilon",epsilon,step_num)
				with torch.no_grad():
					q_values = self.Q_net(np.expand_dims(state_t,axis=0))
				
				action = self.epsilon_greedy_policy(q_values.cpu().numpy(),epsilon)
				
				## take a step in the env using the action
				state_t_1, reward, done, info = self.env.step(action)
				
				## store the transition in the replay buffer
				self.replay_buffer.append((state_t,action,reward,state_t_1,done))
				
				## sample a minibatch of random transitions from the replay buffer
				sampled_transitions = self.replay_buffer.sample_batch(batch_size=self.batch_size)
					
				q_values_target = [None]*self.batch_size

				q_pred_action_mask = np.zeros((self.batch_size,self.action_space.n))

				X_train = [None]*self.batch_size
	
				for transition_id, transition in enumerate(sampled_transitions):
					s = transition[0]
					a = transition[1]
					r = transition[2]
					s1 = transition[3]
					d = transition[4]
					
					q_pred_action_mask[transition_id,a] = 1
					if(d):
						q_values_target[transition_id] = r
					else:
						with torch.no_grad():
							q_values_target[transition_id] = r + self.gamma * torch.max(self.Q_net(np.expand_dims(s1,axis=0))).item()

					X_train[transition_id] = s.copy()

				X_train = np.array(X_train)
				
				Y_train = torch.tensor(q_values_target).float().to(device=self.device)
				
				Y_pred_all_actions = self.Q_net(X_train)
				
				Y_pred_all_actions = torch.mul(Y_pred_all_actions,torch.tensor(q_pred_action_mask).float().to(device=self.device))
				Y_pred = torch.sum(Y_pred_all_actions,dim=1)
				batch_loss = F.mse_loss(Y_pred, Y_train)
				

				self.optimizer.zero_grad()
				batch_loss.backward()
				self.optimizer.step()
				
				loss_per_step.append(batch_loss.item())
				step_num += 1
				if(summary_writer):
					summary_writer.add_scalar("train_loss_per_step",loss_per_step[-1],len(loss_per_step))
				
				if done:
					state_t = self.env.reset()
					
				else:
					state_t = state_t_1.copy()
			done = False
			if(ep%test_after==0):
				test_rew = self.test(test_num_episodes=eval_episodes)
				test_reward.append(test_rew)
				if(summary_writer):
					summary_writer.add_scalar("test_cum_reward",test_reward[-1],len(test_reward))
				print("Episode: {}".format(ep))
				print("Test-----> Cum_reward: {}".format(test_rew))
			
		return loss_per_step, test_reward

	def test(self, model_file=None, test_num_episodes=100):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
		
		if(model_file):
			self.Q_net.load_model_weights(model_file)
		done = False
		cum_reward = 0
	#	self.Q_net.eval()
		for episode in range(test_num_episodes):
			state_t = self.test_env.reset()
			done = False
			while not done:
				with torch.no_grad():
					q_values = self.Q_net(np.expand_dims(state_t,axis=0))
				action = self.greedy_policy(q_values.cpu().numpy())
				state_t_1, reward, done, info = self.test_env.step(action)
				cum_reward += reward
				state_t = state_t_1.copy()

		cum_reward /= test_num_episodes
	#	self.Q_net.train()
		return cum_reward
		


	def burn_in_memory(self,burn_in):
		# Initialize your replay memory with a burn_in number of episodes / transitions. 
		
		state_t = self.env.reset()
		done = False

		transition_counter = 0
		
		print("burn_in_start")
		while transition_counter<burn_in:
			
			with torch.no_grad():
				q_values = self.Q_net(np.expand_dims(state_t,axis=0))
			action = self.greedy_policy(q_values.cpu().numpy())
	#		
			state_t_1,reward,done,info = self.env.step(action)

			self.replay_buffer.append((state_t,action,reward,state_t_1,done))
			transition_counter += 1
			if(done):
				state_t = self.env.reset()
				done = False
			else:
				state_t = state_t_1.copy()


		print("burn_in_over")
		pass


# Note: if you have problems creating video captures on servers without GUI,
#       you could save and relaod model to create videos on your laptop. 
def test_video(agent, env, epi):
	# Usage: 
	# 	you can pass the arguments within agent.train() as:
	# 		if episode % int(self.num_episodes/3) == 0:
	#       	test_video(self, self.environment_name, episode)
	save_path = "./videos-%s-%s" % (env, epi)
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	# To create video
	env = gym.wrappers.Monitor(agent.env, save_path, force=True)
	reward_total = []
	state = env.reset()
	done = False
	while not done:
		env.render()
		action = agent.epsilon_greedy_policy(state, 0.05)
		next_state, reward, done, info = env.step(action)
		state = next_state
		reward_total.append(reward)
	print("reward_total: {}".format(np.sum(reward_total)))
	agent.env.close()


def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str)
	parser.add_argument("--lr",dest="lr",type=float,default=1e-5)
	parser.add_argument("--num-episodes",dest="num_episodes",type=int,default=1000)
	parser.add_argument("--test-after",dest="test_after",type=int,default=100)
	parser.add_argument("--eval-episodes",dest="eval_episodes",type=int,default=20)
	
	
	
	return parser.parse_args()


def main(args):

	args = parse_arguments()
	environment_name = args.env
	
	num_episodes = args.num_episodes
	test_after = args.test_after
	eval_episodes = args.eval_episodes
	lr = args.lr

	# You want to create an instance of the DQN_Agent class here, and then train / test it.
	
	log_path = os.path.join(os.getcwd(),"logs{}_{}_{}".format(args.env,args.lr,args.num_episodes))
	if not os.path.isdir(log_path):
		os.mkdir(log_path)
	summary_writer = SummaryWriter(log_path)
	agent = DQN_Agent(environment_name,lr)
	
	if(torch.cuda.is_available()):
		print("on_cuda")
		agent.Q_net.cuda()
	if (args.train):
		
		train_loss, test_reward = agent.train(num_episodes,test_after,eval_episodes,summary_writer)
		

		fig1 = plt.figure()
		plt.plot(train_loss)
		plt.xlabel("num_steps")
		plt.ylabel("train_loss")
		plt.savefig("train_loss_{}_lr_{}_eps_{}.png".format(args.env,args.lr,args.num_episodes))
		
		fig2 = plt.figure()
		plt.plot(range(0,num_episodes,test_after),test_reward)
		plt.xlabel("num_episodes")
		plt.ylabel("cummulative_test_reward")
		plt.savefig("test_reward_{}_lr_{}_eps_{}.png".format(args.env,args.lr,args.num_episodes))

if __name__ == '__main__':
	main(sys.argv)


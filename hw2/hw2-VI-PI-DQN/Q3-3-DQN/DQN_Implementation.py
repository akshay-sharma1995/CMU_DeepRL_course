#!/usr/bin/env python
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import gym
import sys
import copy
import argparse
from collections import deque
import os
import random
import pdb

class QNetwork(tf.keras.Model):

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

	def __init__(self, environment_namen,obs_space,action_space):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here.  
		#pdb.set_trace()

		super(QNetwork,self).__init__()
		self.obs_space = obs_space
		self.action_space = action_space
		self.model = Sequential()
	#	pdb.set_trace()
		self.input_dim = self.obs_space.shape[0]
		self.model.add(Dense(32,input_dim=self.input_dim, activation='relu'))
		self.model.add(Dense(16,activation='relu'))		
		self.model.add(Dense(self.action_space.n))
		
		self.model.compile(optimizer='adam',
				loss=keras.losses.mean_squared_error,
				metrics=['accuracy'])

		pass

	def save_model_weights(self, suffix):
		# Helper function to save your model / weights. 
		file_path = os.path.join("./model_weights/",suffix)		
		self.model.save_weights(file_path)
		pass

	def load_model(self, model_file):
		# Helper function to load an existing model.
		# e.g.: torch.save(self.model.state_dict(), model_file)
		
		self.model = tf.keras.load_model(os.path.join("./model_weights",model_file))
		

	def load_model_weights(self,weight_file):
		# Helper funciton to load model weights. 
		# e.g.: self.model.load_state_dict(torch.load(model_file))
		self.model.load_weights(model_file)
		
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
		
		# self.replay_buffer = []*memory_size
		
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
	
	def __init__(self, environment_name, render=False):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc. 
		self.environment_name = environment_name
		self.env = gym.make(self.environment_name)
		self.obs_space = self.env.observation_space
		self.action_space = self.env.action_space
		#input_dim = self.env.observation_space.shape[0] + self.env.action_space.n
		self.Q_net = QNetwork(self.environment_name,self.obs_space, self.action_space)
		keras.initializers.Initializer()
		self.num_episodes = 10
		self.num_iterations = 10
		self.replay_buffer = Replay_Memory()
		self.burn_in_memory(self.replay_buffer.burn_in)
		self.gamma = 0.99
		self.batch_size = 32
		pass 

	def epsilon_greedy_policy(self, q_values):
		# Creating epsilon greedy probabilities to sample from. 
		epsilon = 0.01
		
		go_greedy = np.random.choice(2,size=1,p=[epsilon,1-epsilon])[0]
		if(go_greedy):
			action = np.argmax(q_values)
		else:
			action = np.random.choice(q_values.shape[1],size=1)[0]
		
		#action = np.argmax(q_values,axis=1) if go_greedy else np.random.choice(q_values.shape[1],size=1)
		return action

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time. 
		action = np.argmax(q_values,axis=1)
		return action
		pass 

	def train(self):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# When use replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.
	
		pdb.set_trace()	
		state_t = self.env.reset()
		loss = []
		acc = []
		for iter in range(self.num_iterations):
			pdb.set_trace()
			## select action using an epsilon greedy policy
			q_values = self.Q_net.model.predict(np.expand_dims(state_t,axis=0))
			action = self.epsilon_greedy_policy(q_values)
			
			## take a step in the env using the action
			state_t_1, reward, done, info = self.env.step(action)
			
			## store the transition in the replay buffer
			self.replay_buffer.append((state_t,action,reward,state_t_1,done))
			
			## sample a minibatch of random transitions from the replay buffer
			sampled_transitions = self.replay_buffer.sample_batch(batch_size=self.batch_size)
			
				
			q_values_target = []*self.batch_size

			X_train = []*self.batch_size

			for transition_id, transition in enumerate(sampled_transitions):
				r = transition[2]
				s1 = transition[3]
				d = transition[4]
				pdb.set_trace()
				if(transition[-1]):
					q_values_target[transition_id] = r
				else:
					q_values_target[transition_id] = r + self.gamma * np.amax(self.Q_net.model.predict(np.expand_dims(s1,axis=0)))[

				X_train[transition_id] = s1.copy()

			X_train = np.array(X_train)
			Y_train = np.array(q_values_target)
			history = self.Q_net.model.fit(X_train,Y_train,epoch=5,verbose=1)
			loss += history.history['loss'][-1]
			acc += history.history['accuracy'][-1]

			if done:
				state_t = self.env.reset()
			else:
				state_t = state_t_1.copy()

		loss /= self.num_iterations
		acc /= self.num_iterations

		return loss, acc

	def test(self, model_file=None, test_num_episodes=100):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
		
		if(model_file):
			self.Q_net.load_model_weights(model_file)

		done = False
		cum_reward = 0
		for episode in range(test_num_episodes):
			state_t = self.env.reset()
			done = False
			while not done:
				q_values = self.Q_net.model.predict(np.expand_dims(state_t,axis=0))
				action = self.epsilon_greedy_policy(q_values)
				state_t_1, reward, done, info = self.env.step(action)
				cum_reward += reward
				state_t = state_t_1.copy()

		cum_reward /= test_num_episodes

		return cum_reward
		


	def burn_in_memory(self,burn_in):
		# Initialize your replay memory with a burn_in number of episodes / transitions. 
		
	#	pdb.set_trace()
		state_t = self.env.reset()
		done = False

		transition_counter = 0
		
	#	X = np.random.normal(size=(1000,4))
	#	Y = np.zeros((1000,2))
	#	his = self.Q_net.model.fit(X,Y,epochs=10,batch_size=256,verbose=1)
	#	loss = his.history['loss'][-1]
	#	acc = his.history['accuracy'][-1]

		#print("loss, acc: ",loss ,acc)
		#pdb.set_trace()
		print("burn_in_start")
		while transition_counter<burn_in:
			#pdb.set_trace()
			q_values = self.Q_net.model.predict(np.expand_dims(state_t,axis=0))
			action = self.epsilon_greedy_policy(q_values)
	#		print("action: {}".format(action))
			#pdb.set_trace()
			state_t_1,reward,done,info = self.env.step(action)
			#if(done):
			#	pdb.set_trace()
			self.replay_buffer.append((state_t,action,reward,state_t_1,done))
			transition_counter += 1
		#	print(transition_counter)
		#	print(done)
			if(done):
				state_t = self.env.reset()
			else:
				state_t = state_t_1.copy()
		#	pdb.set_trace()
			#state_t = state_t_1.copy()
			if(transition_counter==burn_in):
				
				break
	#	pdb.set_trace()
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
	return parser.parse_args()


def main(args):

	args = parse_arguments()
	environment_name = args.env

#	#pdb.set_trace()
	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)
	num_episodes = 100
	# Setting this as the default tensorflow session. 
	keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DQN_Agent class here, and then train / test it.
	
	agent = DQN_Agent(environment_name)
	if (args.train):
		train_loss = []
		train_acc = []

		for episode in range(num_episodes):
			print("Starting episode: {}".format(episode))

			loss,acc = agent.train()
			train_loss.extend(loss)
			train_acc.extend(acc)


if __name__ == '__main__':
	main(sys.argv)


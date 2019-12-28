#!/usr/bin/env python
import numpy as np
import gym
import sys
import copy
import os
import random
import pdb
import torch 
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils import Replay_Memory



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
	def __init__(self, environment_name, Q_model, 
                    render=False, learning_rate=1e-5, gamma=0.99, 
                    replay_size=100000, burn_in=20000, 
                    save_weights_path=None, replay_mem=None):

            # Create an instance of the network itself, as well as the memory. 
            # Here is also a good place to set environmental parameters,
            # as well as training parameters - number of episodes / iterations, etc. 
    
            self.environment_name = environment_name
    
            # env for generating and storing transistions
            self.env = gym.make(self.environment_name)
            
            # test env
            self.test_env = gym.make(self.environment_name)
            
            self.obs_space = self.env.observation_space
            
            self.action_space = self.env.action_space
            self.nb_actions = self.action_space.n
            
            # creating the Q network		
            self.Q_net = Q_model

            self.replay_buffer = Replay_Memory(memory_size=replay_size,burn_in=burn_in)
            if(replay_mem):
                    self.replay_buffer.replay_deque = replay_mem
            else:
                    self.burn_in_memory(burn_in)
            self.gamma = gamma
            self.batch_size = 256
		

	def epsilon_greedy_policy(self, q_values, epsilon):
            # Creating epsilon greedy probabilities to sample from. 
            # go_greedy = np.random.choice(2,size=1,p=[epsilon,1-epsilon])[0]
            
            if(random.random() > epsilon):
                action = np.argmax(q_values)
            else:
                action = np.random.choice(q_values.shape[1],size=1)[0]

            return action

	def greedy_policy(self, q_values):
            # Creating greedy policy for test time. 
            
            action = np.argmax(q_values)
            return action


	def action_to_one_hot(env, action):
            action_vec = np.zeros(env.action_space.n)
            action_vec[action] = 1
            return action_vec 

	def custom_mse_loss(self, Y_pred, Y_target, actions):
            loss = 0
            for i in range(len(actions)):
                    loss += torch.pow(Y_pred[i,actions[i]] - Y_target[actions[i]], 2)
            
            loss /= len(actions)

            # loss = torch.tensor([Y_pred[i,actions[i]] - Y_target[i,actions[i]] for i in range(0,len(actions))])
            # loss = torch.mean(torch.pow(loss,2))
            return loss

	def create_action_mask(self,actions):
            action_mask = np.zeros((actions.shape[0],self.nb_actions))
            for id, mask in enumerate(action_mask):
                    mask[actions[id]] = 1
            return action_mask

	
	def train(self,epsilon):
            # In this function, we will train our network. 
            # If training without experience replay_memory, then you will interact with the environment 
            # in this function, while also updating your network parameters. 

            # When use replay memory, you should interact with environment here, and store these 
            # transitions to memory, while also updating your model.

            curr_state = self.env.reset() ## getting the first state 
            done = False

            while not(done):
                
                with torch.no_grad():
                    q_values = self.Q_net(torch.unsqueeze(torch.from_numpy(curr_state),dim=0))
                
                action = self.epsilon_greedy_policy(q_values.cpu().numpy(), epsilon)
                next_state, reward, done, info = self.env.step(action) 
                
                self.replay_buffer.append((curr_state,action,reward,next_state,done))
                curr_state = next_state.copy()

                ## sample a minibatch of random transitions from the replay buffer
                sampled_transitions = self.replay_buffer.sample_batch(batch_size=self.batch_size)
                pdb.set_trace()
                X_train = np.array([transition[0] for transition in sampled_transitions])
                transition_actions = np.array([transition[1] for transition in sampled_transitions])
                action_mask = torch.tensor(self.create_action_mask(transition_actions),dtype=torch.bool).to(device=DEVICE)
                exp_rewards = torch.tensor([transition[2] for transition in sampled_transitions]).float().to(device=DEVICE)
                sampled_nxt_states = np.array([transition[3] for transition in sampled_transitions])
                dones = np.array([int(transition[4]) for transition in sampled_transitions])

                with torch.no_grad():
                        q_max_nxt_state,_ = torch.max(self.Q_net(torch.from_numpy(sampled_nxt_states)),axis=1) 
                q_values_target = exp_rewards + self.gamma * q_max_nxt_state * torch.tensor(1-dones).float().to(device=DEVICE)
        
                Y_pred_all_actions = self.Q_net(torch.from_numpy(X_train))

                Y_pred = torch.masked_select(Y_pred_all_actions,action_mask)

                batch_loss = F.mse_loss(Y_pred,q_values_target)
                self.Q_net.optimizer.zero_grad()
                batch_loss.backward()
                self.Q_net.optimizer.step()
                    
                #print("train_loss: {}".format(batch_loss.item()))
                    
            return batch_loss.item()

	def test(self, test_num_episodes=100):
            # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
            # Here you need to interact with the environment, irrespective of whether you are using a memory. 
            
            # load the model weights if provided with a saved model	
    
            self.Q_net.eval()
            done = False
            episodic_test_rewards = []
            for episode in range(test_num_episodes):
                episode_reward = 0
                state_t = self.env.reset()
                done = False
                while not done:
                    with torch.no_grad():
                        q_values = self.Q_net(torch.unsqueeze(torch.from_numpy(state_t),dim=0))
                    action = self.greedy_policy(q_values.cpu().numpy())
                    state_t_1, reward, done, info = self.env.step(action)
                    episode_reward += reward
                    state_t = state_t_1.copy()
                episodic_test_rewards.append(episode_reward)
            self.Q_net.train()
    
            return np.mean(episodic_test_rewards), np.std(episodic_test_rewards)
		


	def burn_in_memory(self,burn_in):
            # Initialize your replay memory with a burn_in number of episodes / transitions. 

            print("burn_in_start")

            nb_actions = self.action_space.n

            for i in range(burn_in):
                curr_state = self.env.reset()	
                done = False
                while not done:
                    action = np.random.randint(0,nb_actions)
                    next_state,reward,done,info = self.env.step(action)
                    self.replay_buffer.append((curr_state,action,reward,next_state,done))
                    curr_state = next_state.copy()

            print("burn_in_over")
            pass


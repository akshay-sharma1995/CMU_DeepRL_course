import os
import numpy as np
import gym
import copy
import torch
import torch.nn as nn
import pdb




class MPC:
	def __init__(self, env, plan_horizon, model, popsize, num_elites, max_iters,
				 num_particles=6,
				 use_gt_dynamics=True,
				 use_mpc=True,
				 use_random_optimizer=False):
		"""

		:param env:
		:param plan_horizon:
		:param model: The learned dynamics model to use, which can be None if use_gt_dynamics is True
		:param popsize: Population size
		:param num_elites: CEM parameter
		:param max_iters: CEM parameter
		:param num_particles: Number of trajectories for TS1
		:param use_gt_dynamics: Whether to use the ground truth dynamics from the environment
		:param use_mpc: Whether to use only the first action of a planned trajectory
		:param use_random_optimizer: Whether to use CEM or take random actions
		"""
		self.env = env
		self.use_gt_dynamics, self.use_mpc, self.use_random_optimizer = use_gt_dynamics, use_mpc, use_random_optimizer
		self.num_particles = num_particles
		self.plan_horizon = plan_horizon
		self.num_nets = None if model is None else model.num_nets

		self.state_dim, self.action_dim = 8, env.action_space.shape[0]
		self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low

		# Set up optimizer
		self.model = model

		if use_gt_dynamics:
			self.predict_next_state = self.predict_next_state_gt
			assert num_particles == 1
		else:
			self.predict_next_state = self.predict_next_state_model




		# TODO: write your code here
		# Initialize your planner with the relevant arguments.
		# Write different optimizers for cem and random actions respectively
		self.popsize = popsize
		self.num_elites = num_elites
		self.action_shape = self.env.action_space.shape[0]
		self.max_iters = max_iters
		# raise NotImplementedError

	def obs_cost_fn(self, state):
		""" Cost function of the current state """
		# Weights for different terms
		W_PUSHER = 1
		W_GOAL = 2
		W_DIFF = 5
		self.goal = state[-2:]

		pusher_x, pusher_y = state[0], state[1]
		box_x, box_y = state[2], state[3]
		goal_x, goal_y = self.goal[0], self.goal[1]

		pusher_box = np.array([box_x - pusher_x, box_y - pusher_y])
		box_goal = np.array([goal_x - box_x, goal_y - box_y])
		d_box = np.sqrt(np.dot(pusher_box, pusher_box))
		d_goal = np.sqrt(np.dot(box_goal, box_goal))
		diff_coord = np.abs(box_x / box_y - goal_x / goal_y)
		# the -0.4 is to adjust for the radius of the box and pusher
		return W_PUSHER * np.max(d_box - 0.4, 0) + W_GOAL * d_goal + W_DIFF * diff_coord

	def predict_next_state_model(self, states, actions):
		""" Given a list of state action pairs, use the learned model to predict the next state"""
		# TODO: write your code here

		next_states = self.model(torch.tensor(states), torch.tensor(actions))
		return next_states

	def predict_next_state_gt(self, state, action):
		""" Given a list of state action pairs, use the ground truth dynamics to predict the next state"""
		# TODO: write your code here

		# next_states = []

		# for state, action in zip(states,actions):
		next_states = self.env.get_nxt_state(state, action)

		return np.array(next_states)


	def train(self, obs_trajs, acs_trajs, rews_trajs, epochs=5):
		"""
		Take the input obs, acs, rews and append to existing transitions the train model.
		Arguments:
		  obs_trajs: states
		  acs_trajs: actions
		  rews_trajs: rewards (NOTE: this may not be used)
		  epochs: number of epochs to train for
		"""
		# TODO: write your code here

		
		# raise NotImplementedError
		pass

	def reset(self):
		# TODO: write your code here
		# raise NotImplementedError
		pass

	def CEM(self, state):

		initial_state = state.copy()


		mean = np.zeros((self.plan_horizon*self.action_shape))
		cov  = 0.5*np.ones((self.plan_horizon*self.action_shape))

		for i in range(self.max_iters):

			trajectory_costs = []

			actions_sequence = np.random.multivariate_normal(mean, np.diag(cov), self.popsize) 

			actions_sequence = np.reshape( actions_sequence, (self.popsize, self.plan_horizon, self.action_shape) )

			# M* plan_H* action_shape
			for m in range(actions_sequence.shape[0]):
				trajectory_cost = []
				state = initial_state
				for t in range(actions_sequence.shape[1]):
					action = actions_sequence[m,t,:] 
					trajectory_cost.append(self.obs_cost_fn(state))
					state = self.predict_next_state(state,action)
				trajectory_costs.append(np.mean(trajectory_cost,axis=0))

			max_indices = np.argsort(np.array(trajectory_costs))

			top_indices = max_indices[0:self.num_elites]

			mean = np.mean(actions_sequence[top_indices,:,:],axis=0).reshape(-1)
			cov  = np.cov(actions_sequence[top_indices,:,:].reshape(self.num_elites,-1).T) 
			cov = np.diagonal(cov)

		return mean.reshape( (self.plan_horizon, self.action_shape) ) 

	def random_action(self,state):

		initial_state = state.copy()

		actions_sequence = np.random.multivariate_normal(mean, np.diag(cov), self.max_iters) 

		actions_sequence = np.reshape( actions_sequence, (self.max_iters, self.plan_horizon, self.action_shape) )

		# M* plan_H* action_shape
		for m in range(actions_sequence.shape[0]):
			trajectory_cost = []
			state = initial_state
			for t in range(actions_sequence.shape[1]):
				action = actions_sequence[m,t,:] 
				trajectory_cost.append(self.obs_cost_fn(state))
				state = self.predict_next_state(state,action)
			trajectory_costs.append(np.mean(trajectory_cost,axis=0))

		max_indices = np.argsort(np.array(trajectory_cost))
		top_index   = max_indices[0]


		return actions_sequence[top_index,:,:]



	def act(self, state, t):
		"""
		Use model predictive control to find the action given current state.

		Arguments:
		  state: current state
		  t: current timestep
		"""
		# TODO: write your code here

		if self.use_random_optimizer:

			actions = self.random_action(state)

		else:

			actions =  self.CEM(state)

		# print("yum=",np.shape(actions))

		# print(np.shape(actions[0]))

		return actions[0]












	# TODO: write any helper functions that you need


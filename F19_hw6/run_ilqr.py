import numpy as np
from ilqr import *
import deeprl_hw6
from copy import deepcopy as dcp
from utils import plot_prop
import pdb
from run_lqr import run_lqr


def run_ilqr(env,plot_path):
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape
    total_reward = 0
    num_steps = 0
    u_arr = []
    q_arr = []
    qdot_arr = []
    u_arr = []
    
    x0, u_seq = generate_init_u(env)
    env.reset()
    env.state = x0*1.0
    done = False
    
    prev_cost = 0
    new_cost = 0
    tN = 115
    # x0 = env.reset()
    # u_seq = 0.1*np.random.uniform([0,-0],[0,0],size=(tN,2))
    
    u_seq_old = u_seq * 1.0
    u_seq = calc_ilqr_input(env,dcp(env),x0,u_seq[0:tN],tN=tN)
     
    x_seq,u_arr, q_arr, qdot_arr, total_reward = use_input(env,x0,u_seq)
    print("total_reward: {}".format(total_reward))


def use_input(env,x,u_seq):
    env.state = x*1.0
    x_seq = []

    q_arr = []
    qdot_arr = []
    u_arr = []
    
    done = False
    max_steps = u_seq.shape[0]
    num_steps = 0
    total_reward = 0
    while (not done) and (num_steps<max_steps):
        x_seq.append(x*1.0)
        u_arr.append(u_seq[num_steps]*1.0)
        q_arr.append(x[0:2]*1.0)
        qdot_arr.append(x[2:]*1.0)
        x_next,reward,done,info = env.step(u_seq[num_steps])
        x = x_next*1.0 
        num_steps += 1
        total_reward += reward
    print(done)
    return np.array(x_seq), np.array(u_arr), np.array(q_arr), np.array(qdot_arr), total_reward
            
def generate_init_u(env):
    x0, u_seq = run_lqr(env)
    return x0, u_seq

    






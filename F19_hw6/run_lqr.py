import numpy as np
from controllers import *
import deeprl_hw6
from copy import deepcopy as dcp
from utils import plot_prop
def run_lqr(env,plot_path=None):
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape

    total_reward = 0
    num_steps = 0
   
    u_arr = []
    q_arr = []
    qdot_arr = []

    u_arr = []
    u = np.zeros(action_dim)
    
    # reward_arr = []
    # total_reward_arr = []
    
    state = env.reset()
    state0 = state*1.0
   
    done = False
    
    q_arr.append(state[0:2]*1.0)
    qdot_arr.append(state[2:]*1.0)
    
    u_arr.append(u*1.0)

    while (not done) and (num_steps<20000):
        # env.render()
        num_steps += 1
        # u = calc_lqr_input(dcp(env),state,u)
        u = calc_lqr_input(env,dcp(env))
        next_state,reward,done,info = env.step(u)
        state = next_state * 1.0
        
        u_arr.append(u*1.0)
        q_arr.append(next_state[0:2]*1.0)
        qdot_arr.append(next_state[2:]*1.0)
        
        total_reward += reward
        
        # reward_arr.append(reward*1.0)
        # total_reward_arr.append(total_reward*1.0)
        print(num_steps,total_reward)
    print("num_steps: {}  return: {}".format(num_steps,total_reward))
    
    q_arr = np.array(q_arr)
    qdot_arr = np.array(qdot_arr)
    u_arr = np.array(u_arr)
    if (plot_path):
        plot_prop([q_arr[:,0],q_arr[:,1]],["q1","q2"],"joint_angles",plot_path)
        plot_prop([qdot_arr[:,0],qdot_arr[:,1]],["qdot1","qdot2"],"joint_velocities",plot_path)
        plot_prop([u_arr[:,0],u_arr[:,1]],["u1","u2"],"control_inputs",plot_path)
        # plot_prop([reward_arr],["reward"],"reward",plot_path)
        # plot_prop([total_reward_arr],["total_reward"],"total_reward",plot_path)

    return state0, u_arr

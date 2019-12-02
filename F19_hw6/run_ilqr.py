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
    horizon = 100 
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
    tN = 180
    # x0 = env.reset()
    # u_seq = 0.1*np.random.uniform([-100,-100],[100,100],size=(tN,2))
    
    u_seq_old = u_seq * 1.0
    u_seq = calc_ilqr_input(env,dcp(env),x0,u_seq[0:tN],tN=tN)
     
    pdb.set_trace()
    x_seq,u_arr, q_arr, qdot_arr, total_reward = use_input(env,x0,u_seq)
    print("total_reward: {}".format(total_reward))


    # if (plot_path):
        # plot_prop([q_arr[:,0],q_arr[:,1]],["q1","q2"],"joint_angles",plot_path)
        # plot_prop([qdot_arr[:,0],qdot_arr[:,1]],["qdot1","qdot2"],"joint_velocities",plot_path)
        # plot_prop([u_arr[:,0],u_arr[:,1]],["u1","u2"],"control_inputs",plot_path)
    # while abs(new_cost-prev_cost)>epsilon:
        # prev_cost = new_cost * 1.0
        # u_seq = calc_ilqr_input(env,dcp(env))
        # x_seq = forward_recursion(env,x0,u_seq)

        # new_cost = calc_cost(x_seq,u_seq)


    # print("num_steps: {}  return: {}".format(num_steps,total_reward))
        
    # plot_prop(np.array(q_arr),"q",plot_path)
    # plot_prop(np.array(qdot_arr),"qdot",plot_path)
    # plot_prop(np.array(u_arr),"u",plot_path)

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

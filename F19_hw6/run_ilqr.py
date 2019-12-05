import numpy as np
from ilqr import *
import deeprl_hw6
from copy import deepcopy as dcp
from utils import plot_prop
import pdb
from run_lqr import run_lqr
from utils import plot_prop, plot_x_v_y

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
    tN = 100
    np.random.seed(1)
    u_seq =  np.random.uniform([-150,-150],[150,150],size=(tN,2))
    # u_seq = u_seq[0:tN] + np.random.uniform([-1,-1],[1,1],size=(tN,2))
    u_seq,total_cost = calc_ilqr_input(env,dcp(env),x0,u_seq[0:tN],tN=tN)
    # print(total_cost) 
    x_seq,u_arr, q_arr, qdot_arr, total_reward = use_input(env,x0,u_seq)
    
    print("num_steps: {}, total_reward: {}".format(u_seq.shape[0],total_reward))
    
    if (plot_path):
        plot_prop([q_arr[:,0],q_arr[:,1]],["q1","q2"],"joint_angles",plot_path)
        plot_prop([qdot_arr[:,0],qdot_arr[:,1]],["qdot1","qdot2"],"joint_velocities",plot_path)
        plot_prop([u_arr[:,0],u_arr[:,1]],["u1","u2"],"control_inputs",plot_path)
        plot_prop([total_cost],["total_cost"],"total_cost",plot_path,xlabel="iterations")
        plot_x_v_y(q_arr[:,0],q_arr[:,1],"q1_vs_q2", plot_path)
    

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

    return np.array(x_seq), np.array(u_arr), np.array(q_arr), np.array(qdot_arr), total_reward
            
def generate_init_u(env):
    x0, u_seq = run_lqr(env,print_res=False)
    return x0, u_seq

    






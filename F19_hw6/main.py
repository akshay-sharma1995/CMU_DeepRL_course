import numpy as np
from controllers import *
import deeprl_hw6
import gym
import pdb
from copy import deepcopy as dcp
import os
import argparse
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
def run_lqr(plot_path):
    env = gym.make("TwoLinkArm-v0")
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape
    
    alb = 0
    aub = 1

    state = env.reset()
    done = False
    total_reward = 0
    num_steps = 0
    u_arr = []
    q_arr = []
    qdot_arr = []
    u_arr = []
    u = np.random.uniform(np.array([12,67]),np.array([34,23]),action_dim)     
    u = np.zeros(action_dim)

    while (not done) and (num_steps<20000):
        num_steps += 1
        u = calc_lqr_input(dcp(env),state,u)
        next_state,reward,done,info = env.step(u)
        u_arr.append(u*1.0)
        q_arr.append(next_state[0:2]*1.0)
        qdot_arr.append(next_state[2:]*1.0)
        total_reward += reward
        print(num_steps,total_reward)
        state = next_state * 1.0
    print("num_steps: {}  return: {}".format(num_steps,total_reward))
        
    plot_prop(np.array(q_arr),"q",plot_path)
    plot_prop(np.array(qdot_arr),"qdot",plot_path)
    plot_prop(np.array(u_arr),"u",plot_path)


def plot_prop(prop,prop_name,plot_path):
    figure = plt.figure(figsize=(16,9))
    plt.plot(prop[:,0],prop[:,1])
    # plt.scatter(prop[:,0],prop[:,1],marker='_')
    plt.xlabel("{}[0]".format(prop_name))
    plt.ylabel("{}[1]".format(prop_name))
    plt.title(prop_name)
    plt.savefig(os.path.join(plot_path,"{}.png".format(prop_name)))
    figure.clf()
    plt.close()


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', dest='algo', type=str,
                        default="lqr", help="lqr or ilqr ")

    return parser.parse_args()


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def main():
    args = parse_arguments()
    algo = args.algo
    if(algo=="lqr"):
        plot_path = os.path.join(os.getcwd(),"lqr_plots")
        make_dir(plot_path)
        run_lqr(plot_path)

    elif(algo=="ilqr"):
        plot_path = os.path.join(os.getcwd(),"ilqr_plots")
        make_dir(plot_path)
        run_ilqr(plot_path)

    else:
        raise ValueError("algo could only be \"lqr\" or \"ilqr\" ")

if __name__ == "__main__":
    main()

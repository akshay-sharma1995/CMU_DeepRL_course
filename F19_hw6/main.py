import gym
import os
from utils import parse_arguments, make_dir
from run_lqr import run_lqr
from run_ilqr import run_ilqr
import pdb

def main():
    args = parse_arguments()
    algo = args.algo
    env = gym.make("TwoLinkArm-v0")
    
    if(algo=="lqr"):
        plot_path = os.path.join(os.getcwd(),"lqr_plots")
        make_dir(plot_path)
        x0,u_seq = run_lqr(env,plot_path)

    elif(algo=="ilqr"):
        plot_path = os.path.join(os.getcwd(),"ilqr_plots")
        make_dir(plot_path)
        run_ilqr(env,plot_path)

    else:
        raise ValueError("algo could only be \"lqr\" or \"ilqr\" ")

if __name__ == "__main__":
    main()

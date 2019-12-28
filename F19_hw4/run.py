import gym
import envs
import envs2
from algo.ddpg import DDPG
from algo.TD3 import DDPG_TD3
from utils import make_dirs, plot_rewards, parse_arguments
import os
from matplotlib import pyplot as plt
import torch
import time
import pdb
from tensorboardX import SummaryWriter

def main():
        args = parse_arguments()
        if(args.try_gpu==1):
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            DEVICE = torch.device("cpu")
        num_episodes = args.num_episodes
        sigma = args.sigma
        lr_a = args.lr_a
        lr_c = args.lr_c
        env_name = args.env
        add_comment = args.add_comment
        TD3 = args.TD3
        delay = args.delay
        hindsight = bool(args.HER)
        results_dir = os.path.join(os.getcwd(),"results")
        
        if(TD3==1 and hindsight==1):
            print("HER + TD3 not implemented. Please turn one of them off")
            raise NotImplementedError

        if(hindsight):
            algo_path = os.path.join(results_dir,"her")
        elif(TD3):
            algo_path = os.path.join(results_dir,"td3")
        else:
            algo_path = os.path.join(results_dir,"ddpg")


        env_path = os.path.join(algo_path,env_name)
        if(TD3):
            curr_run_path = os.path.join(env_path,"num_ep_{}_lra_{}_lrc_{}_sigma_{}_delay_{}{}".format(num_episodes,lr_a,lr_c,sigma,delay,add_comment))
        else:
            curr_run_path = os.path.join(env_path,"num_ep_{}_lra_{}_lrc_{}_sigma_{}{}".format(num_episodes,lr_a,lr_c,sigma,add_comment))
        data_path = os.path.join(curr_run_path,"data")
        plots_path = os.path.join(curr_run_path,"plots")
        log_file_path = os.path.join(curr_run_path,"logs")

        make_dirs([results_dir,algo_path,env_path,curr_run_path,data_path,plots_path,log_file_path])


        sum_writer = SummaryWriter(logdir=log_file_path)
        outfile = os.path.join(curr_run_path,'ddpg_log.txt') 
        env = gym.make(env_name)
        if(TD3):
            algo = DDPG_TD3(env,lr_a,lr_c,sigma,data_path,plots_path,outfile,device=DEVICE,delay=delay,logger=sum_writer)
        else:
            algo = DDPG(env,lr_a,lr_c,sigma,data_path,plots_path,outfile,device=DEVICE,logger=sum_writer)
        
        start_time = time.time()
        algo.train(num_episodes,hindsight=hindsight)
        end_time = time.time()
        time_elasped = end_time - start_time
        print("Time_taken_for_{}_episodes_on_{}: {:.0f} min {:.2f} sec".format(num_episodes, 
                                                                        DEVICE.type, 
                                                                        time_elasped//60, 
                                                                        time_elasped%60))

if __name__ == '__main__':
	main()

import gym
import envs
from algo.ddpg import DDPG
import os
from matplotlib import pyplot as plt
import argparse
import torch
import time
import pdb
def parse_arguments():
# Command-line flags are defined here.
        parser = argparse.ArgumentParser()

        parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                                                default=50000, help="Number of episodes to train on.")
        parser.add_argument("--sigma",dest="sigma",type=float,
                                                default=0.5,help="std for action selection")
#        parser.add_argument("--epsilon",dest="epsilon",type=float,
 #                                               default=0.5,help="epsilon for the random normal process")
        parser.add_argument('--lr-a', dest='lr_a', type=float,
                                                default=5e-4, help="The actor's learning rate.")
        parser.add_argument('--lr-c', dest='lr_c', type=float,
                                                default=5e-4, help="The critic's learning rate.")
        parser.add_argument('--env', dest='env', type=str,
                                                default='Pushing2D-v0', help="environment_name")
        parser.add_argument("--add-comment", dest="add_comment", type=str,
                                                default = "", help="any special comment for the model name")
        parser.add_argument("--try-gpu", dest="try_gpu", type=int,
                                                default = 1, help="try to look for gpu")
        parser.add_argument("--HER", dest="HER", type=int,
                                                default = 1, help="try to look for gpu")

        return parser.parse_args()

def main():
        args = parse_arguments()

        if(args.try_gpu==1):
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            DEVICE = torch.device("cpu")
        num_episodes = args.num_episodes
        sigma = args.sigma
  #      epsilon = args.epsilon
        lr_a = args.lr_a
        lr_c = args.lr_c
        env_name = args.env
        add_comment = args.add_comment
        # pdb.set_trace()
        hindsight = bool(args.HER)
        results_dir = os.path.join(os.getcwd(),"results")
        if(hindsight):
            algo_path = os.path.join(results_dir,"her")
        else:
            algo_path = os.path.join(results_dir,"ddpg")


        env_path = os.path.join(algo_path,env_name)
        curr_run_path = os.path.join(env_path,"num_ep_{}_lra_{}_lrc_{}_sigma_{}{}".format(num_episodes,lr_a,lr_c,sigma,add_comment))
        data_path = os.path.join(curr_run_path,"data")
        plots_path = os.path.join(curr_run_path,"plots")
       
        make_dirs([results_dir,algo_path,env_path,curr_run_path,data_path,plots_path])
        outfile = os.path.join(curr_run_path,'ddpg_log.txt') 
        env = gym.make(env_name)
        algo = DDPG(env,lr_a,lr_c,sigma,data_path,plots_path,outfile,device=DEVICE)
        
        start_time = time.time()
        algo.train(num_episodes,hindsight=hindsight)
        end_time = time.time()
        time_elasped = end_time - start_time
        print("Time_taken_for_{}_episodes_on_{}: {:.0f} min {:.2f} sec".format(num_episodes, 
                                                                        DEVICE.type, 
                                                                        time_elasped//60, 
                                                                        time_elasped%60))

def make_dirs(path_list):
    for path in path_list:
        if not os.path.isdir(path):
            os.mkdir(path)

def plot_rewards(mean_arr,std_arr):
	mean_arr = np.mean(mean_arr)
	std_arr = np.mean(std_arr)
	fig = plt.figure(figsize=(16, 9))
	x =	np.arange(0,mean.shape[0])
	plt.plot(x,mean, label="mean_test_reward",color='orangered')
	plt.fill_between(x,mean-std, mean+std,facecolor='peachpuff',alpha=0.5)
	plt.xlabel("num episodes X {}".format(100))
	plt.ylabel("test_reward")
	plt.legend()
	plt.savefig("test_rewards.png")
	# plt.savefig(os.path.join(plots_path,"test_reward_num_ep_{}_lr_{}_gamma_{}.png".format(num_episodes,lr,gamma)))
if __name__ == '__main__':
	main()

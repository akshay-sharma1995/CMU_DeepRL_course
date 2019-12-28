import os
import argparse
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('Agg')


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
        parser.add_argument("--TD3", dest="TD3", type=int,
                                                default = 0, help="want to run TD3")
        parser.add_argument("--delay", dest="delay", type=int,
                                                default = 2, help="update delay for TD3")

        return parser.parse_args()

def make_dirs(path_list):
    for path in path_list:
        if not os.path.isdir(path):
            os.mkdir(path)

def plot_rewards(mean_arr,std_arr):
	mean = np.mean(mean_arr)
	std = np.mean(std_arr)
	fig = plt.figure(figsize=(16, 9))
	x = np.arange(0,mean.shape[0])
	plt.plot(x,mean, label="mean_test_reward",color='orangered')
	plt.fill_between(x,mean-std, mean+std,facecolor='peachpuff',alpha=0.5)
	plt.xlabel("num episodes X {}".format(100))
	plt.ylabel("test_reward")
	plt.legend()
	plt.savefig("test_rewards.png")
	# plt.savefig(os.path.join(plots_path,"test_reward_num_ep_{}_lr_{}_gamma_{}.png".format(num_episodes,lr,gamma)))

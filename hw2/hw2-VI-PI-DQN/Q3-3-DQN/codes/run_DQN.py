import numpy as np
import os
import gym
import matplotlib.pyplot as plt
import sys
from utils import *
from Q_net import QNetwork
import torch
from DQN_Implementation import *
import matplotlib as mpl
mpl.use('Agg')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):

    args = parse_arguments()

    environment_name = args.env
    num_episodes = args.num_episodes
    test_after = args.test_after
    lr = args.lr
    gamma = args.gamma
    replay_size = args.replay_size
    burn_in = args.burn_in
    save_data = 1
    save_after = 100
    checkpoint_file = args.checkpoint_file
    new_num_episodes = args.new_num_episodes
    num_trained_episodes = 0
    add_comment = args.add_comment

    ques_path = os.path.join(os.getcwd(),"dqn") 
    env_path = os.path.join(ques_path,"env_{}".format(environment_name))
    curr_run_path = os.path.join(env_path,"num_ep_{}_lr_{}_gamma_{}{}".format(num_episodes,lr,gamma,add_comment))
    plots_path = os.path.join(curr_run_path,"plots")
    data_path = os.path.join(curr_run_path,"data")

    if not os.path.isdir(ques_path):
            os.mkdir(ques_path)	

    if not os.path.isdir(env_path):
            os.mkdir(env_path)

    if not os.path.isdir(curr_run_path):
            os.mkdir(curr_run_path)

    if not os.path.isdir(plots_path):
            os.mkdir(plots_path)

    if not os.path.isdir(data_path):
            os.mkdir(data_path)

    env = gym.make(environment_name)
    
    ## defining the Q_network
    Q_net = QNetwork(environment_name,env.observation_space,env.action_space,lr)
    
    replay_mem = None

    if(DEVICE.type=="cuda"):
            Q_net.cuda()
            print("model shifted to gpu")
    
    if(checkpoint_file):
            checkpoint_file = os.path.join(curr_run_path,checkpoint_file)
            num_trained_episodes = Q_net.load_checkpoint(checkpoint_file)
            num_episodes,replay_mem = new_num_episodes

    agent = DQN_Agent(environment_name=environment_name, 
                            Q_model = Q_net,
                            render=False, 
                            learning_rate=lr, 
                            gamma=gamma, 
                            replay_size=replay_size,
                            burn_in=burn_in,
                            replay_mem = replay_mem)
    

    train_loss = []
    mean_test_reward = [] 
    std_test_reward = []
    epsilon = 1.00
    

    if (args.train):
        for ep in range(num_trained_episodes,num_episodes):
            epsilon = max((1.00 - 0.95*ep/2000),0.05)
            train_loss.append(agent.train(epsilon))
            if(ep%test_after==0):
                print("episode : {}".format(ep))
                mean_reward, std_reward = agent.test(test_num_episodes=20)
                mean_test_reward.append(mean_reward)
                std_test_reward.append(std_reward)
                print("epsilon: {:.4f}".format(epsilon))
                print("mean: {}, std: {:.2f}".format(mean_reward,std_reward))
                    
            if(save_data and ep%save_after==0):	
            ## saving data
                np.save(os.path.join(data_path,"mean_test_reward.npy"),mean_test_reward)
                np.save(os.path.join(data_path,"std_test_reward.npy"),std_test_reward)
            if(False):
                agent.Q_net.save_checkpoint(os.path.join(curr_run_path,"checkpoint.pth"),ep,agent.replay_buffer.replay_deque)
        
        fig = plt.figure(figsize=(16,9))
        plt.plot(train_loss,label="train_loss")
        plt.xlabel("num_stps")
        plt.ylabel("train_loss")
        plt.legend()
        plt.savefig(os.path.join(plots_path,"train_loss.png"))

        plt.plot(train_loss,label="train_loss")
        mean = np.array(mean_test_reward)
        std = np.array(std_test_reward)
        x = range(0,mean.shape[0])

        plt.clf()
        plt.close()
        fig = plt.figure(figsize=(16,9))
        plt.plot(mean,label="mean_cumulative_test_reward",color='crimson')
        plt.fill_between(x,mean-std,mean+std,facecolor="lightpink",alpha=0.7)
        plt.xlabel("num_trainingepisodes / 100")
        plt.ylabel("test_reward")
        plt.legend()
        plt.savefig(os.path.join(plots_path,"test_reward.png"))

if __name__ == '__main__':
	main(sys.argv)


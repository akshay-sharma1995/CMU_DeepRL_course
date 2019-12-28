import argparse
import os
import sys
import pdb
from collections import deque
import numpy as np

class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the 
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
        
        # Hint: you might find this useful:
        # 		collections.deque(maxlen=memory_size)
        
        
        self.replay_deque = deque(maxlen=memory_size)
        self.memory_size = memory_size
        self.burn_in = burn_in ## just for reference purposes
    

    def sample_batch(self, batch_size=32):
            
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
        # You will feed this to your model to train.
        
        sampled_ids = np.random.choice(np.arange(len(self.replay_deque)),size=batch_size)

        sampled_transitions = [self.replay_deque[id] for id in sampled_ids]

        return np.array(sampled_transitions)

    def append(self, transition):
        # Appends transition to the memory. 
    
        self.replay_deque.append(transition)
		 
def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str)
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
    parser.add_argument("--lr",dest="lr",type=float,default=1e-5)
    parser.add_argument("--num-episodes",dest="num_episodes",type=int,default=1000)
    parser.add_argument("--test-after",dest="test_after",type=int,default=100)
    parser.add_argument("--gamma",dest="gamma",type=float,default=0.99)
    parser.add_argument("--replay-size",dest="replay_size",type=int,default=100000)
    parser.add_argument("--burn-in",dest="burn_in",type=int,default=20000)	
    parser.add_argument("--checkpoint-file",dest="checkpoint_file",type=str,default=None)
    parser.add_argument("--new-num-episodes",dest="new_num_episodes",type=int,default=None)
    parser.add_argument("--add-comment",dest="add_comment",type=str,default="")
    
    return parser.parse_args()


def test_video(agent, env, epi):
    # Usage: 
    # 	you can pass the arguments within agent.train() as:
    # 		if episode % int(self.num_episodes/3) == 0:
    #       	test_video(self, self.environment_name, episode)
    save_path = "./videos-%s-%s" % (env, epi)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
# To create video
    env = gym.wrappers.Monitor(agent.env, save_path, force=True)
    reward_total = []
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.epsilon_greedy_policy(state, 0.05)
        next_state, reward, done, info = env.step(action)
        state = next_state
        reward_total.append(reward)
    print("reward_total: {}".format(np.sum(reward_total)))
    agent.env.close()

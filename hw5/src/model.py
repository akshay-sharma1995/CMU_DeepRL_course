# import tensorflow as tf
# from keras.layers import Dense, Flatten, Input, Concatenate, Lambda, Activation
# from keras.models import Model
# from keras.regularizers import l2
# import keras.backend as K
import torch 
import torch.nn as nn
import torch.functional as f
import numpy as np
from util import ZFilter

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400
HIDDEN3_UNITS = 400


class PENN:
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(self, num_nets, state_dim, action_dim, learning_rate):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """

        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Log variance bounds
        self.max_logvar = torch.tensor(-3 * np.ones([1, self.state_dim]), requires_grad = True)
        self.min_logvar = torch.tensor(-7 * np.ones([1, self.state_dim]), requires_grad = True)

        self.network = Network(state_dim, action_dim, learning_rate)

        # TODO write your code here
        # Create and initialize your mode

    def get_output(self, output):
        """
        Argument:
          output: tf variable representing the output of the keras models, i.e., model.output
        Return:
          mean and log variance tf tensors
        Note that you will still have to call sess.run on these tensors in order to get the
        actual output.
        """
        mean = output[:, 0:self.state_dim]
        raw_v = output[:, self.state_dim:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    # def create_network(self):
    #     I = Input(shape=[self.state_dim + self.action_dim], name='input')
    #     h1 = Dense(HIDDEN1_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(I)
    #     h2 = Dense(HIDDEN2_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(h1)
    #     h3 = Dense(HIDDEN3_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(h2)
    #     O = Dense(2 * self.state_dim, activation='linear', kernel_regularizer=l2(0.0001))(h3)
    #     model = Model(input=I, output=O)
    #     return model

    def lossFun(self, mean, cov, targets):

        retrun ((mean - targets).T@np.linalg.inverse(cov)@(mean - targets))+ np.log(np.linalg.det(cov))


    def train(self, inputs, targets, batch_size=128, epochs=5):
        """
        Arguments:
          inputs: state and action inputs.  Assumes that inputs are standardized.
          targets: resulting states
        """
        # TODO: write your code here\

        data_indices = np.arange(len(inputs))
        for n in range(self.num_nets):
            sampled_indices = np.random.choice(data_indices,len(data_indices),replace=True)

            for e in range(epochs):
                np.random.shuffle(sampled_indices)
                num_batches = sampled_indices // batch_size

                for b in range(num_batches):
                    batch_indices = sampled_indices[b*batch_size:(b+1)*batch_size]
                    input_batch = inputs[batch_indices]
                    target_batch = targets[batch_indices]

                    out = self.network(input_batch)

                    mean , logvar = get_output(out)

                    cov = np.diag(np.exp(logvar))

                    loss = lossFun(mean,cov,target_batch)

                    loss = torch.sum(loss)

                    self.network.optimizer.zero_grad()
                    loss.backward()
                    self.network.optimizer.step()




    # TODO: Write any helper functions that you need

class Network(nn.Module):


    def __init__(self,state_dim, action_dim, learning_rate):

        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.l1   = nn.Linear(self.state_dim + self.action_dim, HIDDEN1_UNITS)
        self.l2   = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.l3   = nn.Linear(HIDDEN2_UNITS, HIDDEN3_UNITS)
        self.out  = nn.Linear(HIDDEN3_UNITS, 2 * self.state_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate, weight_decay = 0.0001)

    def forward(self,input):

        input = torch.tensor(input)

        l1 = F.relu( self.l1(input) )
        l2 = F.relu( self.l2(l1) )
        l3 = F.relu( self.l3(l2) )
        out = self.out(l3)
        return out



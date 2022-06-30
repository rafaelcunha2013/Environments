from torch import nn
import torch
from collections import deque
import numpy as np
import random

# from network import Network
# from agents.network import Network
# from PettingZoo.pettingzoo.agents.network import Network
from agents.network import Network


# GAMMA=0.99
# BATCH_SIZE=32
# BUFFER_SIZE=50000
# MIN_REPLAY_SIZE=1000
# EPSILON_START=1.0
# EPSILON_END=0.02
# EPSILON_DECAY=10000
# TARGET_UPDATE_FREQ=1000*20
# LEARNING_RATE = 5e-4

class Agent:
    def __init__(self, input_dim, output_dim, ag_id, env, parameters, load=False):
        self.ag_id = ag_id
        self.N = env.max_num_agents
        self.step = 0

        self.replay_buffer = deque(maxlen=parameters['BUFFER_SIZE'])
        self.online_net = Network(input_dim, output_dim, hidden_dim=parameters['HIDDEN_DIM'])
        self.target_net = Network(input_dim, output_dim, hidden_dim=parameters['HIDDEN_DIM'])

        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=parameters['LEARNING_RATE'])

        self.min_replay_size = parameters['MIN_REPLAY_SIZE']
        self.epsilon_start = parameters['EPSILON_START']
        self.epsilon_end = parameters['EPSILON_END']
        self.epsilon_decay = parameters['EPSILON_DECAY']
        self.batch_size = parameters['BATCH_SIZE']
        self.gamma = parameters['GAMMA']
        self.target_update_freq = parameters['TARGET_UPDATE_FREQ']
        self.learning_rate = parameters['LEARNING_RATE']

        if load:
            self.load()


    def train(self, step):
        # Start Gradient Step
        transitions = random.sample(self.replay_buffer, self.batch_size)

        obses = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_obses = np.asarray([t[4] for t in transitions])

        obses_t = torch.as_tensor(obses, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

        # Compute Targets
        with torch.no_grad():
            if self.online_net.double:
                online_q_values = self.online_net(new_obses_t)
                argmax_online_q_values = online_q_values.argmax(dim=1, keepdim=True)

                target_q_values = self.target_net(new_obses_t)
                target_selected_q_values = torch.gather(input=target_q_values, dim=1, index=argmax_online_q_values)
                targets = rews_t + self.gamma * (1-dones_t) * target_selected_q_values
            else:
                target_q_values = self.target_net(new_obses_t)
                max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

                targets = rews_t + self.gamma * (1-dones_t) * max_target_q_values


        # Compute Loss
        q_values = self.online_net(obses_t)

        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        loss = nn.functional.smooth_l1_loss(action_q_values, targets)
        # loss = nn.functional.mse_loss(action_q_values, targets)

        # Gradient Descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update Target Network
        if step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, PATH=None):
        torch.save(self.online_net.state_dict(), PATH)

    def load(self, PATH=None):
        self.online_net.load_state_dict(torch.load(PATH))
        self.online_net.eval()
        self.target_net.load_state_dict(self.online_net.state_dict())

        # self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=5e-4)

    def expand_state(self, obs, action, agent):
        return obs[agent]


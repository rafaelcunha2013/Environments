from torch import nn
import torch
from collections import deque
import numpy as np
import random

from agent import Agent


GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=50000
MIN_REPLAY_SIZE=1000
EPSILON_START=1.0
EPSILON_END=0.02
EPSILON_DECAY=10000
TARGET_UPDATE_FREQ=1000*20

class BertesekasDDQN(Agent):
    def __init__(self, input_dim, output_dim, ag_id, env, load=False):
        super().__init__(input_dim, output_dim, ag_id, env, load=False)
        self.name = 'bertsekas_qlearning'

    def train(self, step):
        # Start Gradient Step
        transitions = random.sample(self.replay_buffer, BATCH_SIZE)

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


        ####################################
        ##########Bertsekas################
        #################################
        q1_values


        v_values = q1_values.max(dim=1, keepdim=True)[0] 
        q1_values = q2_values.max(dim=1, keepdim=True)[0] 
        q2_values = q3_values.max(dim=1, keepdim=True)[0] 
        q3_values = rews_t + GAMMA * (1-dones_t) * v_values

        # Compute Targets
        with torch.no_grad():
            if self.online_net.double:
                online_q_values = self.online_net(new_obses_t)
                argmax_online_q_values = online_q_values.argmax(dim=1, keepdim=True)

                target_q_values = self.target_net(new_obses_t)
                target_selected_q_values = torch.gather(input=target_q_values, dim=1, index=argmax_online_q_values)
                targets = rews_t + GAMMA * (1-dones_t) * target_selected_q_values
            else:
                target_q_values = self.target_net(new_obses_t)
                max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

                targets = rews_t + GAMMA * (1-dones_t) * max_target_q_values


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
        if step % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())



    def expand_state(self, obs, action, agent):
        if agent == 'agent_1':
            obs[agent] = np.insert(obs[agent], len(obs[agent]), action['agent_0'])
        elif agent == 'agent_2':
            obs[agent] = np.insert(obs[agent], len(obs[agent]), action['agent_0'])
            obs[agent] = np.insert(obs[agent], len(obs[agent]), action['agent_1'])
        return obs[agent]


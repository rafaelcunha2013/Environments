from torch import nn
import torch
from collections import deque
import numpy as np
import random

from agents.agent import Agent



class OneAgentTimeDDQN(Agent):
    def __init__(self, input_dim, output_dim, ag_id, env, parameter, load=False):
        super().__init__(input_dim, output_dim, ag_id, env, parameter, load=False)
        self.name = 'oneattime'


    def expand_state(self, obs, action, agent):
        if agent == 'agent_1':
            obs[agent] = np.insert(obs[agent], len(obs[agent]), action['agent_0'])
        elif agent == 'agent_2':
            obs[agent] = np.insert(obs[agent], len(obs[agent]), action['agent_0'])
            obs[agent] = np.insert(obs[agent], len(obs[agent]), action['agent_1'])
        return obs[agent]



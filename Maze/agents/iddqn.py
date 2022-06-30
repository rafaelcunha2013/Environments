from torch import nn
import torch
from collections import deque
import numpy as np
import random

from agents.agent import Agent



class IDDQN(Agent):
    def __init__(self, input_dim, output_dim, ag_id, env, parameter,load=False):
        super().__init__(input_dim, output_dim, ag_id, env, parameter,  load=False)
        self.name = 'IDDQN'
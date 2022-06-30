from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

import time
from pettingzoo.mpe import simple_spread_v2

from pettingzoo.agents.oneagentimeddqn import OneAgentTimeDDQN
from pettingzoo.agents.iddqn import IDDQN
from pettingzoo.agents.bertsekas_qlearning import BertesekasDDQN



GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=50000
MIN_REPLAY_SIZE=1000
EPSILON_START=1.0
EPSILON_END=0.02
EPSILON_DECAY=10000
TARGET_UPDATE_FREQ=1000*20

alg = 'oneattime'
alg = 'iddqn'


env = simple_spread_v2.env()
input_dim = env.observation_spaces['agent_0'].shape
output_dim = env.action_spaces['agent_0'].n
change_name = True
agents = {}

# Decides which type of agent will use
if alg == 'oneattime':
    agents['agent_0'] = OneAgentTimeDDQN(input_dim, output_dim, 'agent_0', env)
    agents['agent_1'] = OneAgentTimeDDQN((input_dim[0]+1,) , output_dim, 'agent_1', env)
    agents['agent_2'] = OneAgentTimeDDQN((input_dim[0]+2,), output_dim, 'agent_2', env)

if alg == 'iddqn':
    for agent in env.possible_agents:
        agents[agent] = IDDQN(input_dim, output_dim, agent, env)


# for agent in env.possible_agents:
#     PATH = '.\\PettingZoo\\models\\' + agent + '100'
#     agents[agent].load(PATH)

rew_buffer = deque([0.0], maxlen=100)
episode_reward = 0.0

def policy(obs, agent):

    epsilon = np.interp(agents[agent].step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    rnd_sample = random.random()

    if rnd_sample <= epsilon:
        action = np.random.randint(0, env.action_spaces[agent].n)
    else:
        action = agents[agent].online_net.act(obs)  
    return action


# Main training Loop

obs = {}
new_obs = {}
action = {}
done = {}
i_episode = 1
num_episodes = 3


env.reset()
for agent in env.agent_iter(env.max_num_agents):
    obs[agent], rew, done[agent], _ = env.last()
    
    if agent is not 'agent_0':
        obs[agent] = agents[agent].expand_state(obs, action, agent)

    action[agent] = policy(obs[agent], agent) if not done[agent] else None

    env.step(action[agent])



    
step = 0
for agent in env.agent_iter(1000000):
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    step += 1
    agents[agent].step += 1

    new_obs[agent], rew, done[agent], _ = env.last()
    new_obs[agent] = agents[agent].expand_state(new_obs, action, agent)
    transition = (obs[agent], action[agent], rew, done[agent], new_obs[agent])
    agents[agent].replay_buffer.append(transition)
    obs[agent] = new_obs[agent]

    action[agent] = policy(obs[agent], agent) if not done[agent] else None
    env.step(action[agent])  

    episode_reward += rew

    if all(done.values()):
        env.reset()

        rew_buffer.append(episode_reward/env.max_num_agents)
        episode_reward = 0.0

        done = {}
        for agent in env.agent_iter(env.max_num_agents):
            obs[agent], rew, done[agent], _ = env.last()
            if agent is not 'agent_0':
                obs[agent] = agents[agent].expand_state(obs, action, agent)
            action[agent] = policy(obs[agent], agent) if not done[agent] else None
            env.step(action[agent])

    # After solved, watch it play
    if len(rew_buffer) >= 100:
        # if np.mean(rew_buffer) >= -34:
        if np.mean(rew_buffer) >= -18:

            for agent in env.agent_iter(10000):
                obs[agent], rew, done[agent], _ = env.last()
                action[agent] = agents[agent].online_net.act(obs[agent]) if not done[agent] else None
                env.step(action[agent]) 
                env.render()
                if all(done.values()):
                    env.reset()

    if len(agents[agent].replay_buffer) > MIN_REPLAY_SIZE:
        agents[agent].train(step)

    # Logging
    if step % 1000 == 0:
        print()
        print('Step', step)
        print('Avg Rew', np.mean(rew_buffer))
        plt.scatter(step, np.mean(rew_buffer))
        if change_name:
            now = datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')
            change_name = False
        NAME_fig = '_111_' + agents[agent].name + '_' + now + ".png"
        plt.savefig(NAME_fig)
        plt.pause(0.001)
        plt.show(block=False)

    # Saving
    if step % 10000 == 0:
        if change_name:
            now = datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')
            change_name = False

        for agent in env.agents: 
            
            PATH = '.\\PettingZoo\\models\\' + agent + '_111_' + agents[agent].name + '_' + now
            agents[agent].save(PATH)
        print('saved')







    

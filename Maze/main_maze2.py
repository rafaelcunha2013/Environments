from torch import nn
import torch
# import gym
from collections import deque
import itertools
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import platform

# import time
# from pettingzoo.mpe import simple_spread_v2

from agents.oneagentimeddqn import OneAgentTimeDDQN
from agents.iddqn import IDDQN
# from agents.bertsekas_qlearning import BertesekasDDQN

from envi.gridworld2 import Shapes
# from agents.random_agent import RandomAgent
# from agents.qlearning import QLearningAgent
# from steps import TimeStep
# from envi.render import Render
from hyperparameters.hyper import hyperparameter

parameters = hyperparameter()

# GAMMA=0.99
# BATCH_SIZE=32
# BUFFER_SIZE=50000
# MIN_REPLAY_SIZE=1000
# EPSILON_START=1.0
# EPSILON_END=0.02
# EPSILON_DECAY=10000
# TARGET_UPDATE_FREQ=1000*20
# LEARNING_RATE = 5e-4
# HIDDEN_DIM = 256

# parameters = {}
# parameters['GAMMA'] = GAMMA
# parameters['BATCH_SIZE'] = BATCH_SIZE
# parameters['BUFFER_SIZE'] = BUFFER_SIZE
# parameters['MIN_REPLAY_SIZE'] = MIN_REPLAY_SIZE
# parameters['EPSILON_START'] = EPSILON_START
# parameters['EPSILON_END'] = EPSILON_END
# parameters['EPSILON_DECAY'] = EPSILON_DECAY
# parameters['TARGET_UPDATE_FREQ'] = TARGET_UPDATE_FREQ
# parameters['LEARNING_RATE'] = LEARNING_RATE
# parameters['HIDDEN_DIM'] = HIDDEN_DIM

GAMMA = parameters['GAMMA'] 
BATCH_SIZE = parameters['BATCH_SIZE'] 
BUFFER_SIZE = parameters['BUFFER_SIZE'] 
MIN_REPLAY_SIZE = parameters['MIN_REPLAY_SIZE'] 
EPSILON_START = parameters['EPSILON_START'] 
EPSILON_END = parameters['EPSILON_END'] 
EPSILON_DECAY = parameters['EPSILON_DECAY'] 
TARGET_UPDATE_FREQ = parameters['TARGET_UPDATE_FREQ'] 
LEARNING_RATE = parameters['LEARNING_RATE'] 
HIDDEN_DIM = parameters['HIDDEN_DIM'] 


alg = 'oneattime'
alg = 'iddqn'

maze=[
['1', ' ', ' ', ' ', ' ', '2', 'X', ' ', ' ', ' ', ' ', ' ', 'G'],
[' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
[' ', ' ', ' ', ' ', ' ', ' ', '1', ' ', ' ', ' ', ' ', ' ', ' '],
[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
[' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
['2', ' ', ' ', ' ', ' ', '3', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
['X', 'X', '3', ' ', 'X', 'X', 'X', 'X', 'X', ' ', '1', 'X', 'X'],
[' ', ' ', ' ', ' ', ' ', ' ', 'X', '2', ' ', ' ', ' ', ' ', '3'],
[' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
[' ', ' ', ' ', ' ', ' ', ' ', '2', ' ', ' ', ' ', ' ', ' ', ' '],
[' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
[' ', ' ', ' ', ' ', ' ', ' ', 'X', '3', ' ', ' ', ' ', ' ', '1']]
maze = np.array(maze)
# n_r, n_c = np.shape(maze)
# initial_position = False
# while not initial_position:
#     r, c = (np.random.randint(0, n_r), np.random.randint(0, n_c))
#     if maze[r, c] == ' ':
#         maze[r,c] = '_'
#         initial_position = True
# my_grid = Render(maze=maze)
# agent = RandomAgent()

def state_convert(state):
    return np.array(list(state[0])+list(state[1]), dtype=np.float32)

rewards = dict(zip(['1', '2', '3'], list(np.random.uniform(low=-1.0, high=1.0, size=3))))
rewards = dict(zip(['1', '2', '3'], [1.0, -1.0, 0.5]))
str_rewards = str(int(rewards['1']*10)) + str(int(rewards['2']*10)) + str(int(rewards['3']*10))

env = Shapes(maze=maze, shape_rewards=rewards)
# env = simple_spread_v2.env()

input_dim = (14,) #env.observation_spaces['agent_0'].shape
output_dim = 4 # env.action_spaces['agent_0'].n
change_name = True
agents = {}

# Decides which type of agent will use
if alg == 'oneattime':
    agents['agent_0'] = OneAgentTimeDDQN(input_dim, output_dim, 'agent_0', env, parameters)
    agents['agent_1'] = OneAgentTimeDDQN((input_dim[0]+1,) , output_dim, 'agent_1', env, parameters)
    agents['agent_2'] = OneAgentTimeDDQN((input_dim[0]+2,), output_dim, 'agent_2', env, parameters)

agent = 'iddqn'
agents[agent] = IDDQN(input_dim, output_dim, agent, env, parameters)
# if alg == 'iddqn':
#     for agent in env.possible_agents:
#         agents[agent] = IDDQN(input_dim, output_dim, agent, env)
file_name = {}
file_name[agent] = alg + '_' + agents[agent].name + '_' + str(BATCH_SIZE) + '_' + str(TARGET_UPDATE_FREQ) + '_' + str(HIDDEN_DIM) + '_' + str(int(LEARNING_RATE*1e4)) + '_' + str(BUFFER_SIZE) + '_' + str_rewards + '_'

# PATH = '.\\PettingZoo\\models\\iddqn_111_IDDQN_2022_06_28-11_47_19_AM'
# agents[agent].load(PATH)
# for agent in env.possible_agents:
#     PATH = '.\\PettingZoo\\models\\' + agent + '100'
#     agents[agent].load(PATH)

rew_buffer = deque([0.0], maxlen=100)
episode_reward = 0.0

def policy(obs, agent):

    epsilon = np.interp(agents[agent].step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    rnd_sample = random.random()

    if rnd_sample <= epsilon:
        # action = np.random.randint(0, env.action_spaces[agent].n)
        action = np.random.randint(0, 4)
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



# env.reset()
# for agent in env.agent_iter(env.max_num_agents):
#     obs[agent], rew, done[agent], _ = env.last()
    
#     if agent is not 'agent_0':
#         obs[agent] = agents[agent].expand_state(obs, action, agent)

#     action[agent] = policy(obs[agent], agent) if not done[agent] else None

#     env.step(action[agent])

done[agent] = False
obs[agent] = env.initialize()
obs[agent] = state_convert(obs[agent])
    
step = 0
episode_step = 0
old_mean = 0
while True: # step < 10000:
# for agent in env.agent_iter(1000000):
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    step += 1
    agents[agent].step += 1

    action[agent] = policy(obs[agent], agent) if not done[agent] else None

    new_obs[agent], rew, done[agent] = env.transition(action[agent])
    new_obs[agent] = state_convert(new_obs[agent])
    new_obs[agent] = agents[agent].expand_state(new_obs, action, agent)
    transition = (obs[agent], action[agent], rew, done[agent], new_obs[agent])
    agents[agent].replay_buffer.append(transition)
    obs[agent] = new_obs[agent]


    episode_reward += rew
    episode_step +=1
    env.max_num_agents = 1


    # if all(done.values()):
    if done[agent]:# or (episode_step>1000):
        obs[agent] = env.initialize()
        obs[agent] = state_convert(obs[agent])
        

        rew_buffer.append(episode_reward/env.max_num_agents)
        episode_reward = 0.0
        episode_step = 0

        done[agent] = False

        # done = {}
        # for agent in env.agent_iter(env.max_num_agents):
        #     obs[agent], rew, done[agent], _ = env.last()
        #     if agent is not 'agent_0':
        #         obs[agent] = agents[agent].expand_state(obs, action, agent)
        #     action[agent] = policy(obs[agent], agent) if not done[agent] else None
        #     env.step(action[agent])

    # After solved, watch it play
    if len(rew_buffer) >= 100:
        # if np.mean(rew_buffer) >= -34:
        if np.mean(rew_buffer) >= 8.0:
            obs[agent] = env.initialize()
            obs[agent] = state_convert(obs[agent])
            done[agent] = False

            for _ in range(10000):
            #for agent in env.agent_iter(10000):
                # action[agent] = policy(obs[agent], agent) if not done[agent] else None
                action[agent] = agents[agent].online_net.act(obs[agent]) if not done[agent] else None
                new_obs[agent], rew, done[agent] = env.transition(action[agent])
                obs[agent] = state_convert(new_obs[agent])


                # obs[agent], rew, done[agent], _ = env.last()
                # action[agent] = agents[agent].online_net.act(obs[agent]) if not done[agent] else None
                # env.step(action[agent]) 
                env.render(new_obs[agent][0])
                # env.render()
                if done[agent]: #all(done.values()):
                    # env.reset()
                    obs[agent] = env.initialize()
                    obs[agent] = state_convert(obs[agent])
                    done[agent] = False

    if len(agents[agent].replay_buffer) > MIN_REPLAY_SIZE:
        agents[agent].train(step)

    # Logging
    if step % 10000 == 0:
        # print()
        # print('Step', step)
        # print('Avg Rew', np.mean(rew_buffer))
        plt.scatter(step, np.mean(rew_buffer))
        if change_name:
            now = datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')
            change_name = False
        if platform.system() == 'Linux':
            sys_path = '/home/p285087/Environments/Maze/figures/'
        elif platform.system() == 'Windows':    
            sys_path = ".\\figures\\"
        elif platform.system() == 'Darwin':
            pass

        NAME_fig = sys_path  + now + '_' + file_name[agent] + "d" +  ".png"
        plt.savefig(NAME_fig)
        # plt.pause(0.001)
        # plt.show(block=False)

    # Saving
    if step % 10000 == 0:
        if change_name:
            now = datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')
            change_name = False

        if np.mean(rew_buffer) > old_mean:
            # for agent in env.agents: 
            if platform.system() == 'Linux':
                sys_path = '/home/p285087/Environments/Maze/models/'
            elif platform.system() == 'Windows':    
                sys_path = ".\\models\\"
            PATH = sys_path + now + '_' + file_name[agent] + "d"
            agents[agent].save(PATH)
            print('saved')
            old_mean = np.mean(rew_buffer)







    

import numpy as np
class RandomAgent:
    def __init__(self):
        pass

    def select_action(self, state):
        action = np.random.randint(0, 4)
        return action

    def observe(self, action, next_timestepv):
        pass

    def update(self):
        pass
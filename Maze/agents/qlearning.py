import numpy as np

class QLearningAgent:
    def __init__(self):
        self._step_size = 0.1
        self.q = None
        self.state = None
        self.action = None
        self.next_state = None
        self.next_action = None
        self._td_error = None

    def select_action(self, state):
        action = np.random.randint(0, 4)
        self.state = state
        self.action = action
        return action

    def observe(self, action, next_timestep):        
        g = next_timestep.discount
        s = self.state
        a = action
        r = next_timestep.reward
        next_s = next_timestep.observation
        next_a = np.argmax(self.q[s,a])

        self._td_error = r + g * self.q[next_s, next_a] - self.q[s,a]
        self.next_state = next_s



    def update(self):
        s = self.state
        a = self.action
        self.q[s,a] += self._step_size * self._td_error
        self.state = self.next_state
        
    def _get_q(self):
        return self.q

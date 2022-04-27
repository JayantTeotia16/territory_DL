import numpy as np
import random

from brain import Brain
from uniform_experience_replay import Memory as UER
from prioritized_experience_replay import Memory as PER

MAX_EPSILON = 1.0
MIN_EPSILON = 0.01

MIN_BETA = 0.4
MAX_BETA = 1.0

class Agent(object):
    
    epsilon = MAX_EPSILON
    beta = MIN_BETA

    def __init__(self, state_size, action_size, bee_index, brain_name, arguments):
        self.state_size = state_size
        self.action_size = action_size
        self.bee_index = bee_index
        self.learning_rate = arguments['learning_rate']
        self.gamma = 0.95
        self.brain = Brain(self.state_size, self.action_size, brain_name, arguments)
        self.memory_model = arguments['memory']

        if self.memory_model == 'UER':
            self.memory = UER(arguments['memory_capacity'])

        elif self.memory_model == 'PER':
            self.memory = PER(arguments['memory_capacity'], arguments['prioritization_scale'])

        else:
            print('Invalid memory model!')

        self.target_type = arguments['target_type']
        self.update_target_frequency = arguments['target_frequency']
        self.max_exploration_step = arguments['maximum_exploration']
        self.batch_size = arguments['batch_size']
        self.step = 0
        self.test = arguments['test']
        if self.test:
            self.epsilon = MIN_EPSILON

    def greedy_actor(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return self.brain.predict_one_sample(state)

    

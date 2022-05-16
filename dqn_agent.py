import numpy as np
import random
import tensorflow as tf

from brain import Brain
from uniform_experience_replay import Memory as UER
from prioritized_experience_replay import Memory as PER

MAX_EPSILON = 1.0
MIN_EPSILON = 0.0 ######  Changed (0.01)

MIN_BETA = 0.4
MAX_BETA = 1.0

n = 100

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

    def greedy_actor(self, state, max_act):
        if np.random.rand() <= self.epsilon:
            a = np.zeros(self.action_size)
            b = np.random.randint(0,max_act+1)
            idx = np.random.choice(range(self.state_size), size = b, replace=False)
            for _ in range(b):
                a[idx[_]] = 1
            return a
        else:
            a = self.brain.predict_one_sample(state)
            print(a,"CHECK")
            for i in range(len(a)):
                if abs(a[i]) >= 0.5:
                    a[i] = 1
                else:
                    a[i] = 0
            
            return a
            
    def observe(self, sample):

        if self.memory_model == 'UER':
            self.memory.remember(sample)

        elif self.memory_model == 'PER':
            _, _, errors = self.find_targets_per([[0, sample]])
            self.memory.remember(sample, errors[0])

        else:
            print('Invalid memory model!')
            
    def find_targets_per(self, batch):
        batch_len = len(batch)

        states = np.array([o[1][0] for o in batch])
        states_ = np.array([o[1][3] for o in batch])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)
        pTarget_ = self.brain.predict(states_, target=True)

        x = np.zeros((batch_len, self.state_size))
        y = np.zeros((batch_len, self.action_size))
        errors = np.zeros(batch_len)

        for i in range(batch_len):
            o = batch[i][1]
            s = o[0]
            a = o[1][self.bee_index]
            r = o[2]
            s_ = o[3]
            done = o[4]

            t = p[i]
            old_value = t[a]
            if done:
                t[a] = r
            else:
                if self.target_type == 'DDQN':
                    t[a] = r + self.gamma * pTarget_[i][np.argmax(p_[i])]
                elif self.target_type == 'DQN':
                    t[a] = r + self.gamma * np.amax(pTarget_[i])
                else:
                    print('Invalid type for target network!')

            x[i] = s
            y[i] = t
            errors[i] = np.abs(t[a] - old_value)

        return [x, y, errors]
        
    def find_targets_uer(self, x, y, r):
        states = x

        p = self.brain.predict(states)
        pTarget = self.brain.predict(states, target=True)
        
        t = p + (y - p)/n

        return t
        
    def decay_epsilon(self):
        # slowly decrease Epsilon based on our experience
        self.step += 1

        if self.test:
            self.epsilon = MIN_EPSILON
            self.beta = MAX_BETA
        else:
            if self.step < self.max_exploration_step:
                self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * (self.max_exploration_step - self.step)/self.max_exploration_step
                self.beta = MAX_BETA + (MIN_BETA - MAX_BETA) * (self.max_exploration_step - self.step)/self.max_exploration_step
            else:
                self.epsilon = MIN_EPSILON
                
    def replay(self):

        if self.memory_model == 'UER':
            batch = self.memory.sample(self.batch_size)
            x, y = self.find_targets_uer(batch)
            self.brain.train(x, y)

        elif self.memory_model == 'PER':
            [batch, batch_indices, batch_priorities] = self.memory.sample(self.batch_size)
            x, y, errors = self.find_targets_per(batch)

            normalized_batch_priorities = [float(i) / sum(batch_priorities) for i in batch_priorities]
            importance_sampling_weights = [(self.batch_size * i) ** (-1 * self.beta)
                                           for i in normalized_batch_priorities]
            normalized_importance_sampling_weights = [float(i) / max(importance_sampling_weights)
                                                      for i in importance_sampling_weights]
            sample_weights = [errors[i] * normalized_importance_sampling_weights[i] for i in range(len(errors))]

            self.brain.train(x, y, np.array(sample_weights))

            self.memory.update(batch_indices, errors)

        else:
            print('Invalid memory model!')
            
    def train(self, x, y, r):
    #Train using reward
        #x = tf.expand_dims(x, axis=1)
        #x = x.reshape((-1,1))
        #y = tf.expand_dims(y, axis=1)
        a = self.find_targets_uer(x, y, r)
        self.brain.train(x, a)
            
    def update_target_model(self):
        if self.step % self.update_target_frequency == 0:
            self.brain.update_target_model()

    

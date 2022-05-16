import random
import operator
import numpy as np
import pygame
import sys
import os
from function import fun

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
ORANGE = (255, 128, 0)

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 60
HEIGHT = 60

# This sets the margin between each cell
MARGIN = 1

class territory:

    def __init__(self, args, current_path):
        # Remove self.reward_mode = args['reward_mode']
        self.num_agents = args['agents_number']
        self.num_landmarks = 0
        self.grid_size = args['grid_size'] ### default 10
        #self.grid_size = 10
        self.state_size = (self.grid_size *4) -4
        self.agents_positions = []
        self.landmarks_positions = []
        self.agents_positions_repeat = []
        self.state_repeat = []


        # enables visualizer
        

        self.cells = []
        self.positions_idx = []
        self.idx_val = []
        self.b3 = []

        # self.agents_collide_flag = args['collide_flag']
        # self.penalty_per_collision = args['penalty_collision']
        self.num_episodes = 0
        self.terminal = False

    def set_positions_idx(self):

        cells = [(i, j) for i in range(0, self.grid_size) for j in range(0, self.grid_size)]
        positions_idx = []
        b2 = []
        b1 = []
        self.b3 = []
        a = [i for i in range(10)]
        for _ in a:
            b2.append(_)
        a = [19,29,39,49,59,69,79,89,10,20,30,40,50,60,70,80]
        for _ in a:
            b2.append(_)
        a = [i for i in range(90,100)]
        for _ in a:
            b2.append(_)
        for _ in b2:    
            self.b3.append(_)
        a1 = np.array([44,53,56])

        if self.game_mode == 0:
            # first enter the positions for the landmarks and then for the agents. If the grid is n*n, then the
            # positions are
            #  0                1             2     ...     n-1
            #  n              n+1           n+2     ...    2n-1
            # 2n             2n+1          2n+2     ...    3n-1
            #  .                .             .       .       .
            #  .                .             .       .       .
            #  .                .             .       .       .
            # (n-1)*n   (n-1)*n+1     (n-1)*n+2     ...   n*n-1
            # , e.g.,
            # positions_idx = [0, 6, 23, 24] where 0 and 6 are the positions of landmarks and 23 and 24 are positions
            # of agents
            positions_idx = []
            
        if self.game_mode == 1:
            idx = np.random.choice(range(self.state_size), size=self.num_landmarks, replace=False)
            idx_value = np.zeros((self.state_size),4)
            for _ in range(self.num_landmarks):
                idx_value[idx[_]][3] = np.random.randint(5,10)
                b1.append(b2[idx[_]])
            positions_idx = np.concatenate((b1,a1))
        return [cells, positions_idx, idx_value]

    def reset(self):  # initialize the world

        self.terminal = False
        self.num_landmarks = np.random.randint(1,6)
        [self.cells, self.positions_idx, self.idx_val] = self.set_positions_idx()

        # separate the generated position indices for pursuers, and evaders
        landmarks_positions_idx = self.positions_idx[0:self.num_landmarks]
        agents_positions_idx = self.positions_idx[self.num_landmarks:self.num_landmarks + self.num_agents]

        # map generated position indices to positions
        self.landmarks_positions = [self.cells[pos] for pos in landmarks_positions_idx]
        self.agents_positions = [self.cells[pos] for pos in agents_positions_idx]
        self.agents_positions_repeat = self.agents_positions 

        initial_state_vis = list(sum(self.landmarks_positions + self.agents_positions, ()))
        self.state_repeat = initial_state_vis

        return initial_state_vis, self.idx_val
        
    def action_space(self):
        return (self.grid_size *4) -4
        
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

    

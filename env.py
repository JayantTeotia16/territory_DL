import random
import operator
import numpy as np
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
        #self.game_mode = args['game_mode']
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
        self.vel = []
        
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
        a1 = [44,53,56]

        x = np.zeros((36,1))
        y = np.zeros((36,1))
        cd = np.zeros((36,1))
        cd_num = self.num_agents
        #self.num_agents = cd_num
        cd_idx = np.random.choice(range(self.state_size), size=cd_num, replace=False)
        for _ in range(cd_num):
            cd[cd_idx[_]] = 1
        for i in range(0,10):   
            y[i] = i
        for i in range(10,19):   
            x[i] = i-9     
            y[i] = 9
        for i in range(19,27):
            x[i] = 9
            y[i] = 27 - i
        for i in range(27,36):
            x[i] = 36- i
        #x = np.reshape(x,(1,36))
        #y = np.reshape(x,(1,36))
        idx = np.random.choice(range(self.state_size), size=self.num_landmarks, replace=False)
        idx_value = np.ones((36,1))*10000
        for _ in range(self.num_landmarks):
            idx_value[idx[_]] = np.random.randint(5,10)
            b1.append(b2[idx[_]])
        #for _ in range(cd_num):
        #   b1.append(b2[cd_idx[_]])
        positions_idx = np.concatenate((b1,a1))
        fin_in = np.concatenate((x, y, idx_value, cd), axis = 1)
        #print(np.shape(fin_in),'ay')
        return [cells, positions_idx, idx_value, fin_in]

    def reset(self):  # initialize the world

        self.terminal = False
        self.num_landmarks = np.random.randint(1,6)
        print(self.num_landmarks,"NUMLAND")
        [self.cells, self.positions_idx, self.idx_val, fin_in] = self.set_positions_idx()
        #self.idx_val = np.reshape(self.idx_val,(1,36))
        # separate the generated position indices for pursuers, and evaders
        landmarks_positions_idx = self.positions_idx[0:self.num_landmarks]
        agents_positions_idx = self.positions_idx[self.num_landmarks:self.num_landmarks + self.num_agents]

        # map generated position indices to positions
        self.landmarks_positions = [self.cells[pos] for pos in landmarks_positions_idx]
        self.agents_positions = [self.cells[pos] for pos in agents_positions_idx]
        self.agents_positions_repeat = self.agents_positions 

        initial_state_vis = list(sum(self.landmarks_positions + self.agents_positions, ()))
        self.state_repeat = initial_state_vis

        return initial_state_vis, self.idx_val, fin_in
        
    def action_space(self):
        return (self.grid_size *4) -4
        
    def step(self, agents_actions):
        action_idx = []    
        reward = 0    
        time = []
        scaff_act = np.zeros((self.num_agents,self.state_size))
        indx = []
        for j,i in enumerate(self.idx_val):
            if i < 100:
                time.append(i)    

        act = fun(self.agents_positions , self.landmarks_positions, time)
        for i in range(len(act)):
            indx.append([])
            for j in range(len(act[i])):
                a = self.landmarks_positions[int(act[i][j])]
                indx[i].append(a[0]*10 + a[1])
        for k in range(self.num_agents):
            indx[k].sort()
            l = 0        
            for idd,j in enumerate(self.b3):
                if l >= len(indx[k]):
                    break
                if j == indx[k][l]:
                    scaff_act[k][idd] = 1
                    l+=1

        for j in range(self.num_agents):
            act = []
            for i in range(self.state_size):
                if agents_actions[j][i] == 1:
                    act.append(self.b3[i])
            action_idx.append([self.cells[pos] for pos in act])
        for i in range(self.state_size): # state size is same as action space
            for j in range(len(agents_actions)):
                if agents_actions[j][i] == 1:
                    if self.idx_val[i] > 100:
                        reward -= (5/self.num_landmarks)
                    else:
                        reward += (5/self.num_landmarks)
                    self.idx_val[i] = 10000 
        if sum(self.idx_val) == 10000*self.state_size:
            reward += 5
        print(np.shape(agents_actions))
        rew_eq = np.equal(scaff_act,agents_actions)
        for i in range(self.num_agents):
            for j in range(self.state_size):
                if rew_eq[i][j] == False:
                    reward -= (5/self.num_landmarks)
        self.terminal = True
        return self.idx_val, reward, self.terminal, scaff_act        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

    

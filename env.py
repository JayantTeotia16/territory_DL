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
        self.game_mode = args['game_mode']
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

        self.render_flag = args['render']
        self.recorder_flag = args['recorder']
        # enables visualizer
        if not self.render_flag:
            [self.screen, self.my_font] = self.gui_setup()
            self.step_num = 1

            resource_path = os.path.join(current_path, 'environments')  # The resource folder path
            resource_path = os.path.join(resource_path, 'agents_landmarks')  # The resource folder path
            image_path = os.path.join(resource_path, 'images')  # The image folder path

            img = pygame.image.load(os.path.join(image_path, 'agent.jpg')).convert()
            self.img_agent = pygame.transform.scale(img, (WIDTH, WIDTH))
            img = pygame.image.load(os.path.join(image_path, 'landmark.jpg')).convert()
            self.img_landmark = pygame.transform.scale(img, (WIDTH, WIDTH))
            img = pygame.image.load(os.path.join(image_path, 'agent_landmark.jpg')).convert()
            self.img_agent_landmark = pygame.transform.scale(img, (WIDTH, WIDTH))
            img = pygame.image.load(os.path.join(image_path, 'agent_agent_landmark.jpg')).convert()
            self.img_agent_agent_landmark = pygame.transform.scale(img, (WIDTH, WIDTH))
            img = pygame.image.load(os.path.join(image_path, 'agent_agent.jpg')).convert()
            self.img_agent_agent = pygame.transform.scale(img, (WIDTH, WIDTH))

            if not self.recorder_flag:
                self.snaps_path = os.path.join(current_path, 'results_territory')  # The resource folder path
                self.snaps_path = os.path.join(self.snaps_path, 'snaps')  # The resource folder path

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
            idx_value = np.ones(self.state_size)*10000
            for _ in range(self.num_landmarks):
                idx_value[idx[_]] = np.random.randint(10,30)
                b1.append(b2[idx[_]])
            positions_idx = np.concatenate((b1,a1))
        return [cells, positions_idx, idx_value]

    def reset(self):  # initialize the world

        self.terminal = False
        self.num_landmarks = np.random.randint(1,20)
        print(self.num_landmarks,"NUMLAND")
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
        new_agent_pos = self.update_positions(self.agents_positions, action_idx, self.landmarks_positions)
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
        
    def update_positions(self, ag_pos_list, act_list, lm_pos_list):
    
        final_positions = []      
        for idx in range(len(ag_pos_list)):
            if len(act_list[idx]) != 0: 
                final_positions.append(act_list[idx][len(act_list[idx]) - 1])    
            else:
                final_positions.append(ag_pos_list[idx])        
        return final_positions
        
    def gui_setup(self):

        # Initialize pygame
        pygame.init()

        # Set the HEIGHT and WIDTH of the screen
        board_size_x = (WIDTH + MARGIN) * self.grid_size
        board_size_y = (HEIGHT + MARGIN) * self.grid_size

        window_size_x = int(board_size_x)
        window_size_y = int(board_size_y * 1.2)

        window_size = [window_size_x, window_size_y]
        screen = pygame.display.set_mode(window_size)

        # Set title of screen
        pygame.display.set_caption("Drone assignment")

        myfont = pygame.font.SysFont("monospace", 30)

        return [screen, myfont]
        
    def render(self,episode_num):

        pygame.time.delay(10)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        self.screen.fill(BLACK)
        text = self.my_font.render("Episode: {0}".format(episode_num), 1, WHITE)
        self.screen.blit(text, (5, 15))

        for row in range(self.grid_size):
            for column in range(self.grid_size):
                pos = (row, column)

                frequency = self.find_frequency(pos, self.agents_positions)

                if pos in self.landmarks_positions and frequency >= 1:
                    if frequency == 1:
                        self.screen.blit(self.img_agent_landmark,
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
                    else:
                        self.screen.blit(self.img_agent_agent_landmark,
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))

                elif pos in self.landmarks_positions:
                    self.screen.blit(self.img_landmark,
                                     ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))

                elif frequency >= 1:
                    if frequency == 1:
                        self.screen.blit(self.img_agent,
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
                    elif frequency > 1:
                        self.screen.blit(self.img_agent_agent,
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
                    else:
                        print('Error!')
                else:
                    pygame.draw.rect(self.screen, WHITE,
                                     [(MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50, WIDTH,
                                      HEIGHT])

#        if not self.recorder_flag:
#            file_name = "%04d.png" % episode_num
#            pygame.image.save(self.screen, os.path.join(self.snaps_path, file_name))

        if not self.terminal:
            self.step_num += 1
            
    def find_frequency(self, a, items):
        freq = 0
        for item in items:
            if item == a:
                freq += 1

        return freq
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

    

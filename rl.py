import numpy as np
import os
import random
import argparse
import pandas as pd
from env import territory
from dqn_agent import Agent
import glob
import time
from function import fun

ARG_LIST = ['learning_rate', 'optimizer', 'memory_capacity', 'batch_size', 'target_frequency', 'maximum_exploration',
            'first_step_memory', 'replay_steps', 'number_nodes', 'target_type', 'memory',
            'prioritization_scale', 'dueling', 'agents_number', 'grid_size', 'game_mode']
            
# Remove ARG_LIST = ['learning_rate', 'optimizer', 'memory_capacity', 'batch_size', 'target_frequency', 'maximum_exploration',
#            'max_timestep', 'first_step_memory', 'replay_steps', 'number_nodes', 'target_type', 'memory',
#            'prioritization_scale', 'dueling', 'agents_number', 'grid_size', 'game_mode', 'reward_mode']


def get_name_brain(args, idx):

    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])

    return './results_territory/weights_files/' + file_name_str + '_' + str(idx) + '.h5'


def get_name_rewards(args):

    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])

    return './results_territory/rewards_files/' + file_name_str + '.csv'


def get_name_timesteps(args):

    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])

    return './results_territory/timesteps_files/' + file_name_str + '.csv'

class Environment(object):

    def __init__(self, arguments):
        current_path = os.path.dirname(__file__)  # Where your .py file is located
        self.env = territory(arguments, current_path)
        self.episodes_number = arguments['episode_number']
        self.render = arguments['render']
        self.recorder = arguments['recorder']
        # Remove self.max_ts = arguments['max_timestep']
        self.test = arguments['test']
        self.filling_steps = arguments['first_step_memory']
        self.steps_b_updates = arguments['replay_steps']
        self.max_random_moves = arguments['max_random_moves']
        self.num_agents = arguments['agents_number']
        self.num_landmarks = self.env.num_landmarks
        self.game_mode = arguments['game_mode']
        self.grid_size = arguments['grid_size']

    def run(self, agents, file1, file2):
        total_step = 0
        rewards_list = []
        max_score = -10000
        for episode_num in range(self.episodes_number):
            state_vis, state = self.env.reset()
            self.num_landmarks = self.env.num_landmarks

            state = state.reshape((1,len(state)))
            #print(state,"1")    
            state = np.array(state)
            #print(state,"2")
            #state = state.ravel()
            #print(state,"3")
            done = False
            actions = []
            
            for agent in agents:
                actions.append(agent.greedy_actor(state,self.num_landmarks))
                
            next_state, reward, done, sca_id = self.env.step(actions)
            #print(sca_id,"SCAFF")
            next_state = np.array(next_state)
            next_state = next_state.ravel()
            #print([state],"ss")
            print(sca_id,"1")
            if not self.test:          
                for idx,agent in enumerate(agents):
                    
                    aa = sca_id[idx]
                    aa = aa.reshape((1,len(aa)))
                    agent.train(state,aa, reward)                    
                    agent.decay_epsilon()
                    if episode_num % 10 == 0:
                        agent.update_target_model()
                       
                
            rewards_list.append(reward)
            
            print("Episode {p}, Score: {s}, Final Step: {t}, Goal: {g}".format(p=episode_num, s=reward,
                                                                               t=episode_num, g=done))

                    
            if not self.test:
                if episode_num % 500 == 0:
                    df = pd.DataFrame(rewards_list, columns=['score'])
                    df.to_csv(file1)

                    if reward >= max_score:
                        for agent in agents:
                            agent.brain.save_model()
                        max_score = reward
            
                                   
if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    # DQN Parameters
    parser.add_argument('-e', '--episode-number', default=100000, type=int, help='Number of episodes')
    parser.add_argument('-l', '--learning-rate', default=0.00005, type=float, help='Learning rate')
    parser.add_argument('-op', '--optimizer', choices=['Adam', 'RMSProp'], default='Adam',
                        help='Optimization method')
    parser.add_argument('-m', '--memory-capacity', default=1000000, type=int, help='Memory capacity')
    parser.add_argument('-b', '--batch-size', default=1, type=int, help='Batch size')
    parser.add_argument('-t', '--target-frequency', default=100, type=int,
                        help='Number of steps between the updates of target network')
    parser.add_argument('-x', '--maximum-exploration', default=1000, type=int, help='Maximum exploration step')
    parser.add_argument('-fsm', '--first-step-memory', default=0, type=float,
                        help='Number of initial steps for just filling the memory')
    parser.add_argument('-rs', '--replay-steps', default=4, type=float, help='Steps between updating the network')
    parser.add_argument('-nn', '--number-nodes', default=256, type=int, help='Number of nodes in each layer of NN')
    parser.add_argument('-tt', '--target-type', choices=['DQN', 'DDQN'], default='DQN')
    parser.add_argument('-mt', '--memory', choices=['UER', 'PER'], default='PER')
    parser.add_argument('-pl', '--prioritization-scale', default=0.5, type=float, help='Scale for prioritization')
    parser.add_argument('-du', '--dueling', action='store_true', help='Enable Dueling architecture if "store_false" ')

    parser.add_argument('-gn', '--gpu-num', default='2', type=str, help='Number of GPU to use')
    parser.add_argument('-test', '--test', default=0, help='Enable the test phase if "1"')

    # Game Parameters
    parser.add_argument('-k', '--agents-number', default=8, type=int, help='The number of agents')
    # Remove parser.add_argument('-lm', '--landmarks-number', default=5, type=int, help='The number of landmarks')
    parser.add_argument('-g', '--grid-size', default=10, type=int, help='Grid size')
    # Remove parser.add_argument('-ts', '--max-timestep', default=100, type=int, help='Maximum number of timesteps per episode')
    parser.add_argument('-gm', '--game-mode', choices=[0, 1], type=int, default=1, help='Mode of the game, '
                                                                                        '0: landmarks and agents fixed, '
                                                                                        '1: landmarks and agents random ')

    # Remove parser.add_argument('-rw', '--reward-mode', choices=[0, 1, 2], type=int, default=0, help='Mode of the reward,'
    #                                                                                         '0: Only terminal rewards'
    #                                                                                         '1: Partial rewards '
    #                                                                                         '(number of unoccupied landmarks'
    #                                                                                         '2: Full rewards '
    #                                                                                        '(sum of dinstances of agents to landmarks)')

    parser.add_argument('-rm', '--max-random-moves', default=0, type=int,
                        help='Maximum number of random initial moves for the agents')


    # Visualization Parameters
    parser.add_argument('-r', '--render', action='store_false', help='Turn on visualization if "store_false"')
    parser.add_argument('-re', '--recorder', default=0, help='Store the visualization as a movie '
                                                                       'if "store_false"')

    args = vars(parser.parse_args())
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_num']

    env = Environment(args)

    state_size = env.env.state_size
    action_space = env.env.action_space()
    print(action_space)

    all_agents = []
    
    for b_idx in range(args['agents_number']):

        brain_file = get_name_brain(args, b_idx)
        all_agents.append(Agent(state_size, action_space, b_idx, brain_file, args))

    rewards_file = get_name_rewards(args)
    timesteps_file = get_name_timesteps(args)

    env.run(all_agents, rewards_file, timesteps_file)
    
    
    
    
    

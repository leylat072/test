#!/usr/bin/env python
# coding: utf-8
#ff
# In[3]:


from agent1 import DDPG

import sys
import pandas as pd
import csv
import numpy as np
#from AnimatedPlot import AnimatedPlot
import time
import math
from engine1 import TetrisEngine 
#%matplotlib notebook
#myplot1 = AnimatedPlot()
# Modify the values below to give the quadcopter a different starting position.
runtime = 7                                     # time limit of the episode
num_episodes = 10000
file_output = 'data.txt' 
engine = TetrisEngine(6,20)
#print(engine.state_size)
#print(engine.width)
#print(engine.height)
agent = DDPG(engine) 
labels = ['episod', 'total_reward']
results = {x : [] for x in labels}
## Run the simulation, and save the results.
#sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
mmm= False
with open(file_output, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(labels)  
    best_total_reward = 0
    for i_episode in range(1, num_episodes+1):
        state = agent.reset_episode() # start a new episode
        timestampStart = time.time();
        total_reward = 0
        j =0
        while True:
            timestamp = time.time() - timestampStart
            #print('---------------------------state------------------------------------------')
            #print(state)
            #print(engine)
            action = agent.act(state) 
            #print('sta')
            #print(state)
            #if action[0] == 1 :
                #print('action[0]')
                #print(action[0])
            next_state , reward, dead= engine.step(action) 
            #print('next_state')
            #print(next_state)
            #print('------------------------------nextstate---------------------------------------')
            #print(next_state)
            print(engine)
            array = np.array(action).reshape(-1,)
            #print(array.argmax())
            writer.writerow(array)
            writer.writerow('action')
            total_reward += reward
            if total_reward > best_total_reward:
                best_total_reward = total_reward
            agent.step(action, reward, next_state, dead)
            #print(reward)
            state = next_state
            #print(engine)
            ##########################################
            if len(agent.memory) > agent.batch_size and mmm == True:
                experiences = agent.memory.sample()
                states = np.vstack([e.state for e in experiences if e is not None]).reshape(-1,120)
                actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, agent.action_size)
                #actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
            
                rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
                dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
                next_states = np.vstack([e.next_state for e in experiences if e is not None]).reshape(-1, agent.state_size)
                m =states[0]
                #state =  np.zeros(shape=(20, 6), dtype=np.bool)
                c = np.zeros(shape=(20, 6), dtype=np.bool)
                for i in range(0,20):
                    for j in range(0,6):
                        c[i][j] = m[j + 6* i]
                        
                _states = []
                for i in range(0,self.buffer_size-1): 
                    _states.append(agent.getState2(states[i]))
                print(_states)
                print(state)
                print('fffff')
                print(c)
                print(len(states))
                print('+++++++++++++++++++++')
                print((states[0]))
                print('+++++++++++++++++++++')
                print(len(actions))
                print('+++++++++action[0]++++++++++++')
                print(actions[0])
                print('+++++++++++++++++++++')
                print(len(rewards))
                print('+++++++++++++++++++++')
                print((rewards[0]))
                print('+++++++++++++++++++++')
                print(len(dones))
                print('+++++++++++++++++++++')
                print(dones[0])
                print('+++++++++++++++++++++')
                print(len(next_states))
                print('+++++++++++++++++++++')
                print(next_states[0])
                print('+++++++++++++++++++++')
                #mmm =False
            #print(reward)
            #print(action)
            #print('ddds')
            #print(action[0])
            #print(action[0]* 5)
            #print(math.ceil(action[0]))
            #print(action)
            #engine._new_piece()
            #print(engine.shape_idx)
            ###########################################
            #engine._update_score(0)
            #print(engine.score)
            # CALL THE METHOD plot(task)
            writer.writerow(action)
            #writer.writerow(engine)
            #writer.writerow(reward)
            
            if dead:
                print('end')
                print(i_episode)
                to_write = [i_episode] + [total_reward]
                #for j in range(len(labels)):
                    #results[labels[j]].append(to_write[j])
                writer.writerow(to_write)
                writer.writerow(action)
                writer.writerow('action')
                print("\rEpisode = {:4d}, total_reward = {:7.3f} (best = {:7.3f})".format(
                    i_episode, total_reward, best_total_reward), end="")
                break
            j = j+1
        sys.stdout.flush()
print('ddd')



        



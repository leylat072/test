#!/usr/bin/env python
# coding: utf-8

# In[3]:


from agent1 import DDPG

import sys
import pandas as pd
import csv
import numpy as np
#from AnimatedPlot import AnimatedPlot
import time
from engine1 import TetrisEngine 
#%matplotlib notebook
#myplot1 = AnimatedPlot()
# Modify the values below to give the quadcopter a different starting position.
runtime = 7                                     # time limit of the episode
num_episodes = 10000
file_output = 'data.txt' 
engine = TetrisEngine(10,20)
#print(engine.state_size)
#print(engine.width)
#print(engine.height)
agent = DDPG(engine) 
labels = ['episod', 'total_reward']
results = {x : [] for x in labels}
## Run the simulation, and save the results.
#sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
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
            #print(state)
            action = agent.act(state) 
            #print(action)
            next_state , reward, dead= engine.step(action) 
            #print(next_state)
            array = np.array(action).reshape(-1,)
            #print(array.argmax())
            writer.writerow(array)
            writer.writerow('action')
            total_reward += reward
            if total_reward > best_total_reward:
                best_total_reward = total_reward
            agent.step(action, reward, next_state, dead)
            #print()
            state = next_state
            print(engine)
            ##########################################
         
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
                #writer.writerow(to_write)
                print("\rEpisode = {:4d}, total_reward = {:7.3f} (best = {:7.3f})".format(
                    i_episode, total_reward, best_total_reward), end="")
                break
            j = j+1
        sys.stdout.flush()
print('ddd')

        



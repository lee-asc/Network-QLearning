

import random
import networkx as nx
import time
import os 
import numpy as np
import pickle
import json
import ast
from pickle import load
from matplotlib.pylab import plt

from math import floor

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras import Sequential
from keras.optimizers import Adam
from keras.optimizers import SGD
from tensorflow.keras.regularizers import l2

from tqdm import tqdm



max_steps = 50
episodes = 1000 #number of past episodes to observe and save into replay memory
replay_memory = []
done = False
total_steps = 0 

for episode in tqdm(range(episodes), ascii =  True, unit = "episode"):
    start_time = time.time()
    G = initial_g(G, initial_infected)   #Reset to initialization 
    step = 0
    
    while not done:
        inactive = [x for x,y in G.nodes(data=True) if y['state'] == 0]
        G_inactive = G.subgraph(inactive)
        if len(G_inactive.nodes()) == 0:
            break
        
        current_state = np.array(observation(G))     

        select_inactive = random.sample(list(G_inactive.nodes()), 1) 
        select_inactive = str(int(select_inactive[0]))     
        G.nodes[select_inactive]['state'] = 2          

        G, n_active_before, n_active_after = diff_step(G)

        r = reward(G, max_steps, n_active_before, n_active_after)

        step += 1
        
        if step == max_steps:
            break
        
        next_state = np.array(observation(G))



        experience = current_state, r, next_state
        replay_memory.append(experience)
        

        total_steps += 1
        
        
    end_time = time.time()
    episode_time = end_time - start_time
    print(f"Episode {episode +1} Time = {episode_time}")
            

print(f'The capacity/size of the replay memory is {len(replay_memory)}')
print(f'The Episode Length benchmark is {total_steps/episodes}')
###Want lower benchmark so have more room for learning

max_steps = 50
episodes = 1000

discount = 0.1
learning_rate = 0.0001  #Adjust based on minibatch update size
update_target = 5

replay_memory_size = len(replay_memory) #Sample from previous experiences of the exploration size
minibatch_size = 256  #Perform GD-Update on a sample past experiences
#IMPORTANT: The ratio of minibatch size to replay memory size is non-trivial. 
#Use minibatch_size = 256, 128, 64, 32, 16


#REPLAY MEMORY SIZE NEEDS TO BE VERY LARGE SINCE IT WOULD PREVENT CATASTROPIC FORGETTING
#THE IMP ENVIRONMENT IS HIGHLY STOCHASTIC, AND HENCE AGENT WOULD NEED TO REMEMBER A LARGE VARIETY OF STATES


#records every diffusion step
active_per_episode = []
immunized_per_episode = []

#records every episode
count_nonrandom_action = []
count_steps = []
count_episode_reward = []

total_steps = 0

model = newmodel(learning_rate)

target_model = model

loss_history = []


for episode in tqdm(range(episodes), ascii =  True, unit = "episode"):
    start_time = time.time()
    G = initial_g(G, initial_infected)   #Reset to initialization 
    step = 0
    nonrandom_action = 0
    episode_reward = 0

    active_per_training = []
    immunized_per_training = []
    
    while not done:
        inactive = [x for x,y in G.nodes(data=True) if y['state'] == 0]
        G_inactive = G.subgraph(inactive)
        if len(G_inactive.nodes()) == 0:
            break
        
        current_state = np.array(observation(G))     

        q_policy = model.predict(onehot(current_state)[None], verbose = False)
        action = np.argmax(q_policy) #Position of node in observation space
        
        
        
        #IMMUNIZATION STEP
        if current_state[action] == 0: #Immunise if only inactive node
            action = list(G.nodes())[action] #Label of node in graph
            G.nodes[action]['state'] = 2
            nonrandom_action += 1
        else: 
            select_inactive = random.sample(list(G_inactive.nodes()), 1) 
            select_inactive = str(int(select_inactive[0]))     
            G.nodes[select_inactive]['state'] = 2 
            
        G, n_active_before, n_active_after = diff_step(G)

        r = reward(G, max_steps, n_active_before, n_active_after)
        
        step += 1
        if step == max_steps:
            break

        next_state = np.array(observation(G))

        

        experience = current_state, r, next_state
        replay_memory.append(experience)
        replay_memory.pop(0)
            
            
        
        episode_reward += r

        active = [x for x,y in G.nodes(data=True) if y['state'] == 1]
        immunized = [x for x,y in G.nodes(data=True) if y['state'] == 2]
        
        active_per_training.append(len(active))
        immunized_per_training.append(len(immunized))
        
        
        active_per_episode.append(active_per_training)
        immunized_per_episode.append(immunized_per_training)
    
        
        
        #TRAINING DEEPQAGENT
        if total_steps % update_target == 0:  #Copy network parameters every C steps
            target_model.set_weights(model.get_weights()) 


        minibatch = random.sample(replay_memory, minibatch_size)

        #CAUTION: THE PARAMETER DOES A GD-UPDATE ONCE PER TIMESTEP.

        s_t = []
        r_t1 = []
        s_t1 = []

        for i in range(len(minibatch)):
            s_t.append(onehot(minibatch[i][0]))
            r_t1.append(minibatch[i][1])
            s_t1.append(onehot(minibatch[i][2]))

        s_t1 = np.array(s_t1)
        r_t1 = np.array(r_t1)
        s_t = np.array(s_t)

        q_target = target_model.predict(s_t1, verbose = False)


        max_target = np.zeros(minibatch_size)
        y_target = np.zeros(minibatch_size)

        for i in range(minibatch_size):
            max_target[i] = np.argmax(q_target[i]) #Index of maximum Q-value
            y_target[i] = r_t1[i] + discount * np.max(q_target[i]) #Bellman Equation


        estimation = model.predict(s_t, verbose = False) 


        for i in range(minibatch_size):
            ind = int(max_target[i])
            estimation[i][ind] = y_target[i] - estimation[i][ind]

        q_pred = estimation
        
        history = LossHistory()
        model.fit(s_t, q_pred, epochs = 1, verbose = False, callbacks=[history])
        
        loss_history.append(history.losses)
            
        total_steps += 1


    count_nonrandom_action.append(nonrandom_action)
    count_steps.append(step)
    count_episode_reward.append(episode_reward)
    
    
    end_time = time.time()
    episode_time = end_time - start_time
    print(f"No. Non-RandomActions: {nonrandom_action}, Episode Length: {step}")

#Better policy => Slower propagation of active nodes => More diffusion-steps in single episode 
#Greater number of non-random actions is preferable. It means the Q-values are updating. 
#Episode length = No. Immunised Nodes

### If Average Episode Length > Benchmark => better policy
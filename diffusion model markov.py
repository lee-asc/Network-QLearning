

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
def initial_g(G, initial_infected):
    #Initialise all node states to 0 except initial infected
    
    for n in G.nodes():
        G.nodes[n]['state'] = 0

    G.nodes[initial_infected]['state'] = 1       
            
    return G

for (u, v) in G.edges():
    G.edges[u,v]['propagation_probability'] = random.uniform(0.1, 0.5)
    #G.edges[u,v]['propagation_probability'] = 0.25

def diff_step(G): 
    
    active_before = [x for x,y in G.nodes(data=True) if y['state'] == 1]
    
    n_active_before = len(active_before)
    
    for node in active_before:
        for (node, neighbor) in G.edges(node):
            #if (np.random.rand() <= G.edges[node, neighbor]['propagation_probability']) and (G.nodes[neighbor]['state'] != 2):
            if (np.random.rand() <= G.edges[node, neighbor]['propagation_probability']) and (G.nodes[neighbor]['state'] == 0):
                G.nodes[neighbor]['state'] = 1

    active_after = [x for x,y in G.nodes(data=True) if y['state'] == 1]
    
    n_active_after = len(active_after)            
                
    return G, n_active_before, n_active_after

def reward(G, max_steps, n_active_before, n_active_after):
    
    #r = (len(G.nodes()) / max_steps ) + n_active_before - n_active_after
    r = 2 + n_active_before - n_active_after
    
    return r

def observation(G):
    #The state space snapshots of the IMP-environment at each timestep as a list
    #Outputs a vector of node-states at a particular diffusion step

    node_state_dict = [y for x,y in G.nodes(data = True)]
    node_state = [d['state'] for d in node_state_dict]

    return node_state

def onehot(state):
    #Takes input of node-state observation and converts it into a stacked vector of one-hot features
    
    b = np.zeros((state.size, 2 + 1))
    b[np.arange(state.size), state] = 1
    
    one_hot = (b.T).flatten()
    
    return one_hot
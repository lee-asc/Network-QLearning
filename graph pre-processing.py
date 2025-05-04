

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

network = 'facebook_network100.txt'
base_filename = os.path.splitext(os.path.basename(network))[0]
# env = SocialNetwork(network_file)

with open(network) as f:
    G = nx.Graph([line.split()[:2] for line in f])
print(G)
print(nx.is_connected(G))
print(nx.is_directed(G))

node_array = np.array(G.nodes())
pagerank = nx.pagerank(G, alpha = 0.98) 
bet_cent = nx.betweenness_centrality(G)

def v_initialnode(initial_infected):

    print(f'Node {initial_infected}:')
    print(f'Degree = {G.degree(initial_infected)}')
    print(f'PageRank = {pagerank[initial_infected]}')
    print(f'BetweennessCentrality = {bet_cent[initial_infected]}')

    nodeindex = int(np.where(node_array == initial_infected)[0])

    blue = 'b'

    colormap = []
    sizemap = []

    for i in range(len(G.nodes())):
        colormap.append(blue)
        sizemap.append(5)


    colormap[nodeindex] = 'r'  
    sizemap[nodeindex] = 50

    fig = plt.figure(figsize=(12, 4))

    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(Gcc, seed=10396953)
    nx.draw_networkx_nodes(Gcc, pos, node_color=colormap,node_size=sizemap)
    nx.draw_networkx_edges(Gcc, pos, alpha=0.4)




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

def newmodel(learning_rate):
    model = tf.keras.Sequential(
        [
            #tf.keras.layers.Dense(256, input_shape=state.shape, activation="relu"),
            tf.keras.layers.Dense(256, input_shape=onehot(state).shape, activation="relu"),  

            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(len(state), activation="linear") 
        ]
    )
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.SGD(learning_rate),
    )
    return model

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
            
model = newmodel(0.001)
model.summary()
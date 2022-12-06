# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:49:12 2022

@author: shane
"""
import pickle
from MLE_functions import fit

def create_asoiaf(): # Function to load in the ASoIaF Graph
    G = pickle.load(open('ASoIaF', 'rb'))
    return G

G = create_asoiaf()
fit('ASoIaF', G, plot_type='both', save=False) # fit the distribution using MLE
# Plot both PDF and CCDF
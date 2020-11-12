#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 23:07:24 2020

@author: virati
Main class for readout project
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.linear_model import ElasticNet

class readout:
    def __init__(self,H,gamma):
        self.H = H
        self.gamma = gamma
        self.N = H.shape[0]
        
        self.form_graph()
        self.H_noise = 5
        
    def readout_overlap(self):
        return np.dot(np.abs(self.H),np.abs(self.gamma))

    def compound(self):
        return np.dot(self.gamma,self.H.T)
    
    def optimal_r(self):
        return np.sum(self.compound())
    
    def form_graph(self):
        self.G = nx.erdos_renyi_graph(self.N,0.3)
        self.L = nx.laplacian_matrix(self.G).todense()
    
    def sim(self,T=100):
        self.x = np.random.multivariate_normal(np.zeros((self.N,)),self.L,size=(T,)).T
        self.y = np.dot(self.H.T,self.x)
        self.y += np.random.normal(0,self.H_noise,size=self.y.shape)
        self.beta = np.dot(self.gamma.T,self.x)
    
    def disease_salient_x(self):
        return self.gamma != 0
    
    def calc_readout(self):
        ro = ElasticNet()
        ro.fit(self.y,self.beta)
        print(ro.coef_)
    
    def scatter_pva(self):
        plt.figure()
        plt.scatter(self.y,self.beta)

N = 500
gamma = np.random.normal(0,1.0,size=(N,1)) * np.random.choice([0,1,2],p=[450/500,45/500,5/500],size=(N,1))

baseH = np.zeros((N,5))
#H_full_cover= np.copy(baseH)
#H_full_cover[gamma!=0] = 1
H_perfect = np.copy(gamma)

primary = readout(H_perfect,gamma)
primary.sim(T=100)
primary.scatter_pva()
#%%
primary.calc_readout()
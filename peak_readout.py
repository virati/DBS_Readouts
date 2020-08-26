#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:57:22 2020

@author: virati
The Maximum accuracy achievable from a given H on a fixed Gamma
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#We start with a brain network graph
G = nx.erdos_renyi_graph(10,0.5)
L = nx.laplacian_matrix(G).todense()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 17:35:39 2021

@author: virati
Readout Design

Use PySensors to do sparse design of sensor strategies
"""
import numpy as np
import pysensors as pysens
import networkx as nx
from pysensors.classification import SSPOC
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

#dunno this shit
x = np.linspace(0,1,1001)
data = np.vander(x,11).T

#X
N = 20
G = nx.erdos_renyi_graph(N,0.8)
L = nx.linalg.laplacianmatrix.laplacian_matrix(G).todense()
#X = np.random.multivariate_normal(np.zeros((N,)),L,size=(1000,1)).squeeze()

X = np.random.normal(0,1.0,size=(N,100)).T

#f_gamma = np.array([-1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,3.0,0.0]).reshape(-1,1)
f_gamma = np.zeros(shape=(N,1))
#f_gamma = np.random.normal(0,2,size=(N,1))
f_gamma[0:10] = np.random.normal(0,5,size=(10,1))
f_gamma[np.abs(f_gamma) < 2] = 0
Gamma = lambda x: np.dot(x,f_gamma)

beta = Gamma(X).squeeze()
dz = beta > 0
#y = H(x)

X_tr,X_te,dz_tr,dz_te = train_test_split(X,dz)

model = SSPOC()
model.fit(X_tr,dz_tr)

plt.figure()
plt.plot(np.sort(np.abs(model.sensor_coef_)), 'o')
plt.title('Coefficient magnitudes');

model.update_sensors(n_sensors=2, xy=(X_tr, dz_tr))
print('Portion of sensors used:', len(model.selected_sensors) / 10)
print('Selected sensors:', model.selected_sensors)
plt.figure()
plt.plot(f_gamma)
plt.stem(model.selected_sensors,np.ones_like(model.selected_sensors))


accuracy = metrics.accuracy_score(dz_te, model.predict(X_te[:, model.selected_sensors]))
print(accuracy)



#%%
def plot_sensor_locations(sensors, ax=None):
    img = np.zeros(64)
    img[sensors] = 16

    if ax is None:
        plt.imshow(img.reshape(8, 8), cmap=plt.cm.binary)
        plt.xticks([])
        plt.yticks([])
        plt.title('Learned sensor locations')
    else:
        ax.imshow(img.reshape(8, 8), cmap=plt.cm.binary)
        ax.set(xticks=[], yticks=[], title='Learned sensor locations')


#%%
n_sensors_array = np.arange(64)
accuracy = np.zeros(64)

# Suppress warnings arising from no sensors being selected
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)

    for n_sensors in n_sensors_array:
        model.update_sensors(n_sensors=n_sensors, xy = (X_train, y_train), quiet=True)
        if n_sensors == 0:
            accuracy[n_sensors] = metrics.accuracy_score(y_test, model.predict(X_test))
        else:
            accuracy[n_sensors] = metrics.accuracy_score(y_test, model.predict(X_test[:, model.selected_sensors]))

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(n_sensors_array, accuracy, '.')
ax.set(xlabel="Number of sensors", ylabel="Accuracy", title="Accuracy as a function of sensor count");
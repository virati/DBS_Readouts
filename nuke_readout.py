#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 21:09:38 2020

@author: virati
nuclear norm approach

"""
#import numpy as np
import jax
import jax.numpy as jnp

from jax import grad
import numpy as nnp
import itertools

def cost(X,gamma,N=100):
    #return  nnp.sum((X-gamma)**2) #+ 0.1 * jnp.linalg.norm(X,ord='nuc')
    return jnp.linalg.norm(X-gamma,ord=2) + 10 * jnp.linalg.norm(X,ord='nuc')

grad_loss = grad(cost,argnums=0)

b=1
n=100
gamma0 = nnp.random.randint(1,10,size=(b,n)).astype(jnp.float32)

gamma_estimate = nnp.copy(gamma0)
for _ in range(100):
    grads = grad_loss(gamma_estimate,gamma0)
    gamma_estimate -= 0.1*grads

print(gamma_estimate)

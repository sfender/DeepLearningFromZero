#%% 
import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train),(x_test, t_test) = \
    load_mnist(normalize=False, one_hot_label=True)    

def cross_entropy_error(y, t):
    if y.ndim == 1: 
       t = t.reshape(1, t.size)
       y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size

#%%
t = [0,1]
y = [0.1, 0.6]

cross_entropy_error(np.array(y), np.array(t))


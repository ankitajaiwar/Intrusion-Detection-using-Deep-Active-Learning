# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:10:42 2019

@author: ankit
"""


import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import show

import time
start_time = time.time()
print(start_time)
#X_unlabeled = np.load(r"X_unlabeled_final_demo.npy")
#X_unlabeled  = np.array([x.reshape(50,50,3) for x in X_unlabeled])
X_labeled = np.load(r"X_labeled_final_demo.npy")
X_labeled  = np.array([x.reshape(50,50,3) for x in X_labeled])
for i in range(0,5):
    
	plt.imshow(X_labeled[i]);show()
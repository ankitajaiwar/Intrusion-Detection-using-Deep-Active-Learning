# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:38:26 2019

@author: ankit
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 20:40:17 2019

@author: ankit
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 13:12:36 2019

@author: ankit
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 12:58:39 2019

@author: ankit
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:09:09 2019

@author: ankit
"""

from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras import optimizers
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from sklearn.utils import parallel_backend
import keras as k
import random
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import time
start_time = time.time()
print(start_time)
#X_unlabeled = np.load(r".\X_unlabeled_final.npy")
#X_unlabeled  = np.array([x.reshape(50,50,3) for x in X_unlabeled])
X_labeled = np.load(r"X_labeled_final_demo.npy")
X_labeled  = np.array([x.reshape(50,50,3) for x in X_labeled])
#Y_unlabeled = np.load(r".\Y_unlabeled_final.npy")
Y_labeled = np.load(r"Y_labeled_final_demo.npy")
X_test = np.load(r"X_test_final_demo.npy")
X_test  = np.array([x.reshape(50,50,3) for x in X_test])
Y_test = np.load(r"Y_test_final_demo.npy")
#Y_data = np.load(r".\Y_data.npy")

#def create_model():


Y_labeled = k.utils.to_categorical(Y_labeled, 2)
Y_test = k.utils.to_categorical(Y_test, 2)
random.seed(0)
RandomIndices = np.random.permutation(X_labeled.shape[0])
X_labeled = X_labeled[RandomIndices]
X_labeled = X_labeled[:5000]
Y_labeled = Y_labeled[RandomIndices]
Y_labeled = Y_labeled[:5000]


random.seed(0)
RandomIndices = np.random.permutation(X_labeled.shape[0])
X_test = X_test[RandomIndices]
X_test = X_test[:1000]
Y_test = Y_test[RandomIndices]
Y_test = Y_test[:1000]
def create_model(lr):
    img_rows, img_cols = 50, 50
    input_shape = (img_rows, img_cols, 3)
       # opt = optimizers.Adam(lr=0.0001)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    
    optimizer = k.optimizers.Adadelta(lr=lr)
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss=k.losses.binary_crossentropy,optimizer = optimizer,
                  metrics=['accuracy'])
    print(model.summary())
    #    return model
#    model.fit(X_labeled,Y_labeled, batch_size = 128,
#              epochs=30,
#              verbose=2)
    return model
    
model = KerasClassifier(build_fn=create_model, verbose=0)
batch_size = [64, 32]

epochs = [20]
lr = [0.001]
param_grid = dict(batch_size=batch_size, epochs=epochs,  lr = lr)
        
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2, verbose = 10)

    
grid_result = grid.fit(X_labeled,Y_labeled)
print("Done!!")
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print("--- %s seconds ---" % (time.time() - start_time))

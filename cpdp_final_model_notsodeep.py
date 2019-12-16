# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:00:28 2019

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
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import random

import time
start_time = time.time()
print(start_time)
#X_unlabeled = np.load(r"X_unlabeled_final.npy")
#X_unlabeled  = np.array([x.reshape(50,50,3) for x in X_unlabeled])
X_labeled = np.load(r"X_labeled_final_demo.npy")
X_labeled  = np.array([x.reshape(50,50,3) for x in X_labeled])
#Y_unlabeled = np.load(r"Y_unlabeled_final_demo.npy")
Y_labeled = np.load(r"Y_labeled_final_demo.npy")
X_test = np.load(r"X_test_final_demo.npy")
X_test  = np.array([x.reshape(50,50,3) for x in X_test])
Y_test = np.load(r"Y_test_final_demo.npy")
#Y_data = np.load(r"Y_data.npy")

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

        
Y_labeled = k.utils.to_categorical(Y_labeled, 2)
Y_test = k.utils.to_categorical(Y_test, 2)


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

optimizer = k.optimizers.RMSprop(lr = 0.01)
model.add(Dense(2, activation='sigmoid'))
model.compile(loss=k.losses.binary_crossentropy,optimizer = optimizer,
              metrics=['accuracy'])
print(model.summary())
#    return model
history = model.fit(X_labeled,Y_labeled, batch_size = 64,
          epochs=20,
          verbose=2, validation_data = (X_test, Y_test))
    
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

predict = model.predict(X_test)
test_loss, test_accuracy = model.evaluate(X_test, Y_test, batch_size=64)
print('Test loss: %.4f accuracy: %.4f' % (test_loss, test_accuracy))
cf = confusion_matrix(Y_test.argmax(axis=1), predict.argmax(axis=1))
print(cf)


print("--- %s seconds ---" % (time.time() - start_time))

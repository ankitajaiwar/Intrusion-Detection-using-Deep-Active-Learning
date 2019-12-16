# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:55:19 2019

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
import keras as k
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import random


#X_unlabeled = np.load(r"X_unlabeled_final.npy")
#X_unlabeled  = np.array([x.reshape(50,50,3) for x in X_unlabeled])
X_labeled = np.load(r"X_labeled_final_demo.npy")
X_labeled  = np.array([x.reshape(50,50,3) for x in X_labeled])
#Y_unlabeled = np.load(r"Y_unlabeled_final.npy")
Y_labeled = np.load(r"Y_labeled_final_demo.npy")
X_test = np.load(r"X_test_final_demo.npy")
X_test  = np.array([x.reshape(50,50,3) for x in X_test])
Y_test = np.load(r"Y_test_final_demo.npy")
#Y_data = np.load(r"Y_data.npy")
Y_labeled = k.utils.to_categorical(Y_labeled, 2)
Y_test = k.utils.to_categorical(Y_test, 2)
img_rows, img_cols = 50, 50
input_shape = (img_rows, img_cols, 3)

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




base_model = VGG16(weights= 'imagenet', include_top=False, input_shape= input_shape)
for layer in base_model.layers:
    layer.trainable = False
    print(layer.name)
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation= 'relu')(x)
x = Dropout(0.25)(x)
x = Dense(128, activation= 'relu')(x)
x = Dropout(0.25)(x)
predictions = Dense(2, activation= 'sigmoid')(x)


model = Model(inputs = base_model.input, outputs = predictions)
model.compile(loss=k.losses.binary_crossentropy,optimizer=k.optimizers.RMSprop(),
              metrics=['accuracy'])
history = model.fit(X_labeled,Y_labeled, batch_size = 128,
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


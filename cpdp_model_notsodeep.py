import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
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
import numpy
from least_confidence_bias import LeastConfidenceBias


import time
start_time = time.time()
print(start_time)

numpy.random.seed(9)

X_unlabeled = np.load("final_data/X_unlabeled_final.npy")
X_unlabeled  = np.array([x.reshape(50,50,3) for x in X_unlabeled])

X_labeled = np.load("final_data/X_labeled_final.npy")
X_labeled  = np.array([x.reshape(50,50,3) for x in X_labeled])

Y_unlabeled = np.load("final_data/Y_unlabeled_final.npy").flatten()
Y_labeled = np.load("final_data/Y_labeled_final.npy").flatten()

X_test = np.load("final_data/X_test_final.npy")
X_test  = np.array([x.reshape(50,50,3) for x in X_test])

Y_test = np.load("final_data/Y_test_final.npy").flatten()



random_indices = numpy.random.permutation(X_labeled.shape[0])[:5000]
X_labeled = X_labeled[random_indices]
Y_labeled = Y_labeled[random_indices]

good_indices = numpy.argwhere(Y_unlabeled == 1).flatten().tolist()
random_indices = numpy.argwhere(Y_unlabeled == 0).flatten().tolist()
random_indices = numpy.random.permutation(random_indices).tolist()[:50000]
random_indices.extend(good_indices)




X_unlabeled = X_unlabeled[random_indices]
Y_unlabeled = Y_unlabeled[random_indices]

print(X_labeled.shape, " ", Y_labeled.shape)
print(numpy.sum(Y_labeled)/Y_labeled.shape[0])
print(X_unlabeled.shape, " ", Y_unlabeled.shape)

#Y_labeled = k.utils.to_categorical(Y_labeled, 2)
#Y_test = k.utils.to_categorical(Y_test, 2)

img_rows, img_cols = 50, 50
input_shape = (img_rows, img_cols, 3)



model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))

optimizer = k.optimizers.RMSprop()
model.add(Dense(2, activation='sigmoid'))
model.compile(loss=k.losses.binary_crossentropy,optimizer = optimizer,
              metrics=['accuracy'])
print(model.summary())


lcb = LeastConfidenceBias( committee = model, 
						   x_train = X_labeled, 
						   y_train = Y_labeled,
						   x_test = X_test,
						   y_test = Y_test)

f1_scores = lcb.iterative_simulation( x_unlabeled = X_unlabeled, 
						  y_unlabeled = Y_unlabeled,
						  x_test = X_test,
						  y_test = Y_test,
						  k = 300, iterations= 100)

numpy.save('F1Scores_159.npy', f1_scores)
'''
history = model.fit(X_labeled,Y_labeled, batch_size = 128,epochs=10,verbose=2, validation_data = (X_test, Y_test))
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
matrix = confusion_matrix(Y_test.argmax(axis=1), predict.argmax(axis=1))
print("Confusion Matrix: \n", matrix)
'''
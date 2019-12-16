# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:47:21 2019

@author: ankit
"""


import numpy as np
import os
import xml.etree.ElementTree as ET


df = "Demo_data/"



files = os.listdir(df)

errors = []


i = -1
data_array = np.empty((0, 2))
counter = 0
actual = (50**2) * 3
#part of this code taken from https://towardsdatascience.com/building-an-intrusion-detection-system-using-deep-learning-b9488332b321
for file in files:
    print(file)
    try:
        tree = ET.parse(df + file)
        print('Reading File ', file)
        root = tree.getroot()
    except:
        errors += file
        continue
    for child in root:
        for next_child in child:
            if next_child.tag == 'destinationPayloadAsUTF':
                if next_child.text is not None:
                    x = next_child.text
                    if len(x) > actual:
                        x = x[: actual]
                    else:
                        while len(x) < actual:
                            x += x
                        x = x[:actual]
                    if child.find('Tag').text == 'Normal':
                        data_array = np.vstack((data_array, np.array([np.fromstring(x, dtype=np.uint8), 0])))
                    else:
                        data_array = np.vstack((data_array, np.array([np.fromstring(x, dtype=np.uint8), 1])))
                    counter += 1



    
img_row = 50
img_col = 50

N = np.shape(data_array)[0]
train_test_split_percentage = 0.80

X_train_1 = data_array[:int(N * train_test_split_percentage), 0]
X_test_1 = data_array[int(N * train_test_split_percentage):, 0]

X_train_1 = np.array([x.reshape(7500,) for x in X_train_1])
X_test_1 = np.array([x.reshape(7500,) for x in X_test_1])


y_train_1 = data_array[:int(N * train_test_split_percentage), 1]
y_test_1 = data_array[int(N * train_test_split_percentage):, 1]

y_train_1 = np.array([[x] for x in y_train_1])
y_test_1 = np.array([[x] for x in y_test_1])


num_unlabeled = int(X_train_1.shape[0]/2)


X_unlabeled_1 = X_train_1[0:num_unlabeled]
X_labeled_1 = X_train_1[num_unlabeled:]

Y_unlabeled_1 = y_train_1[0:num_unlabeled]
Y_labeled_1 = y_train_1[num_unlabeled:]

np.save("X_test_final_demo",X_test_1)
np.save("Y_test_final_demo",y_test_1)
np.save("X_train_final_demo",X_train_1)
np.save("Y_train_final_demo",y_train_1)
np.save("X_unlabeled_final_demo",X_unlabeled_1)
np.save("Y_unlabeled_final_demo",Y_unlabeled_1)
np.save("X_labeled_final_demo",X_labeled_1)
np.save("Y_labeled_final_demo",Y_labeled_1)

x_data = data_array[:, 0]
x = np.array([x.reshape(7500,) for x in x_data])
np.save("X_data_demo",x)
y_data = data_array[:, 1]
y = np.array([[x] for x in y_data])
np.save("Y_data_demo",y)


# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 09:18:38 2020

@author: pete_
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans

from skopt import BayesSearchCV

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve


from sklearn.datasets import load_diabetes
from sklearn.datasets import make_regression
from sklearn.datasets import make_circles
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_classification
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml
from sklearn.datasets import make_blobs


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix 


from sklearn import tree

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Model


#%%

#Import data

train = pd.read_csv(r"C:\Users\pete_\Documents\Coding\git\Fashion\train.csv")
test = pd.read_csv(r"C:\Users\pete_\Documents\Coding\git\Fashion\test.csv")

X_train = train.drop('label',axis=1).values
y_train = train['label'].values
X_test = test.drop('label',axis=1).values
y_test = test['label'].values

print("Size of data \n")
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print("45.000 vs. 7.500 examples of 28x28 pixel greyscale")

labels = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']

#Normalize data
X_train, X_test = X_train/255, X_test/255

#Data is perfectly balanced

#%%

#Visualize

#Visualize the pictures in greyscale using matplotlib


fig, axes = plt.subplots(4, 4)
for i in range(4):
    for j in range (4):
        axes[i][j].matshow(X_train[4*i+j].reshape(28, 28), cmap=plt.cm.gray)
        axes[i][j].xaxis.set_visible(False)
        axes[i][j].yaxis.set_visible(False)
plt.show()


#%%

#Building a NN model

#NORMALISATION HELPED A LOT

#Decrease the regularization parameter alpha
mlp = MLPClassifier(hidden_layer_sizes=(80,60),max_iter =500, alpha=1e-4,solver='sgd',random_state=1)
mlp.fit(X_train,y_train)
print('Accuracy on test set: ',mlp.score(X_test,y_test))

#Human level is 83.5% <- benchmark
#Best: 88.95%

#%%

#Which did we get wrong?
y_pred = mlp.predict(X_test)
wrong_X = X_test[y_pred != y_test]
wrong_true = y_test[y_pred != y_test]
wrong_pred = y_pred[y_pred != y_test]

#let's look at a value
j = 5
plt.matshow(wrong_X[j].reshape(28, 28), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.show()
print("true value:", labels[wrong_true[j]])
print("predicted value:", labels[wrong_pred[j]])



#%%


#Try with CNN!

train_images = X_train.reshape(-1,28,28,1)
train_labels = y_train.reshape(-1,1)
test_images = X_test.reshape(-1,28,28,1)
test_labels = y_test.reshape(-1,1)


#%%

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=6, 
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)


#91.02% instead of 88.95% !!!

#%%


# visualize feature maps output from each block

# redefine model to output right after conv layers
ixs = [0,2,4]
outputs = [model.layers[i].output for i in ixs]
mod = Model(inputs=model.inputs, outputs=outputs)
# load the image with the required shape
img = train_images[0].reshape(1,28,28,1)

#Plot image
plt.figure()
plt.imshow(img[0,:,:,0], cmap='gray')
plt.show

# get feature map for first hidden layer
feature_maps = mod.predict(img)
# plot the output from each block
square = 8
for fmap in feature_maps: 
	# plot first 64 maps in an 8x8 squares
    ix = 1
    plt.figure()
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
			# plot filter channel in grayscale
            plt.imshow(fmap[0, :, :, ix-1], cmap='gray')
            ix += 1
	# show the figure
    plt.show()
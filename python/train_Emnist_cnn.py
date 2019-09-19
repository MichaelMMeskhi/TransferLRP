'''
@author: Sebastian Lapuschkin
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 30.11.2016
@version: 1.2+
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import modules
import model_io
import data_io

import importlib.util as imp
import numpy
import numpy as np
if imp.find_spec("cupy"): #use cupy for GPU support if available
    import cupy
    import cupy as np
na = np.newaxis

# base_nn = model_io.read('../models/LeNet-5-maxpooling.txt')


#load the mnist data
Xtrain = np.load('../data/EMNIST/Emnist_train_images.npy')
Ytrain = np.load('../data/EMNIST/Emnist_train_labels.npy')

Xtrain = np.array([x.T for x in Xtrain])

Xtest = np.load('../data/EMNIST/Emnist_test_images.npy')
Ytest = np.load('../data/EMNIST/Emnist_test_labels.npy')

Xtest = np.array([x.T for x in Xtest])

print("Size of training data set ", len(Ytrain))
print("Size of testing data set", len(Ytest))


# transfer pixel values from [0 255] to [-1 1] to satisfy the expected input / training paradigm of the model
Xtrain =  Xtrain / 127.5 - 1
Xtest =  Xtest / 127.5 - 1

#reshape the vector representations of the mnist data back to image format. extend the image vertically and horizontally by 4 pixels each.
Xtrain = np.reshape(Xtrain,[Xtrain.shape[0],28,28,1])
Xtrain = np.pad(Xtrain,((0,0),(2,2),(2,2),(0,0)), 'constant', constant_values = (-1.,))

Xtest = np.reshape(Xtest,[Xtest.shape[0],28,28,1])
Xtest = np.pad(Xtest,((0,0),(2,2),(2,2),(0,0)), 'constant', constant_values = (-1.,))

# transform numeric class labels to vector indicator for uniformity. assume presence of all classes within the label set
I = Ytrain.astype(int) - 1
Ytrain = np.zeros([Xtrain.shape[0],np.unique(Ytrain).size])
Ytrain[np.arange(Ytrain.shape[0]),I] = 1

I = Ytest.astype(int) - 1
Ytest = np.zeros([Xtest.shape[0],np.unique(Ytest).size])
Ytest[np.arange(Ytest.shape[0]),I] = 1

if False:
    #model a network according to LeNet-5 architecture
    lenet = modules.Sequential([
                                modules.Convolution(filtersize=(5,5,1,10),stride = (1,1)),\
                                modules.Rect(),\
                                modules.SumPool(pool=(2,2),stride=(2,2)),\
                                modules.Convolution(filtersize=(5,5,10,25),stride = (1,1)),\
                                modules.Rect(),\
                                modules.SumPool(pool=(2,2),stride=(2,2)),\
                                modules.Convolution(filtersize=(4,4,25,100),stride = (1,1)),\
                                modules.Rect(),\
                                modules.SumPool(pool=(2,2),stride=(2,2)),\
                                modules.Convolution(filtersize=(1,1,100,10),stride = (1,1)),\
                                modules.Flatten()
                            ])

    #train the network.
    lenet.train(   X=Xtrain,\
                    Y=Ytrain,\
                    Xval=Xtest,\
                    Yval=Ytest,\
                    iters=10**5,\
                    lrate=0.001,\
                    batchsize=128)

    #save the network
    model_io.write(lenet, '../LeNet-5.txt')



if True:
    #a slight variation to test max pooling layers. this model should train faster.
    maxnet = modules.Sequential([
                                modules.Convolution(filtersize=(5,5,1,10),stride = (1,1)),\
                                modules.Rect(),\
                                modules.MaxPool(pool=(2,2),stride=(2,2)),\
                                modules.Convolution(filtersize=(5,5,10,25),stride = (1,1)),\
                                modules.Rect(),\
                                modules.MaxPool(pool=(2,2),stride=(2,2)),\
                                modules.Convolution(filtersize=(4,4,25,100),stride = (1,1)),\
                                modules.Rect(),\
                                modules.MaxPool(pool=(2,2),stride=(2,2)),\
                                modules.Convolution(filtersize=(1,1,100,26),stride = (1,1)),\
                                modules.Flatten(),\
                                modules.SoftMax()
                            ])




    #train the network.
    maxnet.train(   X=Xtrain,\
                    Y=Ytrain,\
                    Xval=Xtest,\
                    Yval=Ytest,\
                    iters=25000,\
                    lrate=0.001,\
                    batchsize=128)

    #save the network
    model_io.write(maxnet, '../Emnist_cnn.txt')
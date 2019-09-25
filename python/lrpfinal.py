import matplotlib.pyplot as plt
import time
import numpy
import numpy as np
import importlib.util as imp
if imp.find_spec("cupy"): #use cupy for GPU support if available
    import cupy
    import cupy as np
na = np.newaxis

import model_io
import data_io
import render
# import cv2


def convertPixels_to_XY(pixels, size, dim=2, scale=3):
    cord = []
    for pixel in pixels:
        y = pixel%size
        x = int(pixel/size)
        for i in range(scale):
            for j in range(scale):
                cord.append((scale*x+i, scale*y+j))
    return cord


def run_demo(x, model, lrp_type, r_approach, reset_threshold, overlap_threshold):
    x = x / 127.5 - 1
    

    if model == 'EMNIST_cnn':
        x = x.reshape((28, 28, 1))
        x = np.array([x])
        x = np.pad(x, ((0,0), (2,2), (2,2), (0,0)), 'constant', constant_values=(-1.,))
        print("shape", x.shape)
        nn = model_io.read('/Users/michaelmmeskhi/Desktop/TEX/models/Mnist_EMnist_cnn', fmt="pickled")
        nn.modules[6].name = "lbf"
        # X = data_io.read('../data/EMNIST/Emnist_test_images.npy')
        # Y = data_io.read('../data/EMNIST/Emnist_test_labels.npy')
        # X = np.array([x.T for x in X])
        # X =  X / 127.5 - 1
        # #reshape the vector representations in X to match the requirements of the CNN input
        # X = np.reshape(X,[X.shape[0],28,28,1])
        # X = np.pad(X,((0,0),(2,2),(2,2),(0,0)), 'constant', constant_values = (-1.,))
        # I = Y.astype(int) - 1
        # Y = np.zeros([X.shape[0],np.unique(Y).size])
        # Y[np.arange(Y.shape[0]),I] = 1
    elif model == 'EMNIST_nn':
        x = x.reshape((1, 28, 28))
        nn = model_io.read('/Users/michaelmmeskhi/Desktop/TEX/models/Mnist_Emnist_nn', fmt="pickled")
        nn.modules[7].name = "lbf"
        # X = data_io.read('../data/EMNIST/Emnist_test_images.npy')
        # Y = data_io.read('../data/EMNIST/Emnist_test_labels.npy')
        # X = np.array([x.T for x in X])
        # X =  X / 127.5 - 1
        # I = Y.astype(int) - 1
        # Y = np.zeros([X.shape[0],np.unique(Y).size])
        # Y[np.arange(Y.shape[0]),I] = 1
    elif model=="MNIST_cnn":
        x = x.reshape((28, 28, 1))
        x = np.array([x])
        x = np.pad(x, ((0,0), (2,2), (2,2), (0,0)), 'constant', constant_values=(-1.,))
        nn = model_io.read('/Users/michaelmmeskhi/Desktop/TEX/models/Emnist_Mnist_cnn', fmt="pickled")
        nn.modules[6].name = "lbf"
        # X = data_io.read('../data/MNIST/test_images.npy')
        # Y = data_io.read('../data/MNIST/test_labels.npy')
        # X =  X / 127.5 - 1
        # X = np.reshape(X,[X.shape[0],28,28,1])
        # X = np.pad(X,((0,0),(2,2),(2,2),(0,0)), 'constant', constant_values = (-1.,))
        # I = Y[:,0].astype(int)
        # Y = np.zeros([X.shape[0],np.unique(Y).size])
        # Y[np.arange(Y.shape[0]),I] = 1
    elif model=="MNIST_nn":
        x = x.reshape((1, 28, 28))
        nn = model_io.read('/Users/michaelmmeskhi/Desktop/TEX/models/Emnist_Mnist_nn', fmt="pickled")
        nn.modules[7].name = "lbf"
        # X = data_io.read('../data/MNIST/test_images.npy')
        # Y = data_io.read('../data/MNIST/test_labels.npy')
        # X =  X / 127.5 - 1
        # I = Y[:,0].astype(int)
        # Y = np.zeros([X.shape[0],np.unique(Y).size])
        # Y[np.arange(Y.shape[0]),I] = 1

    nn.drop_softmax_output_layer()	
    ypred = nn.forward(x)
    mask = np.zeros_like(ypred)
    mask[:,np.argmax(ypred)] = 1
    Rinit = ypred*mask
    activations = nn.forward(x, lbfBreak=True)
    activations = activations[0]
    if lrp_type == "alphabeta":
        epsilon = 4
    elif lrp_type == "epsilon":
        epsilon = 0.01
    else:
        epsilon = 2
    if model == "MNIST_cnn" or model=="EMNIST_cnn":
        network = "cnn"
    else:
        network = "nn"
    
    R = nn.lrp(Rinit, lrp_type, epsilon, r_approach, t_A=reset_threshold, t_R=reset_threshold, act=activations, net=network)
    R_normal = nn.lrp(Rinit, lrp_type, epsilon, 0)

    hm_normal = render.hm_to_rgb_2(R_normal, X = x, scaling = 3, sigma = 2)

    if model == "MNIST_cnn" or model=="EMNIST_cnn":
        R = R.sum(axis=3)
        R = R[0].reshape((1,1024))[0]
        R_normal = R_normal.sum(axis=3)
        R_normal = R_normal[0].reshape((1,1024))[0]
    else:
        R = R[0].reshape((1, 784))[0]
        R_normal = R_normal[0].reshape((1, 784))[0]

    if model == "EMNIST_cnn" or model =="MNIST_cnn":
        target = int(1024 - 10.24*overlap_threshold)
    else:
        target = int(784 - 7.84*overlap_threshold)

    threshold1 = np.partition(R_normal, target)[target]
    threshold2 = np.partition(R, target)[target]

    indices1 = np.where(R_normal > threshold1)

    indices2 = np.where(R > threshold2)

    if model == "EMNIST_cnn" or model=="MNIST_cnn":
        size = 32
    else:
        size = 28
    common_red = np.intersect1d(indices1, indices2)

    common_red_cord = convertPixels_to_XY(common_red, size)

    indices1_cord = convertPixels_to_XY(list(indices1[0]), size)

    if model == "MNIST_cnn" or model == "EMNIST_cnn":
        R_overlap = np.zeros(1024)
    else:
        R_overlap = np.zeros(784)

    R_overlap[common_red] = 10
    image1 = hm_normal.copy()
    image2 = hm_normal.copy()

    for row in range(len(image1)):
        for col in range(len(image1[row])):
            if image1[row][col][0] != 0 or image1[row][col][1] != 0 or image1[row][col][2] != 0:
                image1[row][col][0] = image1[row][col][1] = image1[row][col][2] = 1.

    for index in indices1_cord:
        image1[index[0]][index[1]][0] = 0.
        image1[index[0]][index[1]][1] = 1.
        image1[index[0]][index[1]][2] = 0.

        image2[index[0]][index[1]][0] = 0.
        image2[index[0]][index[1]][1] = 0.
        image2[index[0]][index[1]][2] = 0.

    for index in common_red_cord:
        image1[index[0]][index[1]][0] = 1.
        image1[index[0]][index[1]][1] = 0.
        image1[index[0]][index[1]][2] = 0.

    render.save_image([image1,image2], '../canvas/images/lrpresult.png')
    # render.save_image(image2,'../colorful.png')
    return ypred[0]
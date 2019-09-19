#!/usr/bin/env python3

'''
@author: Sebastian Lapuschkin
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 14.08.2015
@version: 1.0
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause

The purpose of this module is to demonstrate the process of obtaining pixel-wise explanations for given data points at hand of the MNIST hand written digit data set.

The module first loads a pre-trained neural network model and the MNIST test set with labels and transforms the data such that each pixel value is within the range of [-1 1].
The data is then randomly permuted and for the first 10 samples due to the permuted order, a prediction is computed by the network, which is then as a next step explained
by attributing relevance values to each of the input pixels.

finally, the resulting heatmap is rendered as an image and (over)written out to disk and displayed.
'''


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

def runlrp(canvas_input):
    #load a neural network, as well as the MNIST test data and some labels
    nn_target = model_io.read('../Mnist_to_letters_2.0', fmt='pickled') # 99.16% prediction accuracy
    nn_target.drop_softmax_output_layer() #drop softnax output layer for analyses

    #predict and perform LRP for the 10 first samples
    x = np.array([canvas_input])

    #forward pass and prediction
    ypred = nn_target.forward(x)

    #prepare initial relevance to reflect the model's dominant prediction (ie depopulate non-dominant output neurons)
    mask = np.zeros_like(ypred)
    mask[:,np.argmax(ypred)] = 1
    Rinit = ypred*mask

    #compute first layer relevance according to prediction
    #R = nn.lrp(Rinit)                   #as Eq(56) from DOI: 10.1371/journal.pone.0130140
    R = nn_target.lrp(Rinit,'simple',0.01)    #as Eq(58) from DOI: 10.1371/journal.pone.0130140
    #R = nn.lrp(Rinit,'alphabeta',2)    #as Eq(60) from DOI: 10.1371/journal.pone.0130140

    #undo input normalization for digit drawing. get it back to range [0,1] per pixel
    x = (x+1.)/2.

    if not np == numpy: # np=cupy
        x = np.asnumpy(x)
        R = np.asnumpy(R)

    #render input and heatmap as rgb images
    digit = render.digit_to_rgb(x, scaling = 3)
    hm = render.hm_to_rgb(R, X = x, scaling = 3, sigma = 2)
    digit_hm = render.save_image([digit,hm],'../canvas/images/lrpresult'+'.png')
    # data_io.write(R,'../Compare/heatmap_target'+str(i)+'.npy')

    return ypred[0]



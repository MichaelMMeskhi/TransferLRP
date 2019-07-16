from __future__ import absolute_import, division, print_function

import numpy
from lrp import nn_lrp
from lrp import cnn_lrp
from utils import utils
from architecture import architecture





if __name__ == '__main__':
    """
    Neural Network Layer-wise Relevance Propagation
    -----------------------------------------------
    * No. of Layers: 6 
    * No. of Hidden Layers: 2
    * No. of Samples (N): 1
    """
    # X, T = utils.getMNISTsample(N = 1, path = './data/', seed = 99)
    # utils.visualize(X, utils.graymap, './results/data.png')
    
    # nn = nn_lrp.Network([
    #     nn_lrp.FirstLinear('./parameters/nn/l1'),nn_lrp.ReLU(),
    #     nn_lrp.NextLinear('./parameters/nn/l2'),nn_lrp.ReLU(),
    #     nn_lrp.NextLinear('./parameters/nn/l3'),nn_lrp.ReLU(),
    #     ])

    # print("Neural Network initialized with size:%d\n" % len(nn.layers))
    
    # Y = nn.forward(X)
    # utils.probability(Y)
    # D = nn.relprop(Y*T)
    # utils.visualize(D, utils.heatmap, './results/mlp-deeptaylor.png')
    # print("\nExperiment complete!")

    """
    Convolutional Neural Network Layer-wise Relevance Propagation
    -------------------------------------------------------------
    * No. of Layers: 11
    * No. of Hidden Layers: 5
    * No. of Samples (N): 1
    """
    X,T = utils.getMNISTsample(N = 12, path = './data', seed = 99)
    
    padding = ((0,0),(2,2),(2,2),(0,0))
    X = numpy.pad(X.reshape([12,28,28,1]),padding,'constant',constant_values=(utils.lowest,))
    T = T.reshape([12,1,1,10])
    
    cnn = cnn_lrp.Network([
            cnn_lrp.FirstConvolution('./parameters/cnn/c1-5x5x1x10'),cnn_lrp.ReLU(),cnn_lrp.Pooling(),
            cnn_lrp.NextConvolutionAlphaBeta('./parameters/cnn/c2-5x5x10x25', 2.0),cnn_lrp.ReLU(),cnn_lrp.Pooling(),
            cnn_lrp.NextConvolutionAlphaBeta('./parameters/cnn/c3-4x4x25x100', 2.0),cnn_lrp.ReLU(),cnn_lrp.Pooling(),
            cnn_lrp.NextConvolutionAlphaBeta('./parameters/cnn/c4-1x1x100x10', 2.0),cnn_lrp.ReLU(),
        ])

    Y = cnn.forward(X)
    D = cnn.relprop(Y*T)
    utils.visualize(D[:,2:-2,2:-2],utils.heatmap,'./results/cnn-deeptaylor.png')
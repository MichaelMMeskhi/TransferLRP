from __future__ import absolute_import, division, print_function

from utils import utils
from architecture import architecture
from lrp import lrp

"""
Neural Network Layer-wise Relevance Propagation
-----------------------------------------------
* No. of Layers: 3 [FirstLinear + ReLU, NextLinear + ReLU, NextLinear + ReLU]
* No. of Hidden Layers: 2 [NextLinear + ReLU, NextLinear + ReLU]
* No. of Samples (N): 5 
"""


if __name__ == '__main__':
    X, T = utils.getMNISTsample(N = 5, path = './data/', seed = 99)
    utils.visualize(X, utils.graymap, './results/data.png')
    
    nn = lrp.Network([
        lrp.FirstLinear('./parameters/l1'),lrp.ReLU(),
        lrp.NextLinear('./parameters/l2'), lrp.ReLU(),
        lrp.NextLinear('./parameters/l3'), lrp.ReLU(),])

    print("Neural Network initialized with size:", len(nn.layers))

    Y = nn.forward(X)
    utils.probability(Y)
    D = nn.relprop(Y*T)
    utils.visualize(D, utils.heatmap, './results/mlp-deeptaylor.png')
    print("\nExperiment complete!")
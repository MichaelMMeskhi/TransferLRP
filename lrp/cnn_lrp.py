import math
import copy
import numpy
from utils import utils
from architecture import architecture

""" 
Extends the architecture.py module to introduce layer-wise 
relevance propagation for convolutional neural networks. 
Computes the Deep Taylor Decomposition using the relprop() 
function extenstion for each class (relevance propagation). 
"""

class Network(architecture.Network):
    """
    Extension to the Network class from architecture.py.
    Takes in the whole network as input and starts the 
    relevance propagation from output layer to input layer.

    Args:
        architecture.Network: Base class Network object. 

    Returns:
        R: Outputs final R value of all layers (relevance).
    """
    def relprop(self,R):
        for i, l in enumerate(self.layers[::-1]): R = l.relprop(R)
        return R

class ReLU(architecture.ReLU):
    """
    Extension to the ReLU class from architecture.py.
    Takes in the ReLU layer as input. LRP is not computed and instead
    directly passed into previous layer as back-propagated input.

    Args:
        architecture.Relu: Base class ReLU layer. 

    Returns:
        R: Output of that ReLU layer (relevance).
    """
    def relprop(self,R): return R

class NextConvolution(architecture.Convolution):
    """
    Extension to the Convolution class from architecture.py.
    Takes in the Convolution layer as input. LRP is computed and
    directly passed into previous layer as back-propagated input.
    Relevance is computed using the z+ rule (See below). 

    Args:
        architecture.Convolution: Base class Convolution layer. 

    Returns:
        R: Output of that Convolution layer (relevance).
    """
    def relprop(self,R):
        pself = copy.deepcopy(self); pself.B *= 0;
        pself.W = numpy.maximum(0,pself.W)

        Z = pself.forward(self.X)+1e-9;
        S = R/Z
        C = pself.gradprop(S); 
        utils.noderel(C,i,net='cnn') # Finds least relevant nodes in a layer
        R = self.X*C
        return R


class FirstConvolution(architecture.Convolution):
    """
    Extension to the FirstConvolution class from architecture.py.
    Takes in the FirstConvolution layer as input. Final LRP is computed.
    Relevance is computed using the zB rule (See below). 

    Args:
        architecture.FirstConvolution: Base class FirstConvolution layer. 

    Returns:
        R: Output of that FirstConvolution layer (relevance).
    """
    def relprop(self,R):
        iself = copy.deepcopy(self); iself.B *= 0
        nself = copy.deepcopy(self); nself.B *= 0; nself.W = numpy.minimum(0,nself.W)
        pself = copy.deepcopy(self); pself.B *= 0; pself.W = numpy.maximum(0,pself.W)
        X,L,H = self.X,self.X*0+utils.lowest,self.X*0+utils.highest

        Z = iself.forward(X)-pself.forward(L)-nself.forward(H)+1e-9; S = R/Z
        R = X*iself.gradprop(S)-L*pself.gradprop(S)-H*nself.gradprop(S)
        return R


class Pooling(architecture.Pooling):
    """
    Extension to the Pooling class from architecture.py.
    Takes in the Pooling layer as input. Proportional 
    redistribution occuring in the sum-pooling layer. 

    Args:
        architecture.Pooling: Base class Pooling layer. 

    Returns:
        R: Output of that Pooling layer (relevance).
    """
    def relprop(self,R):
        Z = (self.forward(self.X)+1e-9);
        S = R / Z
        C = self.gradprop(S)  
        R = self.X*C
        return R        

class NextConvolutionAlphaBeta(architecture.Convolution,object):
    """
    Computes the AlphaBeta Relevance score (negative relvance).

    Args:
        architecture.Convolution: Base class Convolution layer.
        object: Aplha value (default=2.0). 

    Returns:
        R: Output of that Convolution layer (negative relevance).
    """
    def __init__(self,name,alpha):
        super(self.__class__, self).__init__(name)
        self.alpha = alpha
        self.beta  = alpha-1
        
    def relprop(self,R):
        pself = copy.deepcopy(self); pself.B *= 0; pself.W = numpy.maximum( 1e-9,pself.W)
        nself = copy.deepcopy(self); nself.B *= 0; nself.W = numpy.minimum(-1e-9,nself.W)

        X = self.X+1e-9
        ZA = pself.forward(X); SA =  self.alpha*R/ZA
        ZB = nself.forward(X); SB = -self.beta *R/ZB
        R = X*(pself.gradprop(SA)+nself.gradprop(SB))
        return R
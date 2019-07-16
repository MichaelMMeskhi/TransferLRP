import numpy
from utils import utils
from architecture import architecture

""" 
Extends the architecture.py module to introduce layer-wise 
relevance propagation for neural networks. Computes the Deep Taylor Decomposition
using the relprop() function extenstion for each class (relevance propagation). 
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
        for i, l in enumerate(self.layers[::-1]): R = l.relprop(R, i)
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
    def relprop(self,R,i): return R
    
class NextLinear(architecture.Linear):
    """
    Extension to the Linear class from architecture.py.
    Takes in the Linear layer as input. LRP is computed and
    directly passed into previous layer as back-propagated input.
    Relevance is computed using the z+ rule (See below). 

    Args:
        architecture.Linear: Base class Linear layer. 

    Returns:
        R: Output of that Linear layer (relevance).
    """
    def relprop(self,R,i):
        """
        Relevance is computed at R = self.X*C 
        self.X is the number of neurons in a given layer at iteration.
        Ci is the constant value for each Xi, where Xi is our i-th neuron.
        We compute a threshold for C and determine every Xi over that threshold
        is more relevant to a given task. 
        """
        V = numpy.maximum(0, self.W)
        Z = numpy.dot(self.X, V) + 1e-9
        S = R/Z
        C = numpy.dot(S, V.T)
        Ci = utils.noderel(C,i, net='nn') # Finds least relevant nodes in a layer
        R = self.X*C
        return R

class FirstLinear(architecture.Linear):
    """
    Extension to the FirstLinear class from architecture.py.
    Takes in the FirstLinear layer as input. Final LRP is computed.
    Relevance is computed using the zB rule (See below). 

    Args:
        architecture.FirstLinear: Base class FirstLinear layer. 

    Returns:
        R: Output of that FirstLinear layer (relevance).
    """
    def relprop(self,R,i):
        W,V,U = self.W,numpy.maximum(0,self.W),numpy.minimum(0,self.W)
        X,L,H = self.X,self.X*0+utils.lowest,self.X*0+utils.highest

        Z = numpy.dot(X,W)-numpy.dot(L,V)-numpy.dot(H,U)+1e-9; S = R/Z
        R = X*numpy.dot(S,W.T)-L*numpy.dot(S,V.T)-H*numpy.dot(S,U.T)
        return R
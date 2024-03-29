'''
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 14.08.2015
@version: 1.2+
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

# -------------------------------
# Modules for the neural network
# -------------------------------
import numpy as np
class Module:
    '''
    Superclass for all computation layer implementations
    '''
    def __init__(self):
        ''' The constructor '''

        #values for presetting lrp decomposition parameters per layer
        self.lrp_var = None
        self.lrp_param = 1.
        count = 0

    def backward(self,DY):
        ''' backward passes the error gradient DY to the input neurons '''
        return DY

    def train(self, X, Y, *args, **kwargs):
        ''' implements (currently in modules.Sequential only) a simple training routine '''

    def forward(self,X,lrp_aware=False):
        ''' forward passes the input data X to the layer's output neurons.

        Parameters
        ----------

        X : numpy.ndarray
            the input activations for this layer, shaped [batchsize, ...]

        lrp_aware : bool
            controls whether the forward pass is to be computed with awareness for multiple following
            LRP calls. this will sacrifice speed in the forward pass but will save time if multiple LRP
            calls will follow for the current X, e.g. wit different parameter settings or for multiple
            target classes.

        '''
        return X

    def update(self, lrate):
        ''' update should implement the layer parameter updating step '''
        pass

    def clean(self):
        ''' clean can be used to remove any temporary variables from the layer, e.g. just before serializing the layer object'''
        pass



    def set_lrp_parameters(self,lrp_var=None,param=None):
        ''' pre-sets lrp parameters to use for this layer. see the documentation of Module.lrp for details '''
        self.lrp_var = lrp_var
        self.lrp_param = param

    def lrp(self, R, lrp_var=None,param=None, reset=0, t_A=15, t_R=15, net="nn", act=[0]):

        '''
        Performs LRP by calling subroutines, depending on lrp_var and param or
        preset values specified via Module.set_lrp_parameters(lrp_var,lrp_param)

        If lrp parameters have been pre-specified (per layer), the corresponding decomposition
        will be applied during a call of lrp().

        Specifying lrp parameters explicitly when calling lrp(), e.g. net.lrp(R,lrp_var='alpha',param=2.),
        will override the preset values for the current call.

        How to use:

        net.forward(X) #forward feed some data you wish to explain to populat the net.

        then either:

        net.lrp() #to perform the naive approach to lrp implemented in _simple_lrp for each layer

        or:

        for m in net.modules:
            m.set_lrp_parameters(...)
        net.lrp() #to preset a lrp configuration to each layer in the net

        or:

        net.lrp(somevariantname,someparameter) # to explicitly call the specified parametrization for all layers (where applicable) and override any preset configurations.

        Parameters
        ----------

        R : numpy.ndarray
            relevance input for LRP.
            should be of the same shape as the previously produced output by <Module>.forward

        lrp_var : str
            either 'none' or 'simple' or None for standard LRP ,
            'slow' or 'simple_slow' for the explicit implementation of LRP,
            'epsilon' for an added epsilon slack in the denominator,
            'epsilon_slow' for the explicit implementation of the epsilon stabilized variant
            'alphabeta' or 'alpha' for weighting positive and negative contributions separately. param specifies alpha with alpha + beta = 1
            'flat' projects an upper layer neuron's relevance uniformly over its receptive field.
            'ww' or 'w^2' only considers the square weights w_ij^2 as qantities to distribute relevances with.

        param : double
            the respective parameter for the lrp method of choice

        Returns
        -------
        R : the backward-propagated relevance scores.
            shaped identically to the previously processed inputs in <Module>.forward
        '''
        # print(self.W[0][0])
        # if net == 'full':
        #     wt = -0.00017673444062546892
        # else:
        #     wt = 0.027585652502580577
        # print(self.name)
        try: 
            if(reset != 0 and self.name == 'lbf'):
                # print("Here is, lrp")
                # print("R", R[0].shape)

                if net == "cnn":
                    # print(R[0])
                    shape = R[0].shape
                    length = 1
                    for s in shape:
                        length = length * s
                    newR = R[0].reshape((1, length))[0]
                    newA = act.reshape((1,length))[0]
                    target_R = int(len(newR) - len(newR)*t_R/100) #used length of W to get number of neurons
                    target_A = int(len(newA) - len(newA)*t_A/100)
                    threshold_R = np.partition(newR, target_R)[target_R]
                    threshold_A = np.partition(newA, target_A)[target_A]
                    top_RN = np.where(newR >= threshold_R)
                    top_AN = np.where(newA >= threshold_A)
                    S = np.intersect1d(top_RN, top_AN)
                    sum_R = sum(newR)
                    if reset == 1:
                        new_R = [0] * len(newR)
                        sum_S = sum(newR[S])
                        for index in S:
                            # print("here")
                            # print((R[0][index]/sum_S) * sum_R)
                            new_R[index] = (newR[index]/sum_S) * sum_R
                            # print(index, new_R[index])
                        # print(new_R)
                        new_R = np.array(new_R)
                        R[0] = new_R.reshape(shape)
                    elif reset == 2:
                        new_R = [0] * len(newR)
                        for index in S:
                            new_R[index] = sum_R/len(S)
                        new_R = np.array(new_R)
                        R[0] = new_R.reshape(shape)
                    elif reset == 3:
                        new_R = [0] * len(newR)
                        for index in S:
                            new_R[index] = newR[index] 
                        new_R = np.array(new_R)
                        R[0] = new_R.reshape(shape)
                    else:
                        print("Invalid argument, reset can only have values as 0, 1 or 2")
                    # print("act---", act)
                    # print("R[0]------", R[0])


                    

                else:
                    # threshold_R = max(R[0]) - ( (max(R[0])-min(R[0])) * tp_R/100 )
                    # threshold_A = max(newW) - ( (max(newW)-min(newW)) * tp_A/100 )
                    target_R = int(len(self.W) - len(self.W)*t_R/100) #used length of W to get number of neurons
                    target_A = int(len(self.W) - len(self.W)*t_A/100)
                    threshold_R = np.partition(R[0], target_R)[target_R]
                    threshold_A = np.partition(act, target_A)[target_A]
                    # print("threshold R", threshold_R)
                    # print("threshold A", threshold_A)

                    top_RN = np.where(R[0] >= threshold_R)

                    top_AN = np.where(act >= threshold_A)

                    S = np.intersect1d(top_RN, top_AN)
                    # print("S", S)
                    # print("common Indices", commonIndices)
                    sum_R = sum(R[0])
                    # print(sum_R)
                    if reset == 1:
                        new_R = [0] * len(R[0])
                        sum_S = sum(R[0][S])
                        for index in S:
                            # print("here")
                            # print((R[0][index]/sum_S) * sum_R)
                            new_R[index] = (R[0][index]/sum_S) * sum_R
                            # print(index, new_R[index])
                        # print(new_R)
                        R[0] = np.array(new_R)
                    elif reset == 2:
                        new_R = [0] * len(R[0])
                        for index in S:
                            new_R[index] = sum_R/len(S)
                        R[0] = np.array(new_R)
                    elif reset == 3:
                        new_R = [0] * len(R[0])
                        for index in S:
                            new_R[index] = R[0][index] 
                        R[0] = np.array(new_R)
                    else:
                        print("Invalid argument, reset can only have values as 0, 1 or 2")
        except:
            pass

        if lrp_var is None and param is None:
            # module.lrp(R) has been called without further parameters.
            # set default values / preset values
            lrp_var = self.lrp_var
            param = self.lrp_param

        if lrp_var is None or lrp_var.lower() == 'none' or lrp_var.lower() == 'simple':
            return self._simple_lrp(R)
        elif lrp_var.lower() == 'slow' or lrp_var.lower() == 'simple_slow':
            return self._simple_lrp_slow(R)

        elif lrp_var.lower() == 'flat':
            return self._flat_lrp(R)
        elif lrp_var.lower() == 'ww' or lrp_var.lower() == 'w^2':
            return self._ww_lrp(R)

        elif lrp_var.lower() == 'epsilon':
            return self._epsilon_lrp(R,param)
        elif lrp_var.lower() == 'epsilon_slow':
            return self._epsilon_lrp_slow(R,param)

        elif lrp_var.lower() == 'alphabeta' or lrp_var.lower() == 'alpha':
            return self._alphabeta_lrp(R,param)
        elif lrp_var.lower() == 'alphabeta_slow' or lrp_var.lower() == 'alpha_slow':
            return self._alphabeta_lrp_slow(R,param)

        else:
            raise Exception('Unknown lrp variant {}'.format(lrp_var))


    # ---------------------------------------------------------
    # Methods below should be implemented by inheriting classes
    # ---------------------------------------------------------

    def _simple_lrp(self,R):
        raise NotImplementedError('_simple_lrp missing in ' + self.__class__.__name__)

    def _simple_lrp_slow(self,R):
        raise NotImplementedError('_simple_lrp_slow missing in ' + self.__class__.__name__)

    def _flat_lrp(self,R):
        raise NotImplementedError('_flat_lrp missing in ' + self.__class__.__name__)

    def _ww_lrp(self,R):
        raise NotImplementedError('_ww_lrp missing in ' + self.__class__.__name__)

    def _epsilon_lrp(self,R,param):
        raise NotImplementedError('_epsilon_lrp missing in ' + self.__class__.__name__)

    def _epsilon_lrp_slow(self,R,param):
        raise NotImplementedError('_epsilon_lrp_slow missing in ' + self.__class__.__name__)

    def _alphabeta_lrp(self,R,param):
        raise NotImplementedError('_alphabeta_lrp missing in ' + self.__class__.__name__)

    def _alphabeta_lrp_slow(self,R,param):
        raise NotImplementedError('_alphabeta_lrp_slow missing in ' + self.__class__.__name__)

    def to_cupy(self):
        raise NotImplementedError('to_cupy missing in ' + self.__class__.__name__)

    def to_numpy(self):
        raise NotImplementedError('to_numpy missing in ' + self.__class__.__name__)
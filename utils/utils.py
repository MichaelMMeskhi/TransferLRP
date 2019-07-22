import numpy
import PIL 
import PIL.Image
import pandas

""" 
Utility module with helper functions. 
"""

lowest = -1.0
highest = 1.0

# --------------------------------------
# Sampling data
# --------------------------------------
"""
    Loads the MNIST dataset and generates a random sample to compute
	the Layer-wise Relevance Propagation on.

    Args:
		N: Sample size.
		seed: Random seed.
		path: MNIST dataset location.

    Returns:
        X: Training instances.
		T: Training labels.
    """
def getMNISTsample(N=12,seed=None,path=''):

	fx = '%s/train-images-idx3-ubyte'%path
	ft = '%s/train-labels-idx1-ubyte'%path

	X  = numpy.fromfile(open(fx),dtype='ubyte',count=16+784*10000)[16:].reshape([10000,784])
	T  = numpy.fromfile(open(ft),dtype='ubyte',count=8+10000)[8:]
	T  = (T[:,numpy.newaxis]  == numpy.arange(10))*1.0

	if seed==None: seed=numpy.random
	else: seed=numpy.random.mtrand.RandomState(seed)

	R = seed.randint(0,len(X),[N])
	X,T = X[R],T[R]

	return X/255.0*(highest-lowest)+lowest,T

# --------------------------------------
# Color maps ([-1,1] -> [0,1]^3)
# --------------------------------------
"""
    Generates the color heatmaps to demonstrate relevance of nodes for task.

    Args:
        x: Training instances. 

    Returns:
        numpy.array: Contains RGB values of heatmap or grayscale values.
    """
def heatmap(x):

	x = x[...,numpy.newaxis]

	# positive relevance
	hrp = 0.9 - numpy.clip(x-0.3,0,0.7)/0.7*0.5
	hgp = 0.9 - numpy.clip(x-0.0,0,0.3)/0.3*0.5 - numpy.clip(x-0.3,0,0.7)/0.7*0.4
	hbp = 0.9 - numpy.clip(x-0.0,0,0.3)/0.3*0.5 - numpy.clip(x-0.3,0,0.7)/0.7*0.4

	# negative relevance
	hrn = 0.9 - numpy.clip(-x-0.0,0,0.3)/0.3*0.5 - numpy.clip(-x-0.3,0,0.7)/0.7*0.4
	hgn = 0.9 - numpy.clip(-x-0.0,0,0.3)/0.3*0.5 - numpy.clip(-x-0.3,0,0.7)/0.7*0.4
	hbn = 0.9 - numpy.clip(-x-0.3,0,0.7)/0.7*0.5

	r = hrp*(x>=0)+hrn*(x<0)
	g = hgp*(x>=0)+hgn*(x<0)
	b = hbp*(x>=0)+hbn*(x<0)

	return numpy.concatenate([r,g,b],axis=-1)


def graymap(x):

	x = x[...,numpy.newaxis]
	return numpy.concatenate([x,x,x],axis=-1)*0.5+0.5

# --------------------------------------
# Visualizing data
# --------------------------------------
"""
    Visualizes the data generated from heatmap/grayscale functions.

    Args:
        x: Training instances.
		colormap: Type of heatmap [heatmap, grayscale].
		name: Output file name.

    Returns:
        Image: Generates .png image file of our input.
    """
def visualize(x,colormap,name):

	N = len(x); assert(N<=16)

	x = colormap(x/numpy.abs(x).max())

	# Create a mosaic and upsample
	x = x.reshape([1,N,28,28,3])
	x = numpy.pad(x,((0,0),(0,0),(2,2),(2,2),(0,0)),'constant',constant_values=1)
	x = x.transpose([0,2,1,3,4]).reshape([1*32,N*32,3])
	x = numpy.kron(x,numpy.ones([2,2,1]))

	PIL.Image.fromarray((x*255).astype('byte'),'RGB').save(name)

# --------------------------------------
# Compute threshold
# --------------------------------------
"""
    Computes the threshold for deciding the relevancy contribution
	of each node of a particular layer in terms of LRP.

    Args:
        C: Matrix of C values computed for a given layer.
		i: Layer index.

    Returns:
        Ci: Numpy array of indices of least relevant nodes.
		Ni: Numpy array of indices of most relevant nodes.
    """
def noderel(C, i, net):
	threshold = (numpy.amax(C) + numpy.amin(C)) / 2
	Ci = numpy.argwhere(C < threshold)
	Ni = numpy.argwhere(C >= threshold)
	Ni = pandas.DataFrame(Ni)
	Ni.to_csv('./results/' + net + '/' + net + '_nodeRel_layer_' + str(i) + '.csv', index=False)
	return Ci
	
# -------------------------
# Probability score
# -------------------------
"""
    Prints the probability score for each output versus input.

    Args:
        Y: Matrix of predicted values.
    """
def probability(Y):
	for l in range(len(Y)): 
		print("============== Instance " + str(l) + " ==============")
		for i, o in enumerate(Y[l]): 
			print("Label: %d with probability %.2f. " % (i, o))
		print("\n")
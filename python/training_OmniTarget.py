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

base_nn = model_io.read('../mnist_mlp-full.txt')
# transferWeights = base_nn.modules[1].W
# print(transferWeights)



# --------------------- Target Network Setting --------------------------------------

Xtrain, Ytrain, Xtest, Ytest = data_io.readDataFromFolder("../data/omniglot/Asomtavruli_(Georgian)/")


print("Size of training data set ", len(Ytrain))
print("Size of testing data set", len(Ytest))

# transfer pixel values from [0 255] to [-1 1] to satisfy the expected input / training paradigm of the model
Xtrain =  Xtrain / 127.5 - 1
Xtest =  Xtest / 127.5 - 1

# transform numeric class labels to vector indicator for uniformity. assume presence of all classes within the label set
I = Ytrain[:,0].astype(int)
Ytrain = np.zeros([Xtrain.shape[0],np.unique(Ytrain).size])
Ytrain[np.arange(Ytrain.shape[0]),I] = 1

I = Ytest[:,0].astype(int)
Ytest = np.zeros([Xtest.shape[0],np.unique(Ytest).size])
Ytest[np.arange(Ytest.shape[0]),I] = 1


nn = modules.Sequential(
    [
        modules.Flatten(),
        modules.Linear(11025, 1296),
        modules.Rect(),
        modules.Linear(1296,1296),
        modules.Rect(),
        modules.Linear(1296,1296),
        modules.Rect(),
        modules.Linear(1296,1296),
        modules.Rect(),
        modules.Linear(1296,1296),
        modules.Rect(),
        modules.Linear(1296,1296),
        modules.Rect(),
        modules.Linear(1296,1296),
        modules.Rect(),
        modules.Linear(1296, 40),
        modules.SoftMax()
    ]
)

# ---------------- Initialize weights from base model ------------------
for i in range(0, 8):
	if i%2 != 0:
	    nn.modules[i].W = base_nn.modules[i].W

# ----------------- Freeze first 4 layers of new network ---------------
for i in range(0, 8):
	if i%2 != 0:
	    nn.modules[i].trainable = False


nn.train(Xtrain, Ytrain, Xtest, Ytest, batchsize=64, iters=20000, status=20000)
acc = np.mean(np.argmax(nn.forward(Xtest), axis=1) == np.argmax(Ytest, axis=1))
if not np == numpy: # np=cupy
    acc = np.asnumpy(acc)
print('model test accuracy is: {:0.4f}'.format(acc))
model_io.write(nn, '../omniglot_mlp-Target_full.txt')

#try loading the model again and compute score, see if this checks out. this time in numpy
nn = model_io.read('../omniglot_mlp-Target_full.txt')
acc = np.mean(np.argmax(nn.forward(Xtest), axis=1) == np.argmax(Ytest, axis=1))
if not np == numpy: acc = np.asnumpy(acc)
print('model test accuracy (numpy) is: {:0.4f}'.format(acc))
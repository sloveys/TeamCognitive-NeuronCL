#TEAM COGNITIVE openCL Library!
from __future__ import absolute_import, print_function
import pyopencl as cl
import numpy as np
import NeuronCL as ncl
class NeuronClass:
    sWeights
    vWeights
    def initNode(numInputArrays):
        sWeight(np.random.rand(numInputArrays).astype(np.float32))
        pass

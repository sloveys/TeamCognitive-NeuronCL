#TEAM COGNITIVE openCL Library!
from __future__ import absolute_import, print_function
import pyopencl as cl
import numpy as np
import NeuronCL as ncl
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'

class NueralNetwork(object):
    vWeight = None
    sWeight = None
    layerCount = None
    vectorSize = None
    context = None
    queue = None
    clProgram = None
    numInputArrays = None
    vectorSize = None
    layers = None


    def __init__(self, numArr, vecSize):
        self.numInputArrays = numArr
        self.vectorSize = vecSize
        self.layers = []
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)
        self.clProgram = ncl.buildNeuron(self.context)
        pass

    def addLayer(self, numNeuron):
        neuronList = []
        if (len(self.layers) == 0):
            numInputs = self.numInputArrays
        else:
            numInputs = len(self.layers[len(self.layers) - 1])
        for i in range(numNeuron):
            neuronList.append(ncl.Neuron(numInputs, self.vectorSize))
        self.layers.append(neuronList)
        pass

    def train(self, trainInput, nOutput, iterations):
        for i in range(iterations):
            layerInput = np.copy(trainInput)
            allLayerOutputs = [np.copy(trainInput)]
            for layerLevel in range(len(self.layers)):
                layerOutput = []
                for neuron in range(len(self.layers[layerLevel])):
                    layerOutput.append(self.layers[layerLevel][neuron].runNeuron(self.context, self.queue, self.clProgram, layerInput))
                layerInput = np.array(layerOutput)
                allLayerOutputs.append(np.array(layerOutput))
            guess = getTruth(layerInput)
            final_error = nOutput - guess
            lastNueronChange = final_error *nonlin(layerInput, True)


            backp = len(self.layers)
            # while (back > 0):
                # if ()
        pass

    def run(self, runInput):
        layerInput = np.copy(runInput)
        for layerLevel in range(len(self.layers)):
            layerOutput = []
            for neuron in range(len(self.layers[layerLevel])):
                layerOutput.append(self.layers[layerLevel][neuron].runNeuron(self.context, self.queue, self.clProgram, layerInput))
            layerInput = np.array(layerOutput)
        return(layerInput)

def nonlin(x, deriv=False):
    if (deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def getTruth(matrix):
    return nonlin(matrix[0])

np.random.seed(42)
nn = NueralNetwork(1, 16)
nn.addLayer(3)
nn.addLayer(2)
nn.addLayer(1)
add_np = np.random.rand(16*16).astype(np.float32)
mArr = np.array([add_np])
nn.train(mArr, 1, 1)
print(getTruth(nn.run(mArr)[0]))

#TEAM COGNITIVE openCL Library!
from __future__ import absolute_import, print_function
import pyopencl as cl
import numpy as np
import NeuronCL as ncl
class NueralNetwork:
    vWeight = None
    sWeight = None
    layerCount = None
    vectorSize = None
    context = None
    queue = None
    clProgram = None
    numInputArrays = None
    vectorSize = None
    def nonlin(x, deriv=False):
        if (deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))


    def initNN(numArr, vecSize):
        numInputArrays = numArr
        vectorSize = vecSize
        layers = []
        context = cl.create_some_context()
        queue = cl.CommandQueue(context)
        clProgram = buildNeuron(context)
        pass

    def addLayer(numNeuron):
        neuronList = []
        for i in range(numNeuron):
            neuronList.append(Neuron(numInputArrays, vectorSize))
        layers.append(neuronList)
        pass

    def train(trainInput, trainOutput, iterations):

        pass

    def run(runInput):
        layerInput = np.copy(runInput)

        for layerLevel in range(len(layers)):
            layerOutput = np.array(len(layers[layerLevel]))
            for neuron in range(len(layers[layerLevel])):
                #queue program matrix
                layerOutput[neuron] = layers[layersLevel][neuron].runNeuron(queue, clProgram, runInput)
            layerInput = layerOutput
    # def updateVariables():

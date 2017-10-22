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
            lastNueronChange = (final_error *nonlin(getTruth(layerInput), True)).astype(np.float32)
            print(lastNueronChange)
            self.layers[layerLevel][neuron].bias += np.dot(np.array(allLayerOutputs[len(self.layers) - 1]).T, lastNueronChange)
            print(guess)
            backp = len(self.layers)
            # while (back > 0):
                # if ()
        pass


    def train2(self, trainInput, nOutput, iterations, tester):
        for i in range(iterations):
            for layerLevelInv in range(len(self.layers)):
                layerLevel = len(self.layers) - 1 - layerLevelInv
                for neuron in range(len(self.layers[layerLevel])):
                    for vecW in range(self.vectorSize):
                        matrix = self.run(trainInput)[0]
                        error = nOutput - matrix[0]
                        currT = getTruth(matrix)
                        print("Actual Output: ",nOutput,"\nAlgorithm guess: ",matrix[0],"\nConfidince: ",currT,"\nTest: ",tester)
                        if (error < 0.01 and error > 0.01 and currT > 0.95):
                            pass
                        self.layers[layerLevel][neuron].vectorWeight[vecW] += error / currT


                    for vecW2 in range(self.vectorSize):
                        matrix = self.run(trainInput)[0]
                        error = nOutput - matrix[0]
                        currT = getTruth(matrix)
                        #print("Actual Output: ",nOutput,"\nAlgorithm guess: ",matrix[0],"\nConfidince: ",currT,"\nTest: ",tester)
                        if (error < 0.01 and error > 0.01 and currT > 0.95):
                            pass
                        self.layers[layerLevel][neuron].vectorWeight2[vecW2] += error / currT

                    for sclW in range(len(self.layers[layerLevel][neuron].bias)):
                        matrix = self.run(trainInput)[0]
                        error = nOutput - matrix[0]
                        currT = getTruth(matrix)
                        #print("Actual Output: ",nOutput,"\nAlgorithm guess: ",matrix[0],"\nConfidince: ",currT,"\nTest: ",tester)
                        if (error < 0.01 and error > 0.01 and currT > 0.95):
                            pass
                        self.layers[layerLevel][neuron].bias[sclW] += error / currT


        print(getTruth(self.run(trainInput)[0]))
        #print(nOutput)
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
    # print(x)
    if (deriv==True):
        return x*(1-x)
    return 1.0/(1.0+np.exp(-x.astype(np.float64)))

def getTruth(matrix):
    nl = nonlin(matrix[0])
    if (nl == 0.0):
        return 0.5
    return nl

np.random.seed(21)
nn = NueralNetwork(1, 16)
nn.addLayer(2)
nn.addLayer(2)
nn.addLayer(1)
# add_np = np.random.rand(16*16).astype(np.float32)
# mArr = np.array([add_np])
# nn.train2(mArr, 1.0, 1)
add_np = np.random.rand(16*16).astype(np.float32)
mArr = np.array([add_np])

num_lines = sum(1 for line in open('datafile.txt'))
for p in range (3):
    # print("test")
    f = open("dataFile.txt", 'r')
    for i in range (num_lines - 1):
        matrix = np.zeros((16*16)).astype(np.float32)
        for j in range(16):
            for p in range(16):
                if (f.read(1) is "1"):
                    matrix[j*p] = np.float32(1.0)
        f.read(1)
        if (i > 18):
            nn.train2(np.array([matrix]), 1, 1, i)
        # elif i> 16:
        #     nn.train2(np.array([matrix]), 1, 1, i)
        print("Next testcase: ")


print(getTruth(nn.run(mArr)[0]))

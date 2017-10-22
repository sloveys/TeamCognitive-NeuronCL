#TEAM COGNITIVE openCL Library!
class NueralNetwork:
    vWeight
    sWeight
    layerCount
    layers

    def nonlin(x, deriv=False):
        if (deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))


    def initNN():
        layerCount = 0
        vWeight = []
        layers = []
        sWeight = []
        return(1)

    def addLayer(layerSize):
        if (layerCount == 0):
            layers = [layerSize]
        else:
            layers.append(layerSize)
        layerCount++
        return()
    def train(trainInput, trainOutput, iterations):

        return()
    def run(runInput):
        np.random.seed(2017)
        layerOutput = np.copy(runInput)
        for iter in layerCount:
            layer

        return()
    # def updateVariables():

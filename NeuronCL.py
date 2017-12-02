from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
# import os
# os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
# os.environ['PYOPENCL_CTX'] = '0:0' # for our GPU change '0:0' to '1'
# should build dynamic system later for choosing GPGPUs
# vectorSize = 16
# matrixSize = vectorSize*vectorSize;

def buildNeuron(context):
    program = cl.Program(context, """
    __kernel void sumMatrix(__global const float *add_g, __global const float *bdd_g, __global float *des_g)
    {
      int grid = get_global_id(0);
      des_g[grid] = add_g[grid] + bdd_g[grid];
    }
    __kernel void scalarWeight(float weight, __global const float *matrix, __global float *matrixDes)
    {
      int grid = get_global_id(0);
      matrixDes[grid] = matrix[grid] * weight;
    }
    __kernel void dotPrime(int vecSize, __global float *vecWeight, __global const float *matrix, __global float *vecDes)
    {
      int grid = get_global_id(0);
      vecDes[grid] = 0.0;
      for (int i=0; i < vecSize; i++) {
        vecDes[grid] += matrix[(vecSize*grid) + i] * vecWeight[i];
      }
    }
    __kernel void dotSecondary(int vecSize, __global float *vecWeight, __global const float *matrix, __global float *vecDes)
    {
      int grid = get_global_id(0);
      vecDes[grid] = 0.0;
      for (int i=0; i < vecSize; i++) {
        vecDes[grid] += matrix[(vecSize*i) + grid] * vecWeight[i];
      }
    }
    __kernel void fullMultiply(int vecSize, __global const float *aVec, __global const float *bVec, __global float *matrix)
    {
      int grid = get_global_id(0);
      int x = grid / vecSize;
      int y = grid % vecSize;
      matrix[grid] = aVec[x]*bVec[y];
    }
    __kernel void nonLinNorm(__global const float *matrix, __global float *mOut)
    {
      int grid = get_global_id(0);
      mOut[grid] = 1.0/(1.0 + exp(-matrix[grid]));
    }
    """).build()
    return program


class Neuron(object):
    vectorSize = 0
    mSize = 0
    vectorWeight = np.array([])
    vectorWeight2 = np.array([])
    bias = np.array([])

    def __init__(self, numInputArr, vecSize):
        self.vectorSize = vecSize
        self.mSize = vecSize*vecSize
        self.vectorWeight = 2.0*np.random.rand(vecSize).astype(np.float32) - 1.0
        self.vectorWeight2 = 2.0*np.random.rand(vecSize).astype(np.float32) - 1.0
        self.bias = 2.0*np.random.rand(numInputArr).astype(np.float32) - 1.0
        pass

    # returns normalized Matrix out
    def runNeuron(self, ctx, queue, program, matrixArr):
        inputs = len(matrixArr)
        weightAve = np.float32(0.0001)

        mf = cl.mem_flags
        vecW_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.vectorWeight)
        vecW2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.vectorWeight2)
        aVec_g = cl.Buffer(ctx, mf.WRITE_ONLY, 32*self.vectorSize)
        bVec_g = cl.Buffer(ctx, mf.WRITE_ONLY, 32*self.vectorSize)
        for i in range(inputs):
            matrix_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=matrixArr[i])
            program.scalarWeight(queue, (self.mSize,), None, np.float32(self.bias[i]), matrix_g, matrix_g)
            program.dotPrime(queue, (self.mSize,), None, np.int32(self.vectorSize), vecW_g, matrix_g, aVec_g)
            program.dotSecondary(queue, (self.mSize,), None, np.int32(self.vectorSize), vecW2_g, matrix_g, bVec_g)
            program.fullMultiply(queue, (self.mSize,), None, np.int32(self.vectorSize), aVec_g, bVec_g, matrix_g)
            if (i == 0):
                mOut_g = matrix_g
            else:
                program.sumMatrix(queue, (self.mSize,), None, mOut_g, matrix_g, mOut_g)

        program.nonLinNorm(queue, (self.mSize,), None, mOut_g, mOut_g)
        mOut_np = np.empty_like(matrixArr[0])
        cl.enqueue_copy(queue, mOut_np, mOut_g)
        return mOut_np

#np.random.seed(41)
#add_np = np.random.rand(16*16).astype(np.float32)
# bdd_np = np.random.rand(16*16).astype(np.float32)
# mtrxArr = np.array([add_np, bdd_np])
#ctx = cl.create_some_context()
#queue = cl.CommandQueue(ctx)
#
# nrn = Neuron(2, 16)
#prg = buildNeuron(ctx)
#
# matrixOut = nrn.runNeuron(queue, prg, mtrxArr)
#mf = cl.mem_flags
#matrix_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=add_np)
#prg.nonLinNorm(queue, (16*16,), None, matrix_g, matrix_g)
#mOut_np = np.empty_like(add_np)
#cl.enqueue_copy(queue, mOut_np, matrix_g)


#print(add_np)
#print(mOut_np)

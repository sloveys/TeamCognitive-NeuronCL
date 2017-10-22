from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '0:0' # for our GPU change '0:0' to '1'
# should build dynamic system later for choosing GPGPUs

vectorSize = 16
matrixSize = vectorSize*vectorSize;

#Returns Program
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
    """).build()
    return program

# returns normalized Matrix out
def runNeuron(queue, program, vectorSize, matrixArr, bias, vectorWeight):
    mSize = vectorSize*vectorSize
    inputs = len(matrixArr)
    weightAve = np.float32(0.0)

    mf = cl.mem_flags
    vecW_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vec_np)
    aVec_g = cl.Buffer(ctx, mf.WRITE_ONLY, 32*vectorSize)
    bVec_g = cl.Buffer(ctx, mf.WRITE_ONLY, 32*vectorSize)
    for i in range(inputs):
        matrix_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=matrixArr[i])
        program.scalarWeight(queue, (mSize,), None, np.float32(bias2[i]), matrix_g, matrix_g)
        program.dotPrime(queue, (mSize,), None, np.int32(vectorSize), vecW_g, matrix_g, aVec_g)
        program.dotSecondary(queue, (mSize,), None, np.int32(vectorSize), vecW_g, matrix_g, bVec_g)
        program.fullMultiply(queue, (mSize,), None, np.int32(vectorSize), aVec_g, bVec_g, matrix_g)
        weightAve += np.float32(bias[i])
        if (i == 0):
            mOut_g = matrix_g
        else:
            program.sumMatrix(queue, (mSize,), None, mOut_g, matrix_g, mOut_g)

    weightAve = weightAve/np.float32(inputs)
    program.scalarWeight(queue, (mSize,), None, np.float32(1/(inputs*weightAve*vectorSize)), mOut_g, mOut_g)
    mOut_np = np.empty_like(matrixArr[0])
    cl.enqueue_copy(queue, mOut_np, mOut_g)
    return mOut_np

np.random.seed(41)
add_np = np.random.rand(matrixSize).astype(np.float32)
bdd_np = np.random.rand(matrixSize).astype(np.float32)
vec_np = np.random.rand(vectorSize).astype(np.float32)
bias2 = np.array([1.0, 0.9, 0.8])
mtrxArr = np.array([add_np, bdd_np])

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

prg = buildNeuron(ctx)

#queue, program, vectorSize, matrixArr, bias, vectorWeight
matrixOut = runNeuron(queue, prg, vectorSize, mtrxArr, bias2, vec_np)
mtrxArr = np.array([add_np, bdd_np, matrixOut])
matrixOut2 = runNeuron(queue, prg, vectorSize, mtrxArr, bias2, vec_np)


print(matrixOut)
print(matrixOut2)

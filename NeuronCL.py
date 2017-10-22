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
    __kernel void mulMatrix(__global const float *aul_g, __global const float *bul_g, __global float *des_g)
    {
      int grid = get_global_id(0);
      des_g[grid] = aul_g[grid] * bul_g[grid];
    }
    /*__kernel void dot(__global const int *vecSize, __global float *vecDes)
    {
      int grid = get_global_id(0);
      vecDes[grid] = *vecSize;
    }*/
    """).build()
    return program
#, __global const float *vecWeight, __global const float *mtx

np.random.seed(42)
add_np = np.random.rand(matrixSize).astype(np.float32)
bdd_np = np.random.rand(matrixSize).astype(np.float32)

vwait - np.random.rand(3).astype(np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
add_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=add_np)
bdd_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bdd_np)

prg = buildNeuron(ctx)

des_g = cl.Buffer(ctx, mf.WRITE_ONLY, 32*matrixSize) #add_np
prg.mulMatrix(queue, add_np.shape, None, add_g, bdd_g, des_g)

des_np = np.empty_like(add_np)
cl.enqueue_copy(queue, des_np, des_g)

# Check on CPU with Numpy:
print(des_np)

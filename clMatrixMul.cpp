#include <iostream>
#include "LSH.h"
#include "SignedRandomProjection.h"
#include <string>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include "lshr.h"
#include "all.h"
#include "indexing.h"
#include "param.h"

int Bucket::_size = 64; //use more if the buckets are heavy

using namespace std;


void parMatMul(int K, int L, double* A, double* B, double* C, int N, int M, int P)
{   
    //Create LSH (Assuming that this runs in parallel) =============================
    //Assuming that lshr_init.cpp initiates the kernel.
    //TODO add MatrixMul Kernel to lshr_init
    LSHReservoirSampler *myReservoir = new LSHReservoirSampler(NUMTABLES, NUMHASH, RESERVOIR_SIZE, PROBES, SAMFACTOR);
    for (size_t c = 0; c < Mdim; c++)
    {
        float arr[Pdim];
        for (size_t r = 0; r < Pdim; r++)
        {
            arr[r] = (float) B[r * Mdim + c];
            cout<<arr[r]<<" ";
        }
        myReservoir->hashToTable(Pdim, arr);
    }

    //Create buffers for the input and output
    inputA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * (N * P), NULL, NULL);
    inputB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * (P * M), NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * (M * M)), NULL, NULL);
    
    //Load data into the input buffer
    clEnqueueWriteBuffer(command_queue, inputA, CL_TRUE, 0, sizeof(double) * (N * P), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, inputB, CL_TRUE, 0, sizeof(double) * (P * M), B, 0, NULL, NULL);

    // set the argument list for the kernel command
    clSetKernelArg(kernel, 0, sizeof(int), (void *)&M);
    clSetKernelArg(kernel, 1, sizeof(int), (void *)&N);
    clSetKernelArg(kernel, 2, sizeof(int), (void *)&P);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &inputA);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &inputB);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 6, sizeof(LSHReservoirSampler), (void *)myReservoir)
    // The arguments will differ according to the LSH_GPU code.

    // enqueue the kernel command for execution
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    clFinish(command_queue);

    //copy the results from out of the output buffer
    clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, sizeof(double) * (N * M), C, 0, NULL, NULL);
 
    //clean memory
    clReleaseMemObject(inputA);
    clReleaseMemObject(inputB);
    clReleaseMemObject(output);
}

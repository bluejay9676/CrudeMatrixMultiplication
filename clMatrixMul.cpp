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

int Bucket::_size = 64; //use more if the buckets are heavy

using namespace std;


void parMatMul(int K, int L, double* A, double* B, double* C, int N, int M, int P)
{   
    //Assuming that lshr_init.cpp initiates the kernel.

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

    //Create LSH (Assuming that this runs in parallel) =============================
    //This could go away depending on the LSH_GPU code.
    /*
    LSH *_Algo = new LSH(K, L);
    SignedRandomProjection *proj = new SignedRandomProjection(Mdim, K * L);
    cout<<"Making Hash Table from B."<<endl;
    for (size_t c = 0; c < Mdim; c++)
    {
        double arr[Pdim];
        for (size_t r = 0; r < Pdim; r++)
        {
            arr[r] = B[r * Mdim + c];
            cout<<arr[r]<<" ";
        }
        cout<<endl;
        int * hashes = proj->getHash(arr, Pdim);
        _Algo->add(hashes, c + 1);
    }
    //=============================================================================== 

    //Run each row of A in parallel.
    //Divide into N
    divideRow(M, N, P, (double *) A, (double *) B, (double *) C, (LSH *) _Algo, (SignedRandomProjection *) proj);
    */
}

#include <iostream>
#include "LSH.h"
#include "SignedRandomProjection.h"
#include <string>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>

int Bucket::_size = 64; //use more if the buckets are heavy

using namespace std;


__kernel
void divideRow(int Mdim, int Ndim, int Pdim, 
    __global double* A,
    __global double* B,
    __global double* C, 
    __global LSH* _Algo,
    __global SignedRandomProjection* proj)
{
    //Run vector multiplication for each columns from the retrieved LSH bucket.
    int i = get_global_id(0); // Index of a row of A
    int *queryhashes = proj->getHash(&A[i * Pdim], Pdim);
    int *retrieved = _Algo->retrieve(queryhashes);//Indices of bucket that collided.

    if (i < Ndim)
    {
        for (int l = 2; l < retrieved[0] + 2; l++)
        {
            for (int k = 0; k < Pdim; k++)
            {
                C[i * Mdim + retrieved[l] - 1] += A[i * Ndim + k] * B[k * Pdim + retrieved[l] - 1];
            }
        }
    }
}

void parMatMul(int K, int L, double* A, double* B, double* C, int N, int M, int P)
{
    //Create LSH (Assuming that this runs in parallel) =============================
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
    
    //Divide into N
    divideRow(M, N, P, (double *) A, (double *) B, (double *) C, (LSH *) _Algo, (SignedRandomProjection *) proj);
}

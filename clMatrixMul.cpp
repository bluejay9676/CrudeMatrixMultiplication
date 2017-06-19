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

    //Run each row of A in parallel.
    //Divide into N
    divideRow(M, N, P, (double *) A, (double *) B, (double *) C, (LSH *) _Algo, (SignedRandomProjection *) proj);
}

#include "LSH.h"
#include "SignedRandomProjection.h"

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
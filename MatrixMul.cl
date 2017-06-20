#include "LSH.h"
#include "SignedRandomProjection.h"

__kernel
void divideRow(int Mdim, int Ndim, int Pdim, 
    __global double* A,
    __global double* B,
    __global double* C, 
    __global LSH* _Algo,
    __global SignedRandomProjection* proj,
    __global LSHReservoirSampler* lsh)
{
    int num_queries = 1;
    int available_nearest = Mdim;
    int topk = 5;
    //Run vector multiplication for each columns from the retrieved LSH bucket.
    int i = get_global_id(0); // Index of a row of A
    unsigned int *queryOutputs = new unsigned int[num_queries * topk]; // Index of collided columns on matrix B
    lsh->getNN(num_queries, &A[i * Pdim], queryOutputs, topk);

    if (i < Ndim)
    {
        for( unsigned int l = 0; l < sizeof(queryOutputs)/sizeof(queryOutputs[0]); l++ )
        {
            for (int k = 0; k < Pdim; k++)
            {
                C[i * Mdim + queryOutputs[l]] += A[i * Ndim + k] * B[k * Pdim + queryOutputs[l]];
            }
        }
    }
}
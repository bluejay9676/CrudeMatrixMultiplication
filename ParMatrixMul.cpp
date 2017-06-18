#include <iostream>
#include "LSH.h"
#include "SignedRandomProjection.h"
#include <string>
#include <ctime>

int Bucket::_size = 64; //use more if the buckets are heavy

using namespace std;


__global__
void divide_row(int Mdim, int Ndim, int Pdim, double* A, double* B, double* C, LSH* _Algo, SignedRandomProjection* proj)
{
    //Run vector multiplication for each columns from the retrieved LSH bucket.
    int i = threadIdx.x; // Index of a row of A
    int *queryhashes = proj->getHash(&A[i * Pdim], Pdim);
    int *retrieved = _Algo->retrieve(queryhashes);//Indices of bucket that collided.

    for (int l = 2; l < retrieved[0] + 2; l++)
    {
        for (int k = 0; k < Pdim; k++)
        {
            C[i * Mdim + retrieved[l] - 1] += A[i * Ndim + k] * B[k * Pdim + retrieved[l] - 1];
        }
    }
}

void par_mat_mul(int K, int L, double* A, double* B, double* C, int N, int M, int P)
{
    //Create LSH (Assuming that this runs in parallel)
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

    const dim3 blockSize(N, 1, 1);
    const dim3 gridSize(1, 1, 1);
    mat_mul<<<gridSize, blockSize>>>(M, N, P, (double *) A, (double *) B, (double *) C, (LSH *) _Algo, (SignedRandomProjection *) proj);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

int main() {  
    //parameters (very critical to set properly)
    int K = 25;
    int L = 10;

    //Data
    double B[5][5] =
    {
        1,5,5,19,4,
        -8,9,9,3,8,
        12,13,13,4,12,
        2,24,24,-13,26,
        -25,26,26,2,26
    };


    int N = 3, M = 5, P = 5;

    double A[3][5] = 
    {
        5,9,13,24,26,
        18.3,3.3,4,-12,2,
        1,-8,12,2,-25
    };    

    double C[3][5] = {0};

    par_mat_mul(K, L, (double *) A, (double *) B, (double *) C, N, M, P);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            cout<<C[i][j]<<" ";
        }
        cout<<endl;
    }

    return 0;
}  

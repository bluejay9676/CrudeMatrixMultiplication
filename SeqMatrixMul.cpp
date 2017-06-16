
void mat_mul(int Mdim, int Ndim, int Pdim, double *A, double *B, double *C)
{
    int i, j, k;
    for (i = 0; i < Ndim; i++)
    {
        for (j = 0; j < Mdim; j++)
        {
            C[i * N + j] = 0.0;
            for (k = 0; k < Pdim; k++)
            {
                C[i * Ndim + j] += A[i * Ndim + k] * B[k * Pdim + j];
            }
        }
    }
}


int main() {  
    double simpleMatrix1[3][3] = {
            { 0.0, 1.0, 2.0 },
            { 3.0, 4.0, 5.0 },
            { 6.0, 7.0, 8.0 }
    };

    double simpleMatrix3[3][2] = {
            { 18.0, 19.0 },
            { 20.0, 21.0 },
            { 22.0, 23.0 }
    };

    double C[3][2];

    mat_mul()

    return 0;  
}  

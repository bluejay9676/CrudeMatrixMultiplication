#include <stdlib.h>
#include <stdio.h> 
#include <time.h>
#include "Matrix.h"
using namespace std;

/*
Author: Jay Ryu
Jun.16.2017
*/
Matrix::Matrix(size_t rows, size_t cols, bool randomize)
: mRows(rows),
  mCols(cols),
  mData(rows * cols)
{
	if (randomize)
	{
		srand (time(NULL));
		for (int r = 0; r < mRows; r++)
		{
			for (int c = 0; c < mCols; c++)
			{
				mData[r * mCols + c] = rand() % 100; 
			}
		}
	}
}

Matrix::Matrix(size_t rows, size_t cols, double* vals)
: mRows(rows),
  mCols(cols),
  mData(rows * cols)
{
	static_assert(sizeof(vals) == rows * cols, "The input array size doesn't match input rows and cols values.");
	for (int r = 0; r < mRows; r++)
	{
		for (int c = 0; c < mCols; c++)
		{
			mData[r * mCols + c] = vals[r * mCols + c]; 
		}
	}
}

double& Matrix::operator()(size_t i, size_t j)
{
    return mData[i * mCols + j];
}

double Matrix::operator()(size_t i, size_t j) const
{
    return mData[i * mCols + j];
}

Matrix Matrix::parallelMultiply(Matrix* other)
{	
	int outNRows = mRows;
	int outNCols = other.ncols();
	double C[outNRows][outNCols];
	//TODO:
	for (int r = 0; r < outNRows; r++)
	{
		for (int c = 0; c < outNCols; c++)
		{
			C[r][c] = 0.0;
			for (int i = 0; i < mCols; i++)
			{
				C[r * outNCols + c] += mData[r * mCols + i] * other.values()[i * outNCols + c];
			}
		}
	}

	return new Matrix(outNRows, outNCols, &C);
}

size_t nrows()
{
	return mRows;
}

size_t ncols()
{
	return mCols;
}

double* values()
{
	return mData;
}
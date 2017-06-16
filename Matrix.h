/*
Author: Jay Ryu
Jun.16.2017
*/

class Matrix
{
public:
    Matrix(size_t rows, size_t cols, bool randomize);
    Matrix(size_t rows, size_t cols, double* vals);
    double& operator()(size_t i, size_t j);
    double operator()(size_t i, size_t j) const;
    Matrix parallelMultiply(Matrix& other);
    size_t nrows();
    size_t ncols();
    double* values();

private:
    size_t mRows;
    size_t mCols;
    double* mData;
};
#include "Kernels.hpp"

__global__
void TestKernel( double* x, double* y, double* result, unsigned size )
{
  int start     = blockIdx.x * blockDim.x + threadIdx.x;
  int increment = blockDim.x * gridDim.x;

  for( int index = start; index < size; index += increment ) 
  {
    result[index] = x[index] + y[index];
  }
}

//template void GPUAddition_( double* host_a, double* host_b, double* host_result, int rows, int columns );
//template void GPUAddition_( float*  host_a, float*  host_b, float*  host_result, int rows, int columns );
//template void GPUAddition_( int*    host_a, int*    host_b, int*    host_result, int rows, int columns );
void GPUAddition_( double* host_a, double* host_b, double* host_result, int rows, int columns )
{
  GPUAddition<double>( host_a, host_b, host_result, rows, columns );
}

void GPUMultiplication_( double* host_a, double* host_b, double* host_result, int a_rows, int a_columns, int b_columns )
{
  GPUAddition<double>( host_a, host_b, host_result, a_rows, a_columns, b_columns );
}

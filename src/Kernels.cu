#include <Kernels.hpp>

////////////////////////////////////////////// ADDITION //////////////////////////////////////////////

void GPUAddition_( double* host_a, const double* host_b, double* host_result, int rows, int columns )
{
  GPUAddition<double>( host_a, host_b, host_result, rows, columns );
}

void GPUAddition_( float* host_a, const float* host_b, float* host_result, int rows, int columns )
{
  GPUAddition<float>( host_a, host_b, host_result, rows, columns );
}

void GPUAddition_( int* host_a, const int* host_b, int* host_result, int rows, int columns )
{
  GPUAddition<int>( host_a, host_b, host_result, rows, columns );
}

////////////////////////////////////////////// DIFFERENCE //////////////////////////////////////////////

void GPUDifference_( double* host_a, const double* host_b, double* host_result, int rows, int columns )
{
  GPUDifference<double>( host_a, host_b, host_result, rows, columns );
}

void GPUDifference_( float* host_a, const float* host_b, float* host_result, int rows, int columns )
{
  GPUDifference<float>( host_a, host_b, host_result, rows, columns );
}

void GPUDifference_( int* host_a, const int* host_b, int* host_result, int rows, int columns )
{
  GPUDifference<int>( host_a, host_b, host_result, rows, columns );
}

////////////////////////////////////////////// MULTIPLICATION //////////////////////////////////////////////

void GPUMultiplication_( double* host_a, const double* host_b, double* host_result, int a_rows, int a_columns, int b_columns )
{
  GPUMultiplication<double>( host_a, host_b, host_result, a_rows, a_columns, b_columns );
}

void GPUMultiplication_( float* host_a, const float* host_b, float* host_result, int a_rows, int a_columns, int b_columns )
{
  GPUMultiplication<float>( host_a, host_b, host_result, a_rows, a_columns, b_columns );
}

void GPUMultiplication_( int* host_a, const int* host_b, int* host_result, int a_rows, int a_columns, int b_columns )
{
  GPUMultiplication<int>( host_a, host_b, host_result, a_rows, a_columns, b_columns );
}

////////////////////////////////////////// MULTIPLICATION SQUARE //////////////////////////////////////////

void GPUMultiplication_( double* host_a, const double host_b, double* host_result, int rows, int columns )
{
  GPUMultiplication<double>( host_a, host_b, host_result, rows, columns );
}

void GPUMultiplication_( float* host_a, const float host_b, float* host_result, int rows, int columns )
{
  GPUMultiplication<float>( host_a, host_b, host_result, rows, columns );
}

void GPUMultiplication_( int* host_a, const int host_b, int* host_result, int rows, int columns )
{
  GPUMultiplication<int>( host_a, host_b, host_result, rows, columns );
}

////////////////////////////////////////////// HADAMARD //////////////////////////////////////////////

void GPUHadamard_( double* host_a, const double* host_b, double* host_result, int rows, int columns )
{
  GPUHadamard<double>( host_a, host_b, host_result, rows, columns );
}

void GPUHadamard_( float* host_a, const float* host_b, float* host_result, int rows, int columns )
{
  GPUHadamard<float>( host_a, host_b, host_result, rows, columns );
}

void GPUHadamard_( int* host_a, const int* host_b, int* host_result, int rows, int columns )
{
  GPUHadamard<int>( host_a, host_b, host_result, rows, columns );
}

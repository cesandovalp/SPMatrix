#include <Kernels.hpp>

////////////////////////////////////////////// ADDITION //////////////////////////////////////////////

void GPUAddition_( double* host_a, const double* d_b, double* host_result, int rows, int columns )
{
  GPUAddition<double>( host_a, d_b, host_result, rows, columns );
}

void GPUAddition_( float* host_a, const float* d_b, float* host_result, int rows, int columns )
{
  GPUAddition<float>( host_a, d_b, host_result, rows, columns );
}

void GPUAddition_( int* host_a, const int* d_b, int* host_result, int rows, int columns )
{
  GPUAddition<int>( host_a, d_b, host_result, rows, columns );
}

////////////////////////////////////////////// DIFFERENCE //////////////////////////////////////////////

void GPUDifference_( double* host_a, const double* d_b, double* host_result, int rows, int columns )
{
  GPUDifference<double>( host_a, d_b, host_result, rows, columns );
}

void GPUDifference_( float* host_a, const float* d_b, float* host_result, int rows, int columns )
{
  GPUDifference<float>( host_a, d_b, host_result, rows, columns );
}

void GPUDifference_( int* host_a, const int* d_b, int* host_result, int rows, int columns )
{
  GPUDifference<int>( host_a, d_b, host_result, rows, columns );
}

////////////////////////////////////////////// MULTIPLICATION //////////////////////////////////////////////

void GPUMultiplication_( double* host_a, const double* d_b, double* host_result, int a_rows, int a_columns, int b_columns )
{
  GPUMultiplication<double>( host_a, d_b, host_result, a_rows, a_columns, b_columns );
}

void GPUMultiplication_( float* host_a, const float* d_b, float* host_result, int a_rows, int a_columns, int b_columns )
{
  GPUMultiplication<float>( host_a, d_b, host_result, a_rows, a_columns, b_columns );
}

void GPUMultiplication_( int* host_a, const int* d_b, int* host_result, int a_rows, int a_columns, int b_columns )
{
  GPUMultiplication<int>( host_a, d_b, host_result, a_rows, a_columns, b_columns );
}

////////////////////////////////////////// MULTIPLICATION SQUARE //////////////////////////////////////////

void GPUMultiplication_( double* host_a, const double b, double* host_result, int rows, int columns )
{
  GPUMultiplication<double>( host_a, b, host_result, rows, columns );
}

void GPUMultiplication_( float* host_a, const float b, float* host_result, int rows, int columns )
{
  GPUMultiplication<float>( host_a, b, host_result, rows, columns );
}

void GPUMultiplication_( int* host_a, const int b, int* host_result, int rows, int columns )
{
  GPUMultiplication<int>( host_a, b, host_result, rows, columns );
}

////////////////////////////////////////////// HADAMARD //////////////////////////////////////////////

void GPUHadamard_( double* host_a, const double* d_b, double* host_result, int rows, int columns )
{
  GPUHadamard<double>( host_a, d_b, host_result, rows, columns );
}

void GPUHadamard_( float* host_a, const float* d_b, float* host_result, int rows, int columns )
{
  GPUHadamard<float>( host_a, d_b, host_result, rows, columns );
}

void GPUHadamard_( int* host_a, const int* d_b, int* host_result, int rows, int columns )
{
  GPUHadamard<int>( host_a, d_b, host_result, rows, columns );
}

/////////////////////////////////////////////// ASSIGN //////////////////////////////////////////////

void GPUAssign_( const double* host_a, double** d_a, int rows, int columns )
{
  GPUAssign<double>( host_a, d_a, rows, columns );
}

void GPUAssign_( const float* host_a, float** d_a, int rows, int columns )
{
  GPUAssign<float>( host_a, d_a, rows, columns );
}

void GPUAssign_( const int* host_a, int** d_a, int rows, int columns )
{
  GPUAssign<int>( host_a, d_a, rows, columns );
}

//////////////////////////////////////////////// FREE ///////////////////////////////////////////////

void GPUFree_( double* device_data )
{
  GPUFree<double>( device_data );
}

void GPUFree_( float* device_data )
{
  GPUFree<float>( device_data );
}

void GPUFree_( int* device_data )
{
  GPUFree<int>( device_data );
}

//////////////////////////////////////////////// COPY ///////////////////////////////////////////////

void GPUCopy_( const double* host_a, double* d_a, int rows, int columns )
{
  GPUCopy<double>( host_a, d_a, rows, columns );
}

void GPUCopy_( const float* host_a, float* d_a, int rows, int columns )
{
  GPUCopy<float>( host_a, d_a, rows, columns );
}

void GPUCopy_( const int* host_a, int* d_a, int rows, int columns )
{
  GPUCopy<int>( host_a, d_a, rows, columns );
}

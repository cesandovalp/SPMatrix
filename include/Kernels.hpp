#pragma once

#include <cstdio>
#include <iostream>

#define GPUErrchk( ans ) { GPUAssert( ( ans ), __FILE__, __LINE__ ); }

inline void GPUAssert( cudaError_t code, const char *file, int line, bool abort = true )
{
  if( code != cudaSuccess )
  {
    fprintf( stderr, "GPUassert: %s %s %d\n", cudaGetErrorString( code ), file, line );
    if( abort )
      exit( code );
  }
}

template<class domain>
__global__
void AdditionKernel( domain* a, const domain* b, domain* result, int rows, int columns )
{
  int start     = blockIdx.x * blockDim.x + threadIdx.x;
  int increment = blockDim.x * gridDim.x;
  
  for( int index = start; index < rows * columns; index += increment ) 
    result[index] = a[index] + b[index];
}

template<class domain>
__global__
void DifferenceKernel( domain* a, const domain* b, domain* result, int rows, int columns )
{
  int start     = blockIdx.x * blockDim.x + threadIdx.x;
  int increment = blockDim.x * gridDim.x;
  
  for( int index = start; index < rows * columns; index += increment ) 
    result[index] = a[index] - b[index];
}

template<class domain>
__global__
void MultiplicationKernel( domain* a, const domain* b, domain* result, int a_rows, int a_columns, int b_columns )
{ 
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int sum = 0;
  
  if( col < b_columns && row < a_rows ) 
  {
    for( int i = 0; i < a_rows; i++ )
      sum += a[row * a_columns + i] * b[i * b_columns + col];
    result[row * b_columns + col] = sum;
  }
}

template<class domain>
__global__
void MultiplicationKernel( domain* a, const domain b, domain* result, int a_rows, int a_columns )
{ 
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if( col < a_columns && row < a_rows ) 
  {
    for( int i = 0; i < a_rows; i++ )
      result[row * a_columns + i] = a[row * a_columns + i] * b;
  }
}

template<class domain>
__global__
void HadamardKernel( domain* a, const domain* b, domain* result, int rows, int columns )
{
  int start     = blockIdx.x * blockDim.x + threadIdx.x;
  int increment = blockDim.x * gridDim.x;
  
  for( int index = start; index < rows * columns; index += increment ) 
    result[index] = a[index] * b[index];
}

template<class domain>
void GPUAddition( domain* host_a, const domain* d_b, domain* host_result, int rows, int columns )
{
  int block_size = 1024;
  int num_blocks = ( ( rows * columns ) / block_size ) + 1;
  
  domain* d_a;

  GPUErrchk( cudaMalloc( (void **) &d_a, sizeof( domain ) * rows * columns ) );
  GPUErrchk( cudaMemcpy( d_a, host_a, sizeof( domain ) * rows * columns, cudaMemcpyHostToDevice ) );
  AdditionKernel<<<num_blocks, block_size>>>( d_a, d_b, d_a, rows, columns );
  GPUErrchk( cudaMemcpy( host_result, d_a, sizeof( domain ) * rows * columns, cudaMemcpyDeviceToHost ) );
}

template<class domain>
void GPUDifference( domain* host_a, const domain* d_b, domain* host_result, int rows, int columns )
{
  int block_size = 1024;
  int num_blocks = ( ( rows * columns ) / block_size ) + 1;

  // Allocate memory space on the device 
  domain* d_a;

  GPUErrchk( cudaMalloc( (void **) &d_a, sizeof( domain ) * rows * columns ) );

  // copy matrix A from host to device memory
  GPUErrchk( cudaMemcpy( d_a, host_a, sizeof( domain ) * rows * columns, cudaMemcpyHostToDevice ) );

  DifferenceKernel<<<num_blocks, block_size>>>( d_a, d_b, d_a, rows, columns );
  
  GPUErrchk( cudaMemcpy( host_result, d_a, sizeof( domain ) * rows * columns, cudaMemcpyDeviceToHost ) );

  // free memory
  GPUErrchk( cudaFree( d_a ) );
}

template<class domain>
void GPUMultiplication( domain* host_a, const domain* d_b, domain* host_result, int a_rows, int a_columns, int b_columns )
{
  dim3 block_dim( 32, 32 );
  dim3 grid_dim( (a_rows / 32) + 1, (b_columns / 32) + 1 );

  // Allocate memory space on the device 
  domain* d_a;
  domain* d_c;

  GPUErrchk( cudaMalloc( (void **) &d_a, sizeof( domain ) * a_rows * a_columns ) );
  GPUErrchk( cudaMalloc( (void **) &d_c, sizeof( domain ) * a_rows * b_columns ) );

  // copy matrix A and B from host to device memory
  GPUErrchk( cudaMemcpy( d_a, host_a, sizeof( domain ) * a_rows * a_columns, cudaMemcpyHostToDevice ) );

  MultiplicationKernel<<<grid_dim, block_dim>>>( d_a, d_b, d_c, a_rows, a_columns, b_columns );

  GPUErrchk( cudaMemcpy( host_result, d_c, sizeof( domain ) * a_rows * b_columns, cudaMemcpyDeviceToHost ) );

  // free memory
  GPUErrchk( cudaFree( d_a ) );
  GPUErrchk( cudaFree( d_c ) );
}

template<class domain>
void GPUMultiplication( domain* host_a, const domain host_b, domain* host_result, int rows, int columns )
{
  dim3 block_dim( 32, 32 );
  dim3 grid_dim( (rows / 32) + 1, (columns / 32) + 1 );

  // Allocate memory space on the device 
  domain* d_a;
  domain* d_c;

  GPUErrchk( cudaMalloc( (void **) &d_a, sizeof( domain ) * rows * columns ) );
  GPUErrchk( cudaMalloc( (void **) &d_c, sizeof( domain ) * rows * columns ) );

  // copy matrix A and B from host to device memory
  GPUErrchk( cudaMemcpy( d_a, host_a, sizeof( domain ) * rows * columns, cudaMemcpyHostToDevice ) );

  MultiplicationKernel<<<grid_dim, block_dim>>>( d_a, host_b, d_c, rows, columns );

  GPUErrchk( cudaMemcpy( host_result, d_c, sizeof( domain ) * rows * columns, cudaMemcpyDeviceToHost ) );
  cudaDeviceSynchronize();
  
  // free memory
  GPUErrchk( cudaFree( d_a ) );
  GPUErrchk( cudaFree( d_c ) );
}

template<class domain>
void GPUHadamard( domain* host_a, const domain* d_b, domain* host_result, int rows, int columns )
{
  int block_size = 1024;
  int num_blocks = ( ( rows * columns ) / block_size ) + 1;

  // Allocate memory space on the device 
  domain* d_a;

  GPUErrchk( cudaMalloc( (void **) &d_a, sizeof( domain ) * rows * columns ) );

  // copy matrix A and B from host to device memory
  GPUErrchk( cudaMemcpy( d_a, host_a, sizeof( domain ) * rows * columns, cudaMemcpyHostToDevice ) );

  HadamardKernel<<<num_blocks, block_size>>>( d_a, d_b, d_a, rows, columns );
  
  GPUErrchk( cudaMemcpy( host_result, d_a, sizeof( domain ) * rows * columns, cudaMemcpyDeviceToHost ) );
  cudaDeviceSynchronize();

  // free memory
  GPUErrchk( cudaFree( d_a ) );
}

template<class domain>
void GPUAssign( const domain* host_a, domain** d_a, int rows, int columns )
{
  // Allocate memory space on the device 
  GPUErrchk( cudaMalloc( (void **) d_a, sizeof( domain ) * rows * columns ) );
  // copy matrix A from host to device memory
  GPUErrchk( cudaMemcpy( *d_a, host_a, sizeof( domain ) * rows * columns, cudaMemcpyHostToDevice ) );
}

template<class domain>
void GPUCopy( const domain* host_a, domain* d_a, int rows, int columns )
{
  // copy matrix A from host to device memory
  GPUErrchk( cudaMemcpy( d_a, host_a, sizeof( domain ) * rows * columns, cudaMemcpyHostToDevice ) );
}

template<class domain>
void GPUFree( domain* device_data )
{
  GPUErrchk( cudaFree( device_data ) );
}

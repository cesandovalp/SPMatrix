#pragma once

#include <cstdio>

__global__
void TestKernel( double*, double*, double*, unsigned );

template<class domain>
__global__
void AdditionKernel( domain*, domain*, domain*, int, int );

template<class domain>
__global__
void MultiplicationKernel( domain*, domain*, domain*, int, int, int );

template<class domain>
__global__
void AdditionKernel( domain* a, domain* b, domain* result, int rows, int columns )
{
  int start     = blockIdx.x * blockDim.x + threadIdx.x;
  int increment = blockDim.x * gridDim.x;
  
  for( int index = start; index < rows * columns; index += increment ) 
    result[index] = a[index] + b[index];
}

template<class domain>
__global__
void MultiplicationKernel( domain* a, domain* b, domain* result, int a_rows, int a_columns, int b_columns )
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
void GPUAddition( domain* host_a, domain* host_b, domain* host_result, int rows, int columns )
{
  int block_size = 512;
  int num_blocks = ( rows * columns + block_size - 1 ) / block_size;

  float gpu_elapsed_time_ms;

  // some events to count the execution time
  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  // start to count execution time of GPU version
  cudaEventRecord( start, 0 );
  // Allocate memory space on the device 
  domain* d_a;
  domain* d_b;
  domain* d_c;

  cudaMalloc( (void **) &d_a, sizeof( domain ) * rows * columns );
  cudaMalloc( (void **) &d_b, sizeof( domain ) * rows * columns );
  cudaMalloc( (void **) &d_c, sizeof( domain ) * rows * columns);

  // copy matrix A and B from host to device memory
  cudaMemcpy( d_a, host_a, sizeof( domain ) * rows * columns, cudaMemcpyHostToDevice );
  cudaMemcpy( d_b, host_b, sizeof( domain ) * rows * columns, cudaMemcpyHostToDevice );

  AdditionKernel<<<num_blocks, block_size>>>( d_a, d_b, d_c, rows, columns );
  
  cudaMemcpy( host_result, d_c, sizeof( domain ) * rows * columns, cudaMemcpyDeviceToHost );
  cudaThreadSynchronize();
  // time counting terminate
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );

  // compute time elapse on GPU computing
  cudaEventElapsedTime( &gpu_elapsed_time_ms, start, stop );
  //printf( "Time elapsed on matrix addition of %dx%d on GPU: %f ms.\n\n", rows, columns, gpu_elapsed_time_ms );
  
  // free memory
  cudaFree( d_a );
  cudaFree( d_b );
  cudaFree( d_c );
}

template<class domain>
void GPUMultiplication( domain* host_a, domain* host_b, domain* host_result, int a_rows, int a_columns, int b_columns )
{
  int block_size = 512;
  int num_blocks = ( a_rows * b_columns + block_size - 1 ) / block_size;

  float gpu_elapsed_time_ms;

  // some events to count the execution time
  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  // start to count execution time of GPU version
  cudaEventRecord( start, 0 );
  // Allocate memory space on the device 
  domain* d_a;
  domain* d_b;
  domain* d_c;

  cudaMalloc( (void **) &d_a, sizeof( domain ) * rows * columns );
  cudaMalloc( (void **) &d_b, sizeof( domain ) * rows * columns );
  cudaMalloc( (void **) &d_c, sizeof( domain ) * rows * columns);

  // copy matrix A and B from host to device memory
  cudaMemcpy( d_a, host_a, sizeof( domain ) * rows * columns, cudaMemcpyHostToDevice );
  cudaMemcpy( d_b, host_b, sizeof( domain ) * rows * columns, cudaMemcpyHostToDevice );

  MultiplicationKernel<<<num_blocks, block_size>>>( d_a, d_b, d_c, a_rows, a_columns, b_columns );
  
  cudaMemcpy( host_result, d_c, sizeof( domain ) * rows * columns, cudaMemcpyDeviceToHost );
  cudaThreadSynchronize();
  // time counting terminate
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );

  // compute time elapse on GPU computing
  cudaEventElapsedTime( &gpu_elapsed_time_ms, start, stop );
  //printf( "Time elapsed on matrix addition of %dx%d on GPU: %f ms.\n\n", rows, columns, gpu_elapsed_time_ms );
  
  // free memory
  cudaFree( d_a );
  cudaFree( d_b );
  cudaFree( d_c );
}

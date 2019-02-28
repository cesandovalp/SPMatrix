#pragma once

#include <cstdio>

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
void GPUAddition( domain* host_a, const domain* host_b, domain* host_result, int rows, int columns )
{
  int block_size = 1024;
  int num_blocks = ( ( rows * columns ) / block_size ) + 1;

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
  cudaDeviceSynchronize();
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
void GPUDifference( domain* host_a, const domain* host_b, domain* host_result, int rows, int columns )
{
  int block_size = 1024;
  int num_blocks = ( ( rows * columns ) / block_size ) + 1;

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

  DifferenceKernel<<<num_blocks, block_size>>>( d_a, d_b, d_c, rows, columns );
  
  cudaMemcpy( host_result, d_c, sizeof( domain ) * rows * columns, cudaMemcpyDeviceToHost );
  cudaDeviceSynchronize();
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
void GPUMultiplication( domain* host_a, const domain* host_b, domain* host_result, int a_rows, int a_columns, int b_columns )
{
  dim3 block_dim( 32, 32 );
  dim3 grid_dim( (a_rows / 32) + 1, (b_columns / 32) + 1 );

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

  cudaMalloc( (void **) &d_a, sizeof( domain ) * a_rows * a_columns );
  cudaMalloc( (void **) &d_b, sizeof( domain ) * a_columns * b_columns );
  cudaMalloc( (void **) &d_c, sizeof( domain ) * a_rows * b_columns );

  // copy matrix A and B from host to device memory
  cudaMemcpy( d_a, host_a, sizeof( domain ) * a_rows * a_columns, cudaMemcpyHostToDevice );
  cudaMemcpy( d_b, host_b, sizeof( domain ) * a_columns * b_columns, cudaMemcpyHostToDevice );

  MultiplicationKernel<<<grid_dim, block_dim>>>( d_a, d_b, d_c, a_rows, a_columns, b_columns );

  cudaMemcpy( host_result, d_c, sizeof( domain ) * a_rows * b_columns, cudaMemcpyDeviceToHost );
  cudaDeviceSynchronize();
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
void GPUMultiplication( domain* host_a, const domain host_b, domain* host_result, int rows, int columns )
{
  dim3 block_dim( 32, 32 );
  dim3 grid_dim( (rows / 32) + 1, (columns / 32) + 1 );

  float gpu_elapsed_time_ms;

  // some events to count the execution time
  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  // start to count execution time of GPU version
  cudaEventRecord( start, 0 );
  // Allocate memory space on the device 
  domain* d_a;
  domain* d_c;

  cudaMalloc( (void **) &d_a, sizeof( domain ) * rows * columns );
  cudaMalloc( (void **) &d_c, sizeof( domain ) * rows * columns );

  // copy matrix A and B from host to device memory
  cudaMemcpy( d_a, host_a, sizeof( domain ) * rows * columns, cudaMemcpyHostToDevice );

  MultiplicationKernel<<<grid_dim, block_dim>>>( d_a, host_b, d_c, rows, columns );

  cudaMemcpy( host_result, d_c, sizeof( domain ) * rows * columns, cudaMemcpyDeviceToHost );
  cudaDeviceSynchronize();
  // time counting terminate
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );

  // compute time elapse on GPU computing
  cudaEventElapsedTime( &gpu_elapsed_time_ms, start, stop );
  //printf( "Time elapsed on matrix addition of %dx%d on GPU: %f ms.\n\n", rows, columns, gpu_elapsed_time_ms );
  
  // free memory
  cudaFree( d_a );
  cudaFree( d_c );
}

template<class domain>
void GPUHadamard( domain* host_a, const domain* host_b, domain* host_result, int rows, int columns )
{
  int block_size = 1024;
  int num_blocks = ( ( rows * columns ) / block_size ) + 1;

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

  HadamardKernel<<<num_blocks, block_size>>>( d_a, d_b, d_c, rows, columns );
  
  cudaMemcpy( host_result, d_c, sizeof( domain ) * rows * columns, cudaMemcpyDeviceToHost );
  cudaDeviceSynchronize();
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

#pragma once

#include "MultiCoreMatrixOperations.hpp"

void GPUAddition_( double* host_a, const double* device_b, double* host_result, int rows, int columns );
void GPUAddition_( float*  host_a, const float*  device_b, float*  host_result, int rows, int columns );
void GPUAddition_( int*    host_a, const int*    device_b, int*    host_result, int rows, int columns );

void GPUDifference_( double* host_a, const double* host_b, double* host_result, int rows, int columns );
void GPUDifference_( float*  host_a, const float*  host_b, float*  host_result, int rows, int columns );
void GPUDifference_( int*    host_a, const int*    host_b, int*    host_result, int rows, int columns );

void GPUMultiplication_( double* host_a, const double* host_b, double* host_result, int a_rows, int a_columns, int b_columns );
void GPUMultiplication_( float*  host_a, const float*  host_b, float*  host_result, int a_rows, int a_columns, int b_columns );
void GPUMultiplication_( int*    host_a, const int*    host_b, int*    host_result, int a_rows, int a_columns, int b_columns );

void GPUMultiplication_( double* host_a, const double host_b, double* host_result, int a_rows, int a_columns );
void GPUMultiplication_( float*  host_a, const float  host_b, float*  host_result, int a_rows, int a_columns );
void GPUMultiplication_( int*    host_a, const int    host_b, int*    host_result, int a_rows, int a_columns );

void GPUHadamard_( double* host_a, const double* host_b, double* host_result, int rows, int columns );
void GPUHadamard_( float*  host_a, const float*  host_b, float*  host_result, int rows, int columns );
void GPUHadamard_( int*    host_a, const int*    host_b, int*    host_result, int rows, int columns );

void GPUAssign_( const double* host_a, double** device_a, int rows, int columns );
void GPUAssign_( const float*  host_a, float**  device_a, int rows, int columns );
void GPUAssign_( const int*    host_a, int**    device_a, int rows, int columns );

void GPUCopy_( const double* host_a, double* device_a, int rows, int columns );
void GPUCopy_( const float*  host_a, float*  device_a, int rows, int columns );
void GPUCopy_( const int*    host_a, int*    device_a, int rows, int columns );

void GPUFree_( double** device_data );
void GPUFree_( float**  device_data );
void GPUFree_( int**    device_data );

namespace sp
{
  template<typename domain>
  void GPUAddition( Matrix<domain>* a, const Matrix<domain>& b )
  {
    //GPUAddition_( a->data, b.device_data, a->data, a->rows, a->columns );
  }

  template<typename domain>
  void GPUAddition( Matrix<domain>* a, const std::vector<domain>& b )
  {
    domain* device_data;
    GPUAssign_( b.data(), &device_data, a->rows, a->columns );
    GPUAddition_( a->device_data, device_data, a->data, a->rows, a->columns );
  }

  template<typename domain>
  void GPUDifference( Matrix<domain>* a, const Matrix<domain>& b )
  {
    GPUDifference_( a->data, b.data, a->data, a->rows, a->columns );
  }

  template<typename domain>
  void GPUDifference( Matrix<domain>* a, const std::vector<domain>& b )
  {
    GPUDifference_( a->data, b.data(), a->data, a->rows, a->columns );
  }

  template<typename domain>
  void GPUMultiplication( Matrix<domain>* a, const Matrix<domain>& b )
  {
    Matrix<domain> result( a->rows, b.columns );
    GPUMultiplication_( a->data, b.data, result.data, a->rows, a->columns, b.columns );
    a->Copy( result );
  }

  template<typename domain>
  void GPUMultiplication( Matrix<domain>* a, const std::vector<domain>& b )
  {
    Matrix<domain> result( a->rows, 1 );
    GPUMultiplication_( a->data, b.data(), result.data, a->rows, a->columns, 1 );
    a->Copy( result );
  }

  template<typename domain>
  void GPUMultiplication( Matrix<domain>* a, const domain& b )
  {
    GPUMultiplication_( a->data, b, a->data, a->rows, a->columns );
  }

  template<typename domain>
  void GPUMultiplicationAux( Matrix<domain>* a, const Matrix<domain>& b )
  {
    domain* swap;

    GPUMultiplication_( a->data, b.data, a->tmp, a->rows, a->columns, b.columns );

    swap    = a->data;
    a->data = a->tmp;
    a->tmp  = swap;
    for( unsigned i = 0; i < a->rows * a->columns; ++i )
      a->tmp[i] = 0;
  }

  template<typename domain>
  void GPUHadamard( Matrix<domain>* a, const Matrix<domain>& b )
  {
    GPUHadamard_( a->data, b.data, a->data, a->rows, a->columns );
  }

  template<typename domain>
  void GPUMultiplicationIn( Matrix<domain>* a, const Matrix<domain>& b, const Matrix<domain>& c )
  {
    GPUMultiplication_( b.data, c.data, a->data, b.rows, b.columns, c.columns );
  }

  template<typename domain>
  void GPUApply( Matrix<domain>* a, domain (*f)( domain ) )
  {
    //https://stackoverflow.com/questions/41381254/cuda-c11-array-of-lambdas-function-by-index-not-working?noredirect=1&lq=1
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
        a->data[i * a->columns + j] = f( a->data[i * a->columns + j] );
  }
  
  template<typename domain>
  void GPUCopy( Matrix<domain>* a )
  {
    GPUCopy_( a->data, a->device_data, a->rows, a->columns );
  }

  template<typename domain>
  void SetGPU( Matrix<domain>* m )
  {
    m->negative              = MultiCoreNegative<domain>   ;
    m->transpose             = MultiCoreTranspose<domain>  ;
    m->copy                  = MultiCoreCopy<domain>       ;
    m->sync                  = GPUCopy<domain>             ;
    m->assign                = MultiCoreAssign<domain>     ;
    m->assign_vector         = MultiCoreAssign<domain>     ;
    m->assign_scalar         = MultiCoreAssign<domain>     ;
    m->addition              = GPUAddition<domain>         ;
    m->addition_vector       = GPUAddition<domain>         ;
    m->difference            = GPUDifference<domain>       ;
    m->difference_vector     = GPUDifference<domain>       ;
    m->multiplication        = GPUMultiplication<domain>   ;
    m->multiplication_       = GPUMultiplicationAux<domain>;
    m->hadamard_product      = GPUHadamard<domain>         ;
    m->multiplication_vector = GPUMultiplication<domain>   ;
    m->multiplication_scalar = GPUMultiplication<domain>   ;
    m->multiplication_in     = GPUMultiplicationIn<domain> ;
    m->apply                 = GPUApply<domain>            ;

    GPUAssign_( m->data, &(m->device_data), m->rows, m->columns );
  }
}

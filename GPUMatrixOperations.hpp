#pragma once

#include "Matrix.hpp"

void GPUAddition_( double* host_a, double* host_b, double* host_result, int rows, int columns );
void GPUAddition_( float*  host_a, float*  host_b, float*  host_result, int rows, int columns );
void GPUAddition_( int*    host_a, int*    host_b, int*    host_result, int rows, int columns );

namespace sp
{
  template<typename domain>
  void GPUAddition( Matrix<domain>* a, const Matrix<domain>& b )
  {
    GPUAddition_( a->data, b.data, a->data, a->rows, a->columns );
  }

  template<typename domain>
  void GPUAddition( Matrix<domain>* a, const std::vector<domain>& b )
  {
    for( unsigned i = 0; i < b.size(); ++i )
      a->data[i] += b[i];
  }

  template<typename domain>
  void GPUDifference( Matrix<domain>* a, const Matrix<domain>& b )
  {
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
        a->data[i * a->columns + j] -= b.data[i * a->columns + j];
  }

  template<typename domain>
  void GPUDifference( Matrix<domain>* a, const std::vector<domain>& b )
  {
    for( unsigned i = 0; i < b.size(); ++i )
      a->data[i] -= b[i];
  }

  template<typename domain>
  void GPUMultiplication( Matrix<domain>* a, const Matrix<domain>& b )
  {
    Matrix<domain> result( a->rows, b.columns );

    /*for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned k = 0; k < a->columns; ++k )
        for( unsigned j = 0; j < b.columns; ++j )
          result.data[i * b.columns + j] += a->data[i * a->columns + k] * b.data[k * b.columns + j];*/

    a->Copy( result );
  }

  template<typename domain>
  void GPUMultiplication( Matrix<domain>* a, const std::vector<domain>& b )
  {
    Matrix<domain> result( a->rows, 1 );

    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned k = 0; k < a->columns; ++k )
        result.data[i] += a->data[i * a->columns + k] * b[k];

    a->Copy( result );
  }

  template<typename domain>
  void GPUMultiplication( Matrix<domain>* a, const domain& b )
  {
    for( unsigned i = 0; i < a->rows * a->columns; ++i )
      a->data[i] *= b;
  }

  template<typename domain>
  void GPUMultiplication_( Matrix<domain>* a, const Matrix<domain>& b )
  {
    domain* swap;
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned k = 0; k < a->columns; ++k )
        for( unsigned j = 0; j < b.columns; ++j )
          a->tmp[i * b.columns + j] += a->data[i * a->columns + k] * b.data[k * b.columns + j];

    swap    = a->data;
    a->data = a->tmp;
    a->tmp  = swap;
    for( unsigned i = 0; i < a->rows * a->columns; ++i )
      a->tmp[i] = 0;
  }

  template<typename domain>
  void GPUHadamard( Matrix<domain>* a, const Matrix<domain>& b )
  {
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
        a->data[i * a->columns + j] *= b.data[i * a->columns + j];
  }

  template<typename domain>
  void GPUMultiplicationIn( Matrix<domain>* a, const Matrix<domain>& b, const Matrix<domain>& c )
  {
    for( unsigned i = 0; i < b.rows; ++i )
      for( unsigned k = 0; k < b.columns; ++k )
        for( unsigned j = 0; j < c.columns; ++j )
          a->data[i * c.columns + j] += b.data[i * b.columns + k] * c.data[k * c.columns + j];
  }

  template<typename domain>
  void GPUNegative( const Matrix<domain>* a, Matrix<domain>& b )
  {
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
        b.data[i * a->columns + j] = -a->data[i * a->columns + j];
  }

  template<typename domain>
  void GPUTranspose( Matrix<domain>* a )
  {
    for( unsigned i = 0; i < a->columns; ++i )
      for( unsigned j = 0; j < a->rows; ++j )
        a->tmp[i * a->rows + j] = a->data[j * a->columns + i];

    domain* tmp = a->tmp;
    a->tmp = a->data;
    a->data = tmp;

    auto rows  = a->rows;
    a->rows    = a->columns;
    a->columns = rows;
  }

  template<typename domain>
  void GPUAssign( Matrix<domain>* a, const Matrix<domain>& b )
  {
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
      {
        a->data[i * a->columns + j] = b.data[i * a->columns + j];
        a->tmp[i * a->columns + j]  = b.tmp[i * a->columns + j];
      }
  }

  template<typename domain>
  void GPUAssign( Matrix<domain>* a, const domain& b )
  {
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
      {
        a->data[i * a->columns + j] = b;
        a->tmp[i * a->columns + j]  = 0;
      }
  }

  template<typename domain>
  void GPUAssign( Matrix<domain>* a, const std::vector<domain>& b )
  {
    a->columns = b.size();
    a->rows    = 1       ;

    for( unsigned i = 0; i < b.size(); ++i )
    {
      a->data[i]   = b[i];
      a->tmp[i]    = 0   ;
    }
  }

  template<typename domain>
  void GPUCopy( Matrix<domain>* a, const Matrix<domain>& b )
  {
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
        a->data[i * a->columns + j] = b.data[i * a->columns + j];
  }

  template<typename domain>
  void GPUApply( Matrix<domain>* a, domain (*f)( domain ) )
  {
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
        a->data[i * a->columns + j] = f( a->data[i * a->columns + j] );
  }

  template<typename domain>
  void SetGPU( Matrix<domain>* m )
  {
    m->negative              = GPUNegative<domain>        ;
    m->transpose             = GPUTranspose<domain>       ;
    m->copy                  = GPUCopy<domain>            ;
    m->assign                = GPUAssign<domain>          ;
    m->assign_vector         = GPUAssign<domain>          ;
    m->assign_scalar         = GPUAssign<domain>          ;
    m->addition              = GPUAddition<domain>        ;
    m->addition_vector       = GPUAddition<domain>        ;
    m->difference            = GPUDifference<domain>      ;
    m->difference_vector     = GPUDifference<domain>      ;
    m->multiplication        = GPUMultiplication<domain>  ;
    m->multiplication_       = GPUMultiplication_<domain> ;
    m->hadamard_product      = GPUHadamard<domain>        ;
    m->multiplication_vector = GPUMultiplication<domain>  ;
    m->multiplication_scalar = GPUMultiplication<domain>  ;
    m->multiplication_in     = GPUMultiplicationIn<domain>;
    m->apply                 = GPUApply<domain>           ;
  }
}

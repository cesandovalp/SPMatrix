#pragma once

#include "Matrix.hpp"
#include <omp.h>

namespace sp
{
  template<typename domain>
  void MultiCoreAddition( Matrix<domain>* a, const Matrix<domain>& b )
  {
    #pragma omp parallel for
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
        a->data[i * a->columns + j] += b.data[i * a->columns + j];
  }
  
  template<typename domain>
  void MultiCoreAddition( Matrix<domain>* a, const std::vector<domain>& b )
  {
    #pragma omp parallel for
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
        a->data[i * a->columns + j] += b[i * a->columns + j];
  }

  template<typename domain>
  void MultiCoreDifference( Matrix<domain>* a, const Matrix<domain>& b )
  {
    #pragma omp parallel for
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
        a->data[i * a->columns + j] -= b.data[i * a->columns + j];
  }

  template<typename domain>
  void MultiCoreDifference( Matrix<domain>* a, const std::vector<domain>& b )
  {
    #pragma omp parallel for
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
        a->data[i * a->columns + j] -= b[i * a->columns + j];
  }

  template<typename domain>
  void MultiCoreMultiplication( Matrix<domain>* a, const Matrix<domain>& b )
  {
    Matrix<domain> result( a->rows, b.columns );

    #pragma omp parallel for
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned k = 0; k < a->columns; ++k )
        for( unsigned j = 0; j < b.columns; ++j )
          result.data[i * b.columns + j] += a->data[i * a->columns + k] * b.data[k * b.columns + j];

    a->Copy( result );
  }

  template<typename domain>
  void MultiCoreMultiplication( Matrix<domain>* a, const std::vector<domain>& b )
  {
    Matrix<domain> result( a->rows, 1 );

    #pragma omp parallel for
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned k = 0; k < a->columns; ++k )
        result.data[i] += a->data[i * a->columns + k] * b[k];

    a->Copy( result );
  }

  template<typename domain>
  void MultiCoreMultiplication( Matrix<domain>* a, const domain& b )
  {
    #pragma omp parallel for
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
        a->data[i * a->columns + j] *= b;
  }

  template<typename domain>
  void MultiCoreMultiplication_( Matrix<domain>* a, const Matrix<domain>& b )
  {
    domain* swap;
    
    #pragma omp parallel for
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned k = 0; k < a->columns; ++k )
        for( unsigned j = 0; j < b.columns; ++j )
          a->tmp[i * b.columns + j] += a->data[i * a->columns + k] * b.data[k * b.columns + j];

    swap    = a->data;
    a->data = a->tmp;
    a->tmp  = swap;

    #pragma omp parallel for
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
        a->tmp[i * a->columns + j] = 0;
  }

  template<typename domain>
  void MultiCoreHadamard( Matrix<domain>* a, const Matrix<domain>& b )
  {
    #pragma omp parallel for
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
        a->data[i * a->columns + j] *= b.data[i * a->columns + j];
  }

  template<typename domain>
  void MultiCoreMultiplicationIn( Matrix<domain>* a, const Matrix<domain>& b, const Matrix<domain>& c )
  {
    #pragma omp parallel for
    for( unsigned i = 0; i < b.rows; ++i )
      for( unsigned k = 0; k < b.columns; ++k )
        for( unsigned j = 0; j < c.columns; ++j )
          a->data[i * c.columns + j] += b.data[i * b.columns + k] * c.data[k * c.columns + j];
  }

  template<typename domain>
  void MultiCoreNegative( const Matrix<domain>* a, Matrix<domain>& b )
  {
    #pragma omp parallel for
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
        b.data[i * a->columns + j] = -a->data[i * a->columns + j];
  }

  template<typename domain>
  void MultiCoreTranspose( Matrix<domain>* a )
  {
    #pragma omp parallel for
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
  void MultiCoreAssign( Matrix<domain>* a, const Matrix<domain>& b )
  {
    #pragma omp parallel for
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
      {
        a->data[i * a->columns + j] = b.data[i * a->columns + j];
        a->tmp[i * a->columns + j]  = b.tmp[i * a->columns + j];
      }
  }

  template<typename domain>
  void MultiCoreAssign( Matrix<domain>* a, const domain& b )
  {
    #pragma omp parallel for
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
      {
        a->data[i * a->columns + j] = b;
        a->tmp[i * a->columns + j]  = 0;
      }
  }

  template<typename domain>
  void MultiCoreAssign( Matrix<domain>* a, const std::vector<domain>& b )
  {
    a->columns = b.size();
    a->rows    = 1       ;

    #pragma omp parallel for
    for( unsigned i = 0; i < b.size(); ++i )
    {
      a->data[i]   = b[i];
      a->tmp[i]    = 0   ;
    }
  }

  template<typename domain>
  void MultiCoreCopy( Matrix<domain>* a, const Matrix<domain>& b )
  {
    #pragma omp parallel for
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
        a->data[i * a->columns + j] = b.data[i * a->columns + j];
  }

  template<typename domain>
  void MultiCoreApply( Matrix<domain>* a, domain (*f)( domain ) )
  {
    #pragma omp parallel for
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
        a->data[i * a->columns + j] = f( a->data[i * a->columns + j] );
  }

  template<typename domain>
  void SetMultiCore( Matrix<domain>* m )
  {
    m->negative              = MultiCoreNegative<domain>        ;
    m->transpose             = MultiCoreTranspose<domain>       ;
    m->copy                  = MultiCoreCopy<domain>            ;
    m->assign                = MultiCoreAssign<domain>          ;
    m->assign_vector         = MultiCoreAssign<domain>          ;
    m->assign_scalar         = MultiCoreAssign<domain>          ;
    m->addition              = MultiCoreAddition<domain>        ;
    m->addition_vector       = MultiCoreAddition<domain>        ;
    m->difference            = MultiCoreDifference<domain>      ;
    m->difference_vector     = MultiCoreDifference<domain>      ;
    m->multiplication        = MultiCoreMultiplication<domain>  ;
    m->multiplication_       = MultiCoreMultiplication_<domain> ;
    m->hadamard_product      = MultiCoreHadamard<domain>        ;
    m->multiplication_vector = MultiCoreMultiplication<domain>  ;
    m->multiplication_scalar = MultiCoreMultiplication<domain>  ;
    m->multiplication_in     = MultiCoreMultiplicationIn<domain>;
    m->apply                 = MultiCoreApply<domain>           ;
  }
}

#pragma once

#include "Matrix.hpp"

namespace sp
{
  template<typename domain>
  void SerialAddition( Matrix<domain>* a, const Matrix<domain>& b )
  {
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
        a->data[i * a->columns + j] += b.data[i * a->columns + j];
  }

  template<typename domain>
  void SerialAddition( Matrix<domain>* a, const std::vector<domain>& b )
  {
    for( unsigned i = 0; i < b.size(); ++i )
      a->data[i] += b[i];
  }

  template<typename domain>
  void SerialDifference( Matrix<domain>* a, const Matrix<domain>& b )
  {
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
        a->data[i * a->columns + j] -= b.data[i * a->columns + j];
  }

  template<typename domain>
  void SerialDifference( Matrix<domain>* a, const std::vector<domain>& b )
  {
    for( unsigned i = 0; i < b.size(); ++i )
      a->data[i] -= b[i];
  }

  template<typename domain>
  void SerialMultiplication( Matrix<domain>* a, const Matrix<domain>& b )
  {
    Matrix<domain> result( a->rows, b.columns );

    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned k = 0; k < a->columns; ++k )
        for( unsigned j = 0; j < b.columns; ++j )
          result.data[i * b.columns + j] += a->data[i * a->columns + k] * b.data[k * b.columns + j];

    a->Copy( result );
  }

  template<typename domain>
  void SerialMultiplication( Matrix<domain>* a, const std::vector<domain>& b )
  {
    Matrix<domain> result( a->rows, 1 );

    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned k = 0; k < a->columns; ++k )
        result.data[i] += a->data[i * a->columns + k] * b[k];

    a->Copy( result );
  }

  template<typename domain>
  void SerialMultiplication( Matrix<domain>* a, const domain& b )
  {
    for( unsigned i = 0; i < a->rows * a->columns; ++i )
      a->data[i] *= b;
  }

  template<typename domain>
  void SerialMultiplication_( Matrix<domain>* a, const Matrix<domain>& b )
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
  void SerialHadamard( Matrix<domain>* a, const Matrix<domain>& b )
  {
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
        a->data[i * a->columns + j] *= b.data[i * a->columns + j];
  }

  template<typename domain>
  void SerialMultiplicationIn( Matrix<domain>* a, const Matrix<domain>& b, const Matrix<domain>& c )
  {
    for( unsigned i = 0; i < b.rows; ++i )
      for( unsigned k = 0; k < b.columns; ++k )
        for( unsigned j = 0; j < c.columns; ++j )
          a->data[i * c.columns + j] += b.data[i * b.columns + k] * c.data[k * c.columns + j];
  }

  template<typename domain>
  void SerialNegative( const Matrix<domain>* a, Matrix<domain>& b )
  {
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
        b.data[i * a->columns + j] = -a->data[i * a->columns + j];
  }

  template<typename domain>
  void SerialTranspose( Matrix<domain>* a )
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
  void SerialAssign( Matrix<domain>* a, const Matrix<domain>& b )
  {
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
      {
        a->data[i * a->columns + j] = b.data[i * a->columns + j];
        a->tmp[i * a->columns + j]  = b.tmp[i * a->columns + j];
      }
  }

  template<typename domain>
  void SerialAssign( Matrix<domain>* a, const domain& b )
  {
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
      {
        a->data[i * a->columns + j] = b;
        a->tmp[i * a->columns + j]  = 0;
      }
  }

  template<typename domain>
  void SerialAssign( Matrix<domain>* a, const std::vector<domain>& b )
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
  void SerialCopy( Matrix<domain>* a, const Matrix<domain>& b )
  {
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
        a->data[i * a->columns + j] = b.data[i * a->columns + j];
  }

  template<typename domain>
  void SerialApply( Matrix<domain>* a, domain (*f)( domain ) )
  {
    for( unsigned i = 0; i < a->rows; ++i )
      for( unsigned j = 0; j < a->columns; ++j )
        a->data[i * a->columns + j] = f( a->data[i * a->columns + j] );
  }

  template<typename domain>
  void SetSerial( Matrix<domain>* m )
  {
    m->negative              = SerialNegative<domain>        ;
    m->transpose             = SerialTranspose<domain>       ;
    m->copy                  = SerialCopy<domain>            ;
    m->assign                = SerialAssign<domain>          ;
    m->assign_vector         = SerialAssign<domain>          ;
    m->assign_scalar         = SerialAssign<domain>          ;
    m->addition              = SerialAddition<domain>        ;
    m->addition_vector       = SerialAddition<domain>        ;
    m->difference            = SerialDifference<domain>      ;
    m->difference_vector     = SerialDifference<domain>      ;
    m->multiplication        = SerialMultiplication<domain>  ;
    m->multiplication_       = SerialMultiplication_<domain> ;
    m->hadamard_product      = SerialHadamard<domain>        ;
    m->multiplication_vector = SerialMultiplication<domain>  ;
    m->multiplication_scalar = SerialMultiplication<domain>  ;
    m->multiplication_in     = SerialMultiplicationIn<domain>;
    m->apply                 = SerialApply<domain>           ;
  }
}

#pragma once

#include <vector>
#include <iostream>
#include <functional>

namespace sp
{
  template<typename domain>
  class Matrix
  {
    using MatrixOperator   = void(*)( const Matrix*, Matrix& );
    using SelfOperation    = void(*)( Matrix* );
    using MatrixOperation  = void(*)( Matrix*, const Matrix& );
    using VectorOperation  = void(*)( Matrix*, const std::vector<domain>& );
    using ScalarOperation  = void(*)( Matrix*, const domain& );
    using InPlaceOperation = void(*)( Matrix*, const Matrix&, const Matrix& );
    using FunctorOperation = void(*)( Matrix*, domain (*)( domain ) );

    public:
      unsigned rows, columns;

      domain*          data        = 0      ;
      domain*          tmp         = 0      ;
      domain*          device_data = 0      ;
      MatrixOperator   negative             ;
      SelfOperation    transpose            ;
      MatrixOperation  copy                 ;
      SelfOperation    sync                 ;
      MatrixOperation  assign               ;
      MatrixOperation  addition             ;
      MatrixOperation  difference           ;
      MatrixOperation  multiplication       ;
      MatrixOperation  multiplication_      ;
      MatrixOperation  hadamard_product     ;
      VectorOperation  assign_vector        ;
      VectorOperation  addition_vector      ;
      VectorOperation  difference_vector    ;
      VectorOperation  multiplication_vector;
      ScalarOperation  assign_scalar        ;
      ScalarOperation  addition_scalar      ;
      ScalarOperation  difference_scalar    ;
      ScalarOperation  multiplication_scalar;
      InPlaceOperation multiplication_in    ;
      FunctorOperation apply                ;

      Matrix()
      {
        sync = []( Matrix* a ){ };
        rows    = 0;
        columns = 0;
      }

      ~Matrix()
      {
        delete[] data;
        delete[] tmp;
      }

      Matrix( int rows, int columns ) : rows(rows), columns(columns)
      {
        sync = []( Matrix* a ){ };
        data = new domain[rows * columns];
        tmp  = new domain[rows * columns];

        for( int i = 0; i < rows * columns; ++i )
        {
          data[i] = 0;
          tmp[i]  = 0;
        }
      }

      Matrix( const Matrix<domain>& other )
      {
        {
          negative              = other.negative             ;
          transpose             = other.transpose            ;
          copy                  = other.copy                 ;
          sync                  = other.sync                 ;
          assign                = other.assign               ;
          assign_vector         = other.assign_vector        ;
          assign_scalar         = other.assign_scalar        ;
          addition              = other.addition             ;
          addition_vector       = other.addition_vector      ;
          addition_scalar       = other.addition_scalar      ;
          difference            = other.difference           ;
          difference_vector     = other.difference_vector    ;
          difference_scalar     = other.difference_scalar    ;
          multiplication        = other.multiplication       ;
          multiplication_       = other.multiplication_      ;
          hadamard_product      = other.hadamard_product     ;
          multiplication_vector = other.multiplication_vector;
          multiplication_scalar = other.multiplication_scalar;
          multiplication_in     = other.multiplication_in    ;
          apply                 = other.apply                ;
        }

        rows    = other.rows;
        columns = other.columns;
        data    = new domain[other.rows * other.columns];
        tmp     = new domain[other.rows * other.columns];
        std::copy( other.data  , other.data   + ( other.rows * other.columns ), data )  ;
        std::copy( other.tmp   , other.tmp    + ( other.rows * other.columns ), tmp )   ;
      }

      void SetSize( int rows, int columns, domain value = 0 )
      {
        this->rows    = rows;
        this->columns = columns;

        data   = new domain[rows * columns];
        tmp    = new domain[rows * columns];

        for(int i = 0; i < rows * columns; ++i)
        {
          data[i]   = value;
          tmp[i]    = value;
        }
      }

      Matrix( const std::vector<domain>& m, bool row = true )
      {
        if( row )
        {
          columns = 1;
          rows    = m.size();
        }
        else
        {
          columns = m.size();
          rows    = 1;
        }
        data   = new domain[m.size()];
        tmp    = new domain[m.size()];
        for( unsigned i = 0; i < m.size(); ++i )
        {
          data[i]   = m[i];
          tmp[i]    = 0;
        }
      }

      domain* operator[]( int row ) const
      {
        return data + ( row * columns );
      }

      domain* operator()( int index ) const
      {
        return data + index;
      }

      Matrix<domain>& operator+=( const Matrix<domain>& b )
      {
        addition( this, b );
        return *this;
      }

      Matrix<domain> operator+( const Matrix<domain>& b )
      {
        auto result = *this;
        //std::cout << "Result memory address: " << &result << std::endl;
        result += b;
        return result;
      }

      Matrix<domain>& operator-=( const Matrix<domain>& b )
      {
        difference( this, b );
        return *this;
      }

      Matrix<domain> operator-( const Matrix<domain>& b )
      {
        auto result = *this;
        result -= b;
        return result;
      }

      Matrix<domain>& operator-=( const std::vector<domain>& b )
      {
        difference_vector( this, b );
        return *this;
      }

      Matrix<domain>& operator*=( const Matrix<domain>& b )
      {
        multiplication( this, b );
        return *this;
      }

      Matrix<domain>& operator*=( const std::vector<domain>& b )
      {
        multiplication_vector( this, b );
        return *this;
      }

      Matrix<domain>& operator*=( const domain& a )
      {
        multiplication_scalar( this, a );
        return *this;
      }

      Matrix<domain> operator*( const Matrix<domain>& b )
      {
        auto result = *this;
        //std::cout << "Result memory address: " << &result << std::endl;
        result *= b;

        return result;
      }

      Matrix<domain> operator*( const std::vector<domain>& b )
      {
        auto result = *this;
        //std::cout << "Result memory address: " << &result << std::endl;
        result *= b;

        return result;
      }

      Matrix<domain> operator*( const domain& a )
      {
        auto result = *this;
        result *= a;

        return result;
      }

      // Use only when dim(A*B) = dim(A)
      void Multiplication( const Matrix<domain>& b )
      {
        multiplication_( this, b );
      }

      void HadamardProduct( const Matrix<domain>& b )
      {
        hadamard_product( this, b );
      }

      void Multiplication( const Matrix<domain>& a, const Matrix<domain>& b )
      {
        multiplication_in( this, a, b );
      }

      Matrix<domain> operator-() const
      {
        Matrix<domain> result( rows, columns );
        negative( this, result );
        return result;
      }

      Matrix<domain> operator!() const
      {
        auto result = *this;
        result.Transpose();
        return result;
      }

      void Transpose( )
      {
        transpose( this );
      }

      Matrix<domain>& operator=( const Matrix<domain>& other )
      {
        delete[] data;
        delete[] tmp;

        data   = new domain[other.rows * other.columns];
        tmp    = new domain[other.rows * other.columns];

        assign( this, other );

        rows    = other.rows;
        columns = other.columns;

        return *this;
      }

      Matrix<domain>& operator=( const domain& other )
      {
        assign_scalar( this, other );
        return *this;
      }

      void Copy( const std::vector<domain>& m )
      {
        assign_vector( this, m );
      }

      void Copy( domain* m )
      {
        for( unsigned i = 0; i < rows * columns; ++i )
        {
          data[i] = m[i];
          tmp[i]  = 0   ;
        }
      }

      void Copy( const Matrix<domain>& other )
      {
        columns = other.columns;
        rows    = other.rows   ;

        delete[] data;
        delete[] tmp ;

        data   = new domain[rows * columns];
        tmp    = new domain[rows * columns];

        copy( this, other );
      }

      void operator()( const Matrix<domain>& other )
      {
        std::copy( other.data, other.data + ( other.rows * other.columns ), data );
      }

      Matrix<domain>& operator()( domain (*f)( domain ) )
      {
        apply( this, f );
        return *this;
      }

      Matrix<domain> Apply( domain (*f)( domain ) )
      {
        auto result = *this;
        result(f);

        return result;
      }

      std::vector<domain> ToVector()
      {
        auto result = std::vector<domain>( data, data + rows * columns );
        return result;
      }

      void ToVector( std::vector<domain>& t )
      {
        t.assign( data, data + (rows * columns) );
      }

      void Initialize( int rows, int columns )
      {
        data   = new domain[rows * columns];
        tmp    = new domain[rows * columns];

        this->rows    = rows;
        this->columns = columns;
      }

      domain SumElements()
      {
        domain result = 0;
        for( unsigned i = 0; i < rows * columns; ++i )
          result += data[i];
        return result;
      }
      
      void Syncronize()
      {
        sync( this );
      }

      friend std::ostream& operator<< ( std::ostream& stream, const Matrix<domain>& m )
      {
        for( unsigned j = 0; j < m.rows; ++j )
        {
          for( unsigned k = 0; k < m.columns; ++k )
            stream << m.data[ j * m.columns + k ] << '|';
          stream << '\n';
        }
        return stream;
      }
  };
}

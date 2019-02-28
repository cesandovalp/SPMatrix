#pragma once

#include "SerialMatrixOperations.hpp"
#include "MultiCoreMatrixOperations.hpp"
#include "GPUMatrixOperations.hpp"

namespace sp
{
  enum Mode { sequential, multicore, gpu };
  class MatrixFactory
  {
    private:
      static MatrixFactory* _instance;
      Mode mode;

      MatrixFactory( );

    public:
      static MatrixFactory* instance(  );
      void SetMode( const Mode& );

      template<class T>
      Matrix<T> CreateMatrix( const unsigned& rows, const unsigned& columns )
      {
        auto result = Matrix<T>( rows, columns );
        switch( mode )
        {
          case sequential: sp::SetSerial( &result )   ; break;
          case multicore : sp::SetMultiCore( &result ); break;
          case gpu       : sp::SetGPU( &result )      ; break;
        }
        return result;
      }

      template<class T>
      Matrix<T>* CreateMatrixPtr( const unsigned& rows, const unsigned& columns )
      {
        auto result = new Matrix<T>( rows, columns );
        switch( mode )
        {
          case sequential: sp::SetSerial( result )   ; break;
          case multicore : sp::SetMultiCore( result ); break;
          case gpu       : sp::SetGPU( result )      ; break;
        }
        return result;
      }

      template<class T>
      Matrix<T> CreateMatrix( const std::vector<double>& v )
      {
        auto result = Matrix<double>( v );
        switch( mode )
        {
          case sequential: sp::SetSerial( &result )   ; break;
          case multicore : sp::SetMultiCore( &result ); break;
          case gpu       : sp::SetGPU( &result )      ; break;
        }
        return result;
      }

      template<class T>
      Matrix<T>* CreateMatrixArray( const unsigned& size )
      {
        auto result = new Matrix<T>[size];

        for( unsigned i = 0; i < size; ++i )
        {
          switch( mode )
          {
            case sequential: sp::SetSerial( &result[i] )   ; break;
            case multicore : sp::SetMultiCore( &result[i] ); break;
            case gpu       : sp::SetGPU( &result[i] )      ; break;
          }
        }
        return result;
      }

      ~MatrixFactory();
  };
}

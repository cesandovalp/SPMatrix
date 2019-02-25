#include "MatrixFactory.hpp"

namespace sp
{
  MatrixFactory* MatrixFactory::_instance = 0;

  MatrixFactory::MatrixFactory( ) { }

  MatrixFactory* MatrixFactory::instance( )
  {
    if( !_instance )
      _instance = new MatrixFactory( );

    return _instance;
  }

  void MatrixFactory::SetMode( const Mode& mode )
  {
    this->mode = mode;
  }

  MatrixFactory::~MatrixFactory()
  {
    _instance = 0;
  }
}

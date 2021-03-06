#include <iostream>
#include <assert.h>
#include <string>

#include <MatrixFactory.hpp>
#include <SerialMatrixOperations.hpp>
#include <MultiCoreMatrixOperations.hpp>
#include <chrono>

using namespace std::chrono;

using sp::Matrix;

void FillMatrix( Matrix<int>& M )
{
  for( unsigned i = 0; i < M.rows; ++i )
    for( unsigned j = 0; j < M.columns; ++j )
      M.data[i * M.columns + j] = i * M.columns + j;
  M.Syncronize();
}

void TestAddition()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 300, 200 );
  auto B = sp::MatrixFactory::instance()->CreateMatrix<int>( 300, 200 );

  FillMatrix( A );
  FillMatrix( B );

  for( int i = 0; i < 100; ++i )
    auto C = A + B;
}

void TestGPU()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 1, 1 );

  FillMatrix( A );
}

void TestMultiplicationVector()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 200, 400 );
  std::vector<int> B( 400 );
  
  for( int i = 0; i < 400; ++i )
    B[i] = i;

  FillMatrix( A );

  for( int i = 0; i < 100; ++i )
    auto C = A * B;
}

void TestMultiplication()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 300, 300 );
  auto B = sp::MatrixFactory::instance()->CreateMatrix<int>( 300, 300 );

  FillMatrix( A );
  FillMatrix( B );

  for( int i = 0; i < 100; ++i )
    auto C = A * B;
}

void TestMultiplicationNoResize()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 300, 300 );
  auto B = sp::MatrixFactory::instance()->CreateMatrix<int>( 300, 300 );

  FillMatrix( A );
  FillMatrix( B );

  for( int i = 0; i < 100; ++i )
  {
    FillMatrix( A );
    A.Multiplication( B );
  }
}

void TestHadamardProduct()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 300, 500 );
  auto B = sp::MatrixFactory::instance()->CreateMatrix<int>( 300, 500 );

  FillMatrix( A );
  FillMatrix( B );

  for( int i = 0; i < 100; ++i )
  {
    FillMatrix( A );
    A.HadamardProduct( B );
  }
}

void TestMultiplicationIn()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 300, 300 );
  auto B = sp::MatrixFactory::instance()->CreateMatrix<int>( 300, 300 );
  auto C = sp::MatrixFactory::instance()->CreateMatrix<int>( 300, 300 );

  FillMatrix( A );
  FillMatrix( B );

  for( int i = 0; i < 100; ++i )
    C.Multiplication( A, B );
}

void TestMultiplicationScalar()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 400, 400 );

  FillMatrix( A );

  auto B = A;

  for( int i = 0; i < 100; ++i )
    auto C = B * 2;
}

void TestTranspose()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 200, 300 );
  FillMatrix( A );

  for( int i = 0; i < 100; ++i )
    auto B = !A;
}

void TestMinus()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 300, 300 );
  FillMatrix( A );

  for( int i = 0; i < 100; ++i )
    auto B = -A;
}

void TestCopy()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 300, 300 );
  FillMatrix( A );

  for( int i = 0; i < 100; ++i )
    Matrix<int> B = A;
}

void TestApply()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 300, 300 );
  FillMatrix( A );

  auto B = A.Apply( []( int x ) { return x * x; } );

  B = A.Apply( []( int x ) { return 2 * x; } );
  B = A.Apply( []( int x ) { return x < 5 ? 0 : x; } );
  for( int i = 0; i < 100; ++i )
    auto C = !B;
}

void TestToTarget()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 300, 300 );
  FillMatrix( A );
  for( int i = 0; i < 100; ++i )
    auto B = A.ToVector();
}

template<typename TestFunction>
void TestOne( TestFunction test_function, std::string test_name )
{
//  std::cout << "#################################################" << std::endl;
//  std::cout << "Test " << test_name << " started." << std::endl;
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  test_function();
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
//  std::cout << "Test " << test_name << " ended." << std::endl;
  auto duration = duration_cast<microseconds>( t2 - t1 ).count();
  std::cout << "Time " << test_name << ": " << duration * 0.000001 << std::endl;
}

void TestAll( std::string mode )
{
  TestOne( TestAddition               , "Addition "              + mode );
  TestOne( TestMultiplication         , "Multiplication "        + mode );
  TestOne( TestMultiplicationNoResize , "Multiplication Same "   + mode );
  TestOne( TestHadamardProduct        , "Hadamard "              + mode );
  TestOne( TestMultiplicationVector   , "Multiplication Vector " + mode );
  TestOne( TestMultiplicationScalar   , "Multiplication Scalar " + mode );
  TestOne( TestMultiplicationIn       , "Multiplication In "     + mode );
  TestOne( TestTranspose              , "Transpose "             + mode );
  TestOne( TestMinus                  , "Minus "                 + mode );
  TestOne( TestCopy                   , "Copy "                  + mode );
  TestOne( TestApply                  , "Apply "                 + mode );
  TestOne( TestToTarget               , "To "                    + mode );
}

int main()
{
  sp::MatrixFactory::instance()->SetMode( sp::sequential );
  TestAll( "sequential" );
  sp::MatrixFactory::instance()->SetMode( sp::multicore );
  TestAll( "multicore" );
  sp::MatrixFactory::instance()->SetMode( sp::gpu );
  TestGPU();
  TestAll( "gpu" );

  return 0;
}
